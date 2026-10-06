"""Run the isolated v28-style generator_1 anomaly sample-weight experiment.

The platform control has no historical OOF predictions, so this runner rebuilds
the v28 long-g1 recipe inside three recent walk-forward folds. The control and
all variants differ only in the LightGBM sample weight assigned to causally
identified low-fuel training origins. It never creates a submission package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.anomaly_weight_utils import (
    FUELS_G1,
    GALL_FUEL_COLUMNS,
    anomaly_sample_weights,
    build_v28_features,
    causal_low_fuel_ratio,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
    v28_g1_fuel_weight,
)
from src.round2_v3.event_ablation_utils import attach_event_context
from src.round2_v3.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "anomaly_weight_g1_v1.yaml"
)
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "round2_v3" / "registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("anomaly_weight_utils.py")
EVENT_UTILS_SOURCE = RUNNER_SOURCE.with_name("event_ablation_utils.py")
METRICS_SOURCE = RUNNER_SOURCE.with_name("metrics.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _run_identity(config_path: Path, input_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *input_paths]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_anomaly_weight_g1_{digest.hexdigest()[:10]}"


def _append_event(path: Path, event: str, **values: object) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        existing = {record["run_id"] for record in reader}
    if not fieldnames:
        raise RuntimeError("experiment registry has no header")
    if str(row["run_id"]) in existing:
        raise RuntimeError(f"run_id already registered: {row['run_id']}")
    normalized = {field: row.get(field, "") for field in fieldnames}
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(normalized)


def _validate_inputs(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    raw_manifest = _read_json(paths["raw_fingerprints"])
    expected_hashes = {
        item["file"]: item["sha256"] for item in raw_manifest["files"]
    }
    raw_tables = [PROJECT_ROOT / value for value in config["inputs"]["raw_train_tables"]]
    for path in raw_tables:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative not in expected_hashes:
            raise AssertionError(f"raw table absent from fingerprint manifest: {relative}")
        if sha256(path) != expected_hashes[relative]:
            raise AssertionError(f"raw table hash mismatch: {relative}")

    split_manifest = _read_json(paths["split_manifest"])
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    label_manifest = _read_json(paths["label_manifest"])
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("interval label hash mismatch")

    experiment = config["experiment"]
    if experiment["target"] != "generator_1":
        raise AssertionError("v1 anomaly weighting is generator_1 only")
    if experiment["tree_recipe"] != "v28_exact":
        raise AssertionError("tree recipe must stay v28_exact")
    if experiment["fuel_recipe"] != "v28_exact_0.8_near_0.6_far":
        raise AssertionError("fuel recipe differs from frozen v28")
    if experiment["future_process_observations_allowed"]:
        raise AssertionError("future process observations are forbidden")
    if experiment["true_target_history_features_allowed"]:
        raise AssertionError("target history features are forbidden")

    variants = config["variants"]
    multipliers = [float(value) for value in variants["multipliers"]]
    if multipliers != [1.0, 2.0, 3.0, 5.0]:
        raise AssertionError("v1 multiplier ladder must remain [1, 2, 3, 5]")
    if float(variants["primary_multiplier"]) != 3.0:
        raise AssertionError("only multiplier 3 is pre-registered for promotion")
    if variants["select_best_sensitivity_variant"]:
        raise AssertionError("OOF selection among sensitivity weights is forbidden")

    folds = list(config["validation"]["fold_ids"])
    horizons = [int(value) for value in config["horizons_minutes"]]
    expected_models = len(folds) * len(horizons) * len(multipliers)
    if expected_models != int(config["budget"]["expected_model_count"]):
        raise AssertionError("model budget is inconsistent with folds/horizons/variants")
    if horizons != list(range(15, 1441, 15)):
        raise AssertionError("horizon contract must remain 15..1440 by 15 minutes")

    return {
        "raw_tables": raw_tables,
        "raw_manifest": raw_manifest,
        "split_manifest": split_manifest,
        "label_manifest": label_manifest,
        "label_path": label_path,
        "fold_ids": folds,
        "horizons": horizons,
        "multipliers": multipliers,
    }


def _variant(multiplier: float) -> str:
    return "control_weight1" if multiplier == 1.0 else f"anomaly_weight{int(multiplier)}"


def _summarize_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(
        source: pd.DataFrame,
        *,
        label_variant: str,
        period: str,
        scope: str,
        group_columns: list[str],
    ) -> None:
        for keys, part in source.groupby(group_columns, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(group_columns, keys, strict=True))
            records.append(
                {
                    "label_variant": label_variant,
                    "period": period,
                    "scope": scope,
                    "variant": identity.get("variant", "all"),
                    "fold_id": identity.get("fold_id", "all"),
                    "horizon_bucket": identity.get("horizon_bucket", "all"),
                    "origin_regime": identity.get("origin_regime", "all"),
                    "episode_id": identity.get("target_episode_id", "all"),
                    **regression_metrics(part["actual"], part["prediction"]),
                }
            )

    label_columns = {
        "official_interval_mean": "actual_interval_mean",
        "legacy_point": "actual_legacy_point",
    }
    for label_variant, actual_column in label_columns.items():
        for period, maximum_horizon in (("short", 120), ("long", 1440)):
            source = oof.loc[oof["horizon_minutes"] <= maximum_horizon].copy()
            source["actual"] = pd.to_numeric(source[actual_column], errors="coerce")
            source = source.loc[
                np.isfinite(source["actual"]) & np.isfinite(source["prediction"])
            ].copy()
            add(
                source,
                label_variant=label_variant,
                period=period,
                scope="overall",
                group_columns=["variant"],
            )
            add(
                source,
                label_variant=label_variant,
                period=period,
                scope="fold",
                group_columns=["variant", "fold_id"],
            )
            add(
                source,
                label_variant=label_variant,
                period=period,
                scope="horizon_bucket",
                group_columns=["variant", "horizon_bucket"],
            )
            if label_variant == "official_interval_mean" and period == "long":
                add(
                    source,
                    label_variant=label_variant,
                    period=period,
                    scope="origin_regime",
                    group_columns=["variant", "origin_regime"],
                )
                event_rows = source.loc[source["target_in_event"]].copy()
                if not event_rows.empty:
                    add(
                        event_rows,
                        label_variant=label_variant,
                        period=period,
                        scope="episode",
                        group_columns=["variant", "target_episode_id"],
                    )
    return pd.DataFrame.from_records(records)


def _evaluate_gates(
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    control = "control_weight1"
    fold_ids = list(config["validation"]["fold_ids"])
    gate = config["selection_gate"]

    def accuracy(
        variant: str,
        *,
        label: str,
        period: str,
        scope: str,
        fold_id: str = "all",
        bucket: str = "all",
        regime: str = "all",
    ) -> float:
        selected = metrics.loc[
            (metrics["variant"] == variant)
            & (metrics["label_variant"] == label)
            & (metrics["period"] == period)
            & (metrics["scope"] == scope)
            & (metrics["fold_id"].astype(str) == str(fold_id))
            & (metrics["horizon_bucket"].astype(str) == str(bucket))
            & (metrics["origin_regime"].astype(str) == str(regime))
        ]
        return np.nan if selected.empty else float(selected.iloc[0]["accuracy_1_minus_mape"])

    label = "official_interval_mean"
    control_long = accuracy(control, label=label, period="long", scope="overall")
    control_short = accuracy(control, label=label, period="short", scope="overall")
    control_legacy = accuracy(
        control, label="legacy_point", period="long", scope="overall"
    )
    control_ordinary = accuracy(
        control,
        label=label,
        period="long",
        scope="origin_regime",
        regime="ordinary",
    )
    records: list[dict[str, object]] = []
    for multiplier in config["variants"]["multipliers"]:
        if float(multiplier) == 1.0:
            continue
        variant = _variant(float(multiplier))
        long_gain = (
            accuracy(variant, label=label, period="long", scope="overall")
            - control_long
        ) * 100
        short_gain = (
            accuracy(variant, label=label, period="short", scope="overall")
            - control_short
        ) * 100
        legacy_gain = (
            accuracy(
                variant,
                label="legacy_point",
                period="long",
                scope="overall",
            )
            - control_legacy
        ) * 100
        fold_deltas = [
            (
                accuracy(
                    variant,
                    label=label,
                    period="long",
                    scope="fold",
                    fold_id=fold,
                )
                - accuracy(
                    control,
                    label=label,
                    period="long",
                    scope="fold",
                    fold_id=fold,
                )
            )
            * 100
            for fold in fold_ids
        ]
        ordinary_loss = (
            control_ordinary
            - accuracy(
                variant,
                label=label,
                period="long",
                scope="origin_regime",
                regime="ordinary",
            )
        ) * 100
        bucket_deltas: dict[str, float] = {}
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440"):
            bucket_deltas[bucket] = (
                accuracy(
                    variant,
                    label=label,
                    period="long",
                    scope="horizon_bucket",
                    bucket=bucket,
                )
                - accuracy(
                    control,
                    label=label,
                    period="long",
                    scope="horizon_bucket",
                    bucket=bucket,
                )
            ) * 100

        control_episodes = metrics.loc[
            (metrics["label_variant"] == label)
            & (metrics["period"] == "long")
            & (metrics["scope"] == "episode")
            & (metrics["variant"] == control),
            ["episode_id", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
        candidate_episodes = metrics.loc[
            (metrics["label_variant"] == label)
            & (metrics["period"] == "long")
            & (metrics["scope"] == "episode")
            & (metrics["variant"] == variant),
            ["episode_id", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "candidate_accuracy"})
        episode_comparison = control_episodes.merge(
            candidate_episodes,
            on="episode_id",
            how="inner",
            validate="one_to_one",
        )
        episode_fraction = (
            float(
                (
                    episode_comparison["candidate_accuracy"]
                    > episode_comparison["control_accuracy"]
                ).mean()
            )
            if not episode_comparison.empty
            else np.nan
        )

        promotion_eligible = float(multiplier) == float(
            config["variants"]["primary_multiplier"]
        )
        pass_overall = long_gain >= float(
            gate["minimum_component_long_g1_accuracy_gain_pct"]
        )
        pass_folds = min(fold_deltas) >= float(gate["minimum_single_fold_gain_pct"])
        pass_ordinary = ordinary_loss <= float(
            gate["maximum_ordinary_origin_accuracy_loss_pct"]
        )
        pass_episodes = np.isfinite(episode_fraction) and episode_fraction > 0.5
        pass_near_far = min(
            bucket_deltas["h015_120"], bucket_deltas["h735_1440"]
        ) >= -float(gate["maximum_near_far_accuracy_loss_pct"])
        pass_auxiliary = legacy_gain >= float(
            gate["minimum_auxiliary_legacy_point_gain_pct"]
        )
        pass_short = short_gain >= -float(
            gate["maximum_short_accuracy_loss_pct"]
        )
        component_passed = bool(
            promotion_eligible
            and pass_overall
            and pass_folds
            and pass_ordinary
            and pass_episodes
            and pass_near_far
            and pass_auxiliary
            and pass_short
        )
        standalone = bool(
            component_passed
            and long_gain
            >= float(gate["minimum_standalone_long_g1_accuracy_gain_pct"])
        )
        records.append(
            {
                "variant": variant,
                "multiplier": float(multiplier),
                "promotion_eligible": promotion_eligible,
                "official_long_gain_pct": long_gain,
                "official_short_gain_pct": short_gain,
                "legacy_point_long_gain_pct": legacy_gain,
                "ordinary_accuracy_loss_pct": ordinary_loss,
                "single_fold_min_gain_pct": float(min(fold_deltas)),
                "single_fold_mean_gain_pct": float(np.mean(fold_deltas)),
                "episode_improvement_fraction": episode_fraction,
                "evaluated_episode_count": int(len(episode_comparison)),
                "near_gain_pct": bucket_deltas["h015_120"],
                "mid1_gain_pct": bucket_deltas["h135_360"],
                "mid2_gain_pct": bucket_deltas["h375_720"],
                "far_gain_pct": bucket_deltas["h735_1440"],
                "pass_overall": pass_overall,
                "pass_folds": pass_folds,
                "pass_ordinary": pass_ordinary,
                "pass_episodes": pass_episodes,
                "pass_near_far": pass_near_far,
                "pass_auxiliary": pass_auxiliary,
                "pass_short": pass_short,
                "component_gate_passed": component_passed,
                "standalone_submission_eligible": standalone,
            }
        )
    return pd.DataFrame.from_records(records)


def _write_report(
    path: Path,
    run_id: str,
    comparison: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# Isolated g1 anomaly sample weighting",
        "",
        f"- run_id: `{run_id}`",
        "- 角色：scientific OOF；不生成提交包。",
        "- 控制与候选均重建 v28 long-g1；唯一变化是低燃料训练样本权重。",
        "- ×3 是唯一预登记晋级候选；×2/×5 只做稳定性灵敏度检查。",
        "",
        "## 官方区间均值口径",
        "",
        "| period | variant | MAPE | accuracy | delta accuracy (pct) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.period} | `{row.variant}` | {row.mape:.6f} | "
            f"{row.accuracy_1_minus_mape:.6f} | {row.delta_accuracy_pct:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 预登记门禁",
            "",
            "| variant | long gain | worst fold | ordinary loss | episodes | component pass | standalone |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gates.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.official_long_gain_pct:.4f} | "
            f"{row.single_fold_min_gain_pct:.4f} | "
            f"{row.ordinary_accuracy_loss_pct:.4f} | "
            f"{row.episode_improvement_fraction:.1%} | "
            f"{row.component_gate_passed} | "
            f"{row.standalone_submission_eligible} |"
        )
    lines.extend(
        [
            "",
            "通过 component gate 只表示可作为后续组合积木；只有 standalone=true 才允许单独生成平台候选。",
            "任何门禁结果都不会自动创建或上传提交包。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    input_paths = {
        key: PROJECT_ROOT / value
        for key, value in config["inputs"].items()
        if key != "raw_train_tables"
    }
    validated = _validate_inputs(config, input_paths)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"folds={len(validated['fold_ids'])} "
            f"horizons={len(validated['horizons'])} "
            f"variants={len(validated['multipliers'])} "
            f"models={config['budget']['expected_model_count']} "
            "primary=anomaly_weight3 fuel_recipe=v28_exact_0.8_near_0.6_far"
        )
        return

    identity_inputs = [
        *validated["raw_tables"],
        *input_paths.values(),
        RUNNER_SOURCE,
        UTILS_SOURCE,
        EVENT_UTILS_SOURCE,
        METRICS_SOURCE,
    ]
    run_id = _run_identity(config_path, identity_inputs)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    event_log = output_dir / "events.jsonl"
    git_commit = _git_value("rev-parse", "HEAD")
    git_branch = _git_value("branch", "--show-current")
    git_dirty = bool(_git_value("status", "--porcelain"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": config["task"],
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "experiment_version": config["experiment_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "command": {
            "working_directory": str(PROJECT_ROOT),
            "argv": [sys.executable, *sys.argv],
        },
        "git": {"branch": git_branch, "commit": git_commit, "dirty": git_dirty},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "holdout_evaluated": False,
        "submission_eligible": False,
        "external_v29b_oof_available": False,
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in [config_path, *identity_inputs]
        },
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False

    try:
        raw = merge_training_tables(validated["raw_tables"])
        grid = to_v28_grid(raw)
        features, feature_columns = build_v28_features(grid)
        timestamps = pd.DatetimeIndex(grid["datetime"])
        position = pd.Series(np.arange(len(grid), dtype=int), index=timestamps)
        if position.index.duplicated().any():
            raise AssertionError("grid position index is not unique")

        interval_labels = pd.read_csv(validated["label_path"])
        interval_labels["datetime"] = pd.to_datetime(
            interval_labels["datetime"], errors="raise"
        )
        interval_mean = pd.to_numeric(
            interval_labels.set_index("datetime")["generator_1"], errors="coerce"
        )
        split = pd.read_csv(input_paths["split_assignments"])
        for column in ("train_end", "validation_origin"):
            split[column] = pd.to_datetime(split[column], errors="raise")
        split = split.loc[
            (split["role"] == "model_selection")
            & (split["fold_id"].isin(validated["fold_ids"]))
        ].copy()
        observed_folds = split["fold_id"].drop_duplicates().tolist()
        if observed_folds != validated["fold_ids"]:
            raise AssertionError(
                f"fold order/content differs: {observed_folds} != {validated['fold_ids']}"
            )
        episodes = pd.read_csv(input_paths["event_episodes"])

        training_start = pd.Timestamp(config["training"]["g1_tree_start"])
        detector = config["anomaly_detector"]
        model_parameters = dict(config["lightgbm"])
        predictions: list[pd.DataFrame] = []
        fit_records: list[dict[str, object]] = []
        fold_groups = list(split.groupby("fold_id", sort=False))
        for fold_number, (fold_id, fold) in enumerate(fold_groups, start=1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            cutoff_candidates = np.flatnonzero(timestamps <= train_end)
            if len(cutoff_candidates) == 0:
                raise AssertionError(f"no training rows for {fold_id}")
            cutoff_position = int(cutoff_candidates[-1])
            detector_frame = grid.iloc[: cutoff_position + 1].copy()
            detector_coefficients = fit_linear_fuel_proxy(
                detector_frame,
                target="generator_all",
                fuel_columns=GALL_FUEL_COLUMNS,
            )
            ratio = causal_low_fuel_ratio(
                detector_frame,
                detector_coefficients,
                smooth_steps=int(detector["smooth_steps"]),
                baseline_steps=int(detector["baseline_steps"]),
                baseline_min_periods=int(detector["baseline_min_periods"]),
            )
            weight_vectors = {
                float(multiplier): anomaly_sample_weights(
                    ratio,
                    threshold=float(detector["threshold"]),
                    multiplier=float(multiplier),
                )
                for multiplier in validated["multipliers"]
            }

            fuel_training = grid.loc[
                (grid["datetime"] >= training_start)
                & (grid["datetime"] <= train_end)
            ].copy()
            fuel_coefficients = fit_linear_fuel_proxy(
                fuel_training,
                target="generator_1",
                fuel_columns=FUELS_G1,
            )
            validation_origins = pd.DatetimeIndex(
                fold["validation_origin"]
            ).sort_values()
            absent = validation_origins.difference(timestamps)
            if len(absent):
                raise AssertionError(
                    f"{fold_id} has validation origins absent from v28 grid: {len(absent)}"
                )
            validation_positions = position.loc[validation_origins].to_numpy(dtype=int)
            fuel_validation = predict_linear_fuel(
                grid.iloc[validation_positions], FUELS_G1, fuel_coefficients
            )
            print(
                f"[{fold_number}/{len(fold_groups)}] {fold_id}: "
                f"validation_origins={len(validation_origins)} "
                f"low_fuel_rows={(weight_vectors[3.0] > 1).sum()}",
                flush=True,
            )

            for horizon_number, horizon in enumerate(validated["horizons"], start=1):
                step = horizon // 15
                eligible_positions = np.flatnonzero(
                    (timestamps >= training_start)
                    & (np.arange(len(grid)) + step <= cutoff_position)
                )
                target_positions = eligible_positions + step
                y_train = grid["generator_1"].to_numpy(dtype=float)[target_positions]
                valid_train = np.isfinite(y_train)
                eligible_positions = eligible_positions[valid_train]
                y_train = y_train[valid_train]
                if len(y_train) < int(config["training"]["minimum_rows_per_model"]):
                    raise AssertionError(
                        f"{fold_id} h={horizon} has only {len(y_train)} training rows"
                    )

                validation_target_positions = validation_positions + step
                if validation_target_positions.max() >= len(grid):
                    raise AssertionError(f"{fold_id} h={horizon} exceeds label timeline")
                actual_point = grid["generator_1"].to_numpy(dtype=float)[
                    validation_target_positions
                ]
                interval_starts = validation_origins + pd.to_timedelta(
                    horizon - 15, unit="min"
                )
                actual_interval = interval_mean.reindex(interval_starts).to_numpy(
                    dtype=float
                )
                x_train = features.iloc[eligible_positions][feature_columns]
                x_validation = features.iloc[validation_positions][feature_columns]
                fuel_weight = v28_g1_fuel_weight(horizon)
                for multiplier in validated["multipliers"]:
                    multiplier = float(multiplier)
                    variant = _variant(multiplier)
                    model = lgb.LGBMRegressor(**model_parameters)
                    fit_kwargs: dict[str, object] = {}
                    if multiplier != 1.0:
                        fit_kwargs["sample_weight"] = weight_vectors[multiplier][
                            eligible_positions
                        ]
                    model.fit(x_train, y_train, **fit_kwargs)
                    tree_prediction = np.clip(
                        model.predict(x_validation), 0.0, None
                    )
                    prediction = (
                        (1.0 - fuel_weight) * tree_prediction
                        + fuel_weight * fuel_validation
                    )
                    if not np.isfinite(prediction).all():
                        raise AssertionError(
                            f"non-finite prediction for {fold_id} h={horizon} {variant}"
                        )
                    predictions.append(
                        pd.DataFrame(
                            {
                                "variant": variant,
                                "multiplier": multiplier,
                                "fold_id": fold_id,
                                "datetime": validation_origins,
                                "interval_start": interval_starts,
                                "horizon_minutes": horizon,
                                "actual_interval_mean": actual_interval,
                                "actual_legacy_point": actual_point,
                                "prediction": prediction,
                            }
                        )
                    )
                    selected_weights = weight_vectors[multiplier][eligible_positions]
                    fit_records.append(
                        {
                            "fold_id": fold_id,
                            "horizon_minutes": horizon,
                            "variant": variant,
                            "multiplier": multiplier,
                            "training_rows": int(len(y_train)),
                            "upweighted_rows": int((selected_weights > 1).sum()),
                            "upweighted_fraction": float((selected_weights > 1).mean()),
                            "fuel_weight": fuel_weight,
                            "training_origin_start": timestamps[eligible_positions.min()],
                            "training_origin_end": timestamps[eligible_positions.max()],
                        }
                    )
                if horizon_number % 16 == 0 or horizon_number == len(
                    validated["horizons"]
                ):
                    print(
                        f"  horizons={horizon_number}/{len(validated['horizons'])} "
                        f"latest_h={horizon}",
                        flush=True,
                    )
            _append_event(
                event_log,
                "fold_completed",
                fold_id=fold_id,
                validation_origins=len(validation_origins),
                low_fuel_rows=int((weight_vectors[3.0] > 1).sum()),
            )

        oof = pd.concat(predictions, ignore_index=True)
        expected_rows = (
            len(split)
            * len(validated["horizons"])
            * len(validated["multipliers"])
        )
        if len(oof) != expected_rows:
            raise AssertionError(f"OOF row count {len(oof)} != {expected_rows}")
        oof = attach_event_context(oof, episodes, context_hours=24)
        oof_path = output_dir / "oof_predictions.csv"
        oof.to_csv(
            oof_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        fit_summary = pd.DataFrame.from_records(fit_records)
        fit_summary_path = output_dir / "fit_summary.csv"
        fit_summary.to_csv(
            fit_summary_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )

        metrics = _summarize_metrics(oof)
        metrics_path = output_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        overall = metrics.loc[
            (metrics["label_variant"] == "official_interval_mean")
            & (metrics["scope"] == "overall")
        ].copy()
        controls = overall.loc[
            overall["variant"] == "control_weight1",
            ["period", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
        comparison = overall.merge(controls, on="period", validate="many_to_one")
        comparison["delta_accuracy_pct"] = (
            comparison["accuracy_1_minus_mape"] - comparison["control_accuracy"]
        ) * 100
        comparison = comparison[
            [
                "period",
                "variant",
                "mape",
                "accuracy_1_minus_mape",
                "delta_accuracy_pct",
            ]
        ].sort_values(["period", "mape"])
        comparison_path = output_dir / "variant_comparison.csv"
        comparison.to_csv(
            comparison_path, index=False, encoding="utf-8", float_format="%.10f"
        )
        gates = _evaluate_gates(metrics, config)
        gates_path = output_dir / "gate_results.csv"
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        report_path = output_dir / "report.md"
        _write_report(report_path, run_id, comparison, gates)

        primary_variant = _variant(float(config["variants"]["primary_multiplier"]))
        primary_gate = gates.loc[gates["variant"] == primary_variant].iloc[0]
        primary_long = comparison.loc[
            (comparison["period"] == "long")
            & (comparison["variant"] == primary_variant)
        ].iloc[0]
        ended_at = datetime.now(timezone.utc)
        component_passed = bool(primary_gate["component_gate_passed"])
        standalone = bool(primary_gate["standalone_submission_eligible"])
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "model_count": int(config["budget"]["expected_model_count"]),
                "feature_count": len(feature_columns),
                "model_parameters": model_parameters,
                "primary_variant": primary_variant,
                "primary_long_g1_mape": float(primary_long["mape"]),
                "primary_official_long_gain_pct": float(
                    primary_gate["official_long_gain_pct"]
                ),
                "component_gate_passed": component_passed,
                "standalone_submission_eligible": standalone,
                "submission_eligible": False,
                "artifacts": {
                    "oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(),
                    "oof_predictions_sha256": sha256(oof_path),
                    "fit_summary": fit_summary_path.relative_to(PROJECT_ROOT).as_posix(),
                    "fit_summary_sha256": sha256(fit_summary_path),
                    "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                    "metrics_sha256": sha256(metrics_path),
                    "variant_comparison": comparison_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "variant_comparison_sha256": sha256(comparison_path),
                    "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(),
                    "gate_results_sha256": sha256(gates_path),
                    "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                    "report_sha256": sha256(report_path),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        registry_row = {
            "run_id": run_id,
            "status": "completed",
            "experiment_role": config["experiment_role"],
            "task": config["task"],
            "protocol_version": "round2_v3",
            "label_version": validated["label_manifest"]["artifact_version"],
            "split_sha256": validated["split_manifest"]["assignment_sha256"],
            "control_run_id": config["control_artifact_id"],
            "hypothesis": config["hypothesis"],
            "primary_change": config["primary_change"],
            "model_name": "v28_g1_anomaly_sample_weight_x3",
            "long_g1_mape": float(primary_long["mape"]),
            "holdout_evaluated": False,
            "submission_eligible": False,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": manifest["duration_seconds"],
            "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "notes": (
                f"component_gate_passed={component_passed}; "
                f"standalone_submission_eligible={standalone}; "
                "scientific OOF only; exact v29b OOF unavailable"
            ),
        }
        if config["output"]["append_registry"]:
            _append_registry(registry_row)
            registry_written = True
        _append_event(
            event_log,
            "run_completed",
            duration_seconds=manifest["duration_seconds"],
            component_gate_passed=component_passed,
            standalone_submission_eligible=standalone,
        )
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} "
            f"component_gate_passed={component_passed} "
            f"standalone_submission_eligible={standalone}",
            flush=True,
        )
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "failed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(
            event_log,
            "run_failed",
            failure_type=type(exc).__name__,
            message=str(exc),
        )
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "split_sha256": validated["split_manifest"]["assignment_sha256"],
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "v28_g1_anomaly_sample_weight_x3",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
