"""Isolate official interval-mean label alignment in the v28 long-g1 recipe."""

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
    build_v28_features,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
    v28_g1_fuel_weight,
)
from src.round2_v3.event_ablation_utils import attach_event_context, summarize_ablation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "v28_label_alignment_v1.yaml"
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
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _run_identity(config_path: Path, inputs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *inputs]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_v28_label_alignment_{digest.hexdigest()[:10]}"


def _append_event(path: Path, event: str, **values: object) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **values}
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
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(
            {field: row.get(field, "") for field in fieldnames}
        )


def _validate_inputs(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    raw_manifest = _read_json(paths["raw_fingerprints"])
    expected_hashes = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
    raw_tables = [PROJECT_ROOT / value for value in config["inputs"]["raw_train_tables"]]
    for path in raw_tables:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256(path) != expected_hashes.get(relative):
            raise AssertionError(f"raw table hash mismatch: {relative}")
    label_manifest = _read_json(paths["label_manifest"])
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("interval label hash mismatch")
    split_manifest = _read_json(paths["split_manifest"])
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    control_manifest = _read_json(paths["control_manifest"])
    if control_manifest["run_id"] != config["control_scientific_run_id"]:
        raise AssertionError("control run id differs from config")
    if control_manifest["status"] != "completed":
        raise AssertionError("control run is not completed")
    if sha256(paths["control_oof"]) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("control OOF hash mismatch")
    if not config["training"]["use_same_training_origins_as_legacy_control"]:
        raise AssertionError("label alignment must use the legacy control origins")
    if config["training"]["target_history_features_allowed"]:
        raise AssertionError("target history features are forbidden")
    if config["training"]["future_process_observations_allowed"]:
        raise AssertionError("future process observations are forbidden")
    if config["variants"]["select_diagnostic_as_winner"]:
        raise AssertionError("diagnostic ablations cannot be selected after OOF")
    if config["variants"]["promotion_candidate"] != "interval_tree_plus_interval_fuel":
        raise AssertionError("promotion candidate differs from preregistration")
    horizons = [int(value) for value in config["horizons_minutes"]]
    folds = list(config["validation"]["fold_ids"])
    expected_models = len(folds) * len(horizons)
    if expected_models != int(config["budget"]["expected_model_count"]):
        raise AssertionError("model budget is inconsistent")
    return {
        "raw_tables": raw_tables,
        "label_manifest": label_manifest,
        "label_path": label_path,
        "split_manifest": split_manifest,
        "control_manifest": control_manifest,
        "folds": folds,
        "horizons": horizons,
    }


def _load_control(path: Path, folds: list[str]) -> pd.DataFrame:
    usecols = [
        "variant",
        "fold_id",
        "datetime",
        "interval_start",
        "horizon_minutes",
        "actual_interval_mean",
        "prediction",
    ]
    control = pd.read_csv(path, usecols=usecols)
    control = control.loc[
        (control["variant"] == "control_weight1") & control["fold_id"].isin(folds)
    ].copy()
    control["datetime"] = pd.to_datetime(control["datetime"], errors="raise")
    control["interval_start"] = pd.to_datetime(control["interval_start"], errors="raise")
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    if control.duplicated(key).any():
        raise AssertionError("control OOF contains duplicate keys")
    return control


def _evaluate_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    control = config["variants"]["control"]
    candidate = config["variants"]["promotion_candidate"]
    gate = config["selection_gate"]

    def accuracy(
        variant: str,
        scope: str,
        *,
        fold: str = "all",
        bucket: str = "all",
        regime: str = "all",
    ) -> float:
        selected = metrics.loc[
            (metrics["variant"] == variant)
            & (metrics["scope"] == scope)
            & (metrics["fold_id"].astype(str) == str(fold))
            & (metrics["horizon_bucket"].astype(str) == str(bucket))
            & (metrics["origin_regime"].astype(str) == str(regime))
        ]
        return np.nan if selected.empty else float(selected.iloc[0]["accuracy_1_minus_mape"])

    overall_gain = (accuracy(candidate, "overall") - accuracy(control, "overall")) * 100
    fold_deltas = [
        (accuracy(candidate, "fold", fold=fold) - accuracy(control, "fold", fold=fold)) * 100
        for fold in config["validation"]["fold_ids"]
    ]
    ordinary_loss = (
        accuracy(control, "origin_regime", regime="ordinary")
        - accuracy(candidate, "origin_regime", regime="ordinary")
    ) * 100
    bucket_deltas = {
        bucket: (
            accuracy(candidate, "horizon_bucket", bucket=bucket)
            - accuracy(control, "horizon_bucket", bucket=bucket)
        )
        * 100
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440")
    }
    control_episodes = metrics.loc[
        (metrics["scope"] == "episode") & (metrics["variant"] == control),
        ["episode_id", "accuracy_1_minus_mape"],
    ].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
    candidate_episodes = metrics.loc[
        (metrics["scope"] == "episode") & (metrics["variant"] == candidate),
        ["episode_id", "accuracy_1_minus_mape"],
    ].rename(columns={"accuracy_1_minus_mape": "candidate_accuracy"})
    episodes = control_episodes.merge(
        candidate_episodes, on="episode_id", how="inner", validate="one_to_one"
    )
    episode_fraction = float(
        (episodes["candidate_accuracy"] > episodes["control_accuracy"]).mean()
    ) if len(episodes) else np.nan
    pass_component = overall_gain >= float(gate["minimum_component_long_g1_accuracy_gain_pct"])
    pass_folds = min(fold_deltas) >= float(gate["minimum_single_fold_gain_pct"])
    pass_ordinary = ordinary_loss <= float(gate["maximum_ordinary_origin_accuracy_loss_pct"])
    pass_episodes = np.isfinite(episode_fraction) and episode_fraction > 0.5
    pass_near_far = min(bucket_deltas["h015_120"], bucket_deltas["h735_1440"]) >= -float(
        gate["maximum_near_far_accuracy_loss_pct"]
    )
    component_passed = bool(
        pass_component and pass_folds and pass_ordinary and pass_episodes and pass_near_far
    )
    standalone = bool(
        component_passed
        and overall_gain >= float(gate["minimum_standalone_long_g1_accuracy_gain_pct"])
    )
    return pd.DataFrame.from_records(
        [
            {
                "variant": candidate,
                "overall_accuracy_gain_pct": overall_gain,
                "single_fold_min_gain_pct": float(min(fold_deltas)),
                "single_fold_mean_gain_pct": float(np.mean(fold_deltas)),
                "ordinary_accuracy_loss_pct": ordinary_loss,
                "episode_improvement_fraction": episode_fraction,
                "evaluated_episode_count": int(len(episodes)),
                "near_gain_pct": bucket_deltas["h015_120"],
                "mid1_gain_pct": bucket_deltas["h135_360"],
                "mid2_gain_pct": bucket_deltas["h375_720"],
                "far_gain_pct": bucket_deltas["h735_1440"],
                "pass_component_gain": pass_component,
                "pass_folds": pass_folds,
                "pass_ordinary": pass_ordinary,
                "pass_episodes": pass_episodes,
                "pass_near_far": pass_near_far,
                "component_gate_passed": component_passed,
                "standalone_submission_eligible": standalone,
            }
        ]
    )


def _write_report(path: Path, run_id: str, comparison: pd.DataFrame, gate: pd.DataFrame) -> None:
    lines = [
        "# v28 g1 official interval-label alignment",
        "",
        f"- run_id: `{run_id}`",
        "- 角色：scientific OOF；不自动生成提交包。",
        "",
        "| variant | MAPE | accuracy | delta accuracy (pct) |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.mape:.6f} | {row.accuracy_1_minus_mape:.6f} | "
            f"{row.delta_accuracy_pct:.4f} |"
        )
    result = gate.iloc[0]
    lines.extend(
        [
            "",
            "## Primary gate",
            "",
            f"- overall gain: {result.overall_accuracy_gain_pct:.4f} pct",
            f"- worst fold: {result.single_fold_min_gain_pct:.4f} pct",
            f"- ordinary loss: {result.ordinary_accuracy_loss_pct:.4f} pct",
            f"- episode improvement: {result.episode_improvement_fraction:.1%}",
            f"- near/far: {result.near_gain_pct:.4f} / {result.far_gain_pct:.4f} pct",
            f"- component_gate_passed: `{result.component_gate_passed}`",
            f"- standalone_submission_eligible: `{result.standalone_submission_eligible}`",
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
    paths = {
        key: PROJECT_ROOT / value
        for key, value in config["inputs"].items()
        if key != "raw_train_tables"
    }
    validated = _validate_inputs(config, paths)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"folds={len(validated['folds'])} horizons={len(validated['horizons'])} "
            f"models={config['budget']['expected_model_count']} "
            "candidate=interval_tree_plus_interval_fuel"
        )
        return

    identity_inputs = [
        *validated["raw_tables"],
        *paths.values(),
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
        "control_scientific_run_id": config["control_scientific_run_id"],
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
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
        interval_labels = pd.read_csv(validated["label_path"])
        interval_labels["datetime"] = pd.to_datetime(interval_labels["datetime"], errors="raise")
        interval_mean = pd.to_numeric(
            interval_labels.set_index("datetime")["generator_1"], errors="coerce"
        )
        split = pd.read_csv(paths["split_assignments"])
        split["train_end"] = pd.to_datetime(split["train_end"], errors="raise")
        split["validation_origin"] = pd.to_datetime(split["validation_origin"], errors="raise")
        split = split.loc[
            (split["role"] == "model_selection")
            & split["fold_id"].isin(validated["folds"])
        ].copy()
        control = _load_control(paths["control_oof"], validated["folds"])
        episodes = pd.read_csv(paths["event_episodes"])
        training_start = pd.Timestamp(config["training"]["g1_tree_start"])
        prediction_parts: list[pd.DataFrame] = []
        fit_records: list[dict[str, object]] = []

        for fold_number, (fold_id, fold) in enumerate(split.groupby("fold_id", sort=False), start=1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            cutoff_position = int(np.flatnonzero(timestamps <= train_end)[-1])
            fuel_mask = (grid["datetime"] >= training_start) & (grid["datetime"] <= train_end)
            point_fuel_frame = grid.loc[fuel_mask].copy()
            point_coefficients = fit_linear_fuel_proxy(
                point_fuel_frame, "generator_1", FUELS_G1
            )
            interval_fuel_frame = point_fuel_frame.copy()
            interval_fuel_frame["generator_1_interval"] = interval_mean.reindex(
                pd.DatetimeIndex(interval_fuel_frame["datetime"])
            ).to_numpy(dtype=float)
            interval_coefficients = fit_linear_fuel_proxy(
                interval_fuel_frame, "generator_1_interval", FUELS_G1
            )
            validation_origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
            validation_positions = position.loc[validation_origins].to_numpy(dtype=int)
            point_fuel = predict_linear_fuel(
                grid.iloc[validation_positions], FUELS_G1, point_coefficients
            )
            aligned_fuel = predict_linear_fuel(
                grid.iloc[validation_positions], FUELS_G1, interval_coefficients
            )
            print(
                f"[{fold_number}/{len(validated['folds'])}] {fold_id}: "
                f"validation_origins={len(validation_origins)}",
                flush=True,
            )

            for horizon_number, horizon in enumerate(validated["horizons"], start=1):
                step = horizon // 15
                eligible_positions = np.flatnonzero(
                    (timestamps >= training_start)
                    & (np.arange(len(grid)) + step <= cutoff_position)
                )
                target_interval_starts = timestamps[eligible_positions] + pd.to_timedelta(
                    horizon - 15, unit="min"
                )
                y_train = interval_mean.reindex(target_interval_starts).to_numpy(dtype=float)
                valid_train = np.isfinite(y_train)
                eligible_positions = eligible_positions[valid_train]
                y_train = y_train[valid_train]
                if len(y_train) < int(config["training"]["minimum_rows_per_model"]):
                    raise AssertionError(
                        f"{fold_id} h={horizon} has only {len(y_train)} training rows"
                    )
                model = lgb.LGBMRegressor(**dict(config["lightgbm"]))
                model.fit(features.iloc[eligible_positions][feature_columns], y_train)
                interval_tree = np.clip(
                    model.predict(features.iloc[validation_positions][feature_columns]),
                    0.0,
                    None,
                )
                control_part = control.loc[
                    (control["fold_id"] == fold_id)
                    & (control["horizon_minutes"] == horizon)
                ].sort_values("datetime").reset_index(drop=True)
                if not pd.DatetimeIndex(control_part["datetime"]).equals(validation_origins):
                    raise AssertionError(f"control origin mismatch {fold_id} h={horizon}")
                control_prediction = control_part["prediction"].to_numpy(dtype=float)
                actual = control_part["actual_interval_mean"].to_numpy(dtype=float)
                fuel_weight = v28_g1_fuel_weight(horizon)
                point_tree = (control_prediction - fuel_weight * point_fuel) / (
                    1.0 - fuel_weight
                )
                variants = {
                    "legacy_point_tree_plus_legacy_point_fuel": control_prediction,
                    "interval_tree_plus_legacy_point_fuel": (
                        (1.0 - fuel_weight) * interval_tree + fuel_weight * point_fuel
                    ),
                    "legacy_point_tree_plus_interval_fuel": (
                        (1.0 - fuel_weight) * point_tree + fuel_weight * aligned_fuel
                    ),
                    "interval_tree_plus_interval_fuel": (
                        (1.0 - fuel_weight) * interval_tree + fuel_weight * aligned_fuel
                    ),
                }
                interval_starts = validation_origins + pd.to_timedelta(
                    horizon - 15, unit="min"
                )
                for variant, prediction in variants.items():
                    prediction_parts.append(
                        pd.DataFrame(
                            {
                                "variant": variant,
                                "fold_id": fold_id,
                                "datetime": validation_origins,
                                "interval_start": interval_starts,
                                "horizon_minutes": horizon,
                                "actual": actual,
                                "prediction": prediction,
                            }
                        )
                    )
                fit_records.append(
                    {
                        "fold_id": fold_id,
                        "horizon_minutes": horizon,
                        "training_rows": int(len(y_train)),
                        "fuel_weight": fuel_weight,
                        "training_origin_start": timestamps[eligible_positions.min()],
                        "training_origin_end": timestamps[eligible_positions.max()],
                    }
                )
                if horizon_number % 16 == 0 or horizon_number == len(validated["horizons"]):
                    print(
                        f"  horizons={horizon_number}/{len(validated['horizons'])} latest_h={horizon}",
                        flush=True,
                    )
            _append_event(event_log, "fold_completed", fold_id=fold_id)

        oof = attach_event_context(pd.concat(prediction_parts, ignore_index=True), episodes)
        metrics = summarize_ablation(oof)
        overall = metrics.loc[metrics["scope"] == "overall"].copy()
        control_accuracy = float(
            overall.loc[
                overall["variant"] == config["variants"]["control"],
                "accuracy_1_minus_mape",
            ].iloc[0]
        )
        overall["delta_accuracy_pct"] = (
            overall["accuracy_1_minus_mape"] - control_accuracy
        ) * 100
        comparison = overall[
            ["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]
        ].sort_values("mape")
        gate = _evaluate_gate(metrics, config)

        oof_path = output_dir / "oof_predictions.csv"
        fit_path = output_dir / "fit_summary.csv"
        metrics_path = output_dir / "metrics.csv"
        comparison_path = output_dir / "variant_comparison.csv"
        gate_path = output_dir / "gate_results.csv"
        report_path = output_dir / "report.md"
        oof.to_csv(oof_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
        pd.DataFrame.from_records(fit_records).to_csv(fit_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        comparison.to_csv(comparison_path, index=False, encoding="utf-8", float_format="%.10f")
        gate.to_csv(gate_path, index=False, encoding="utf-8", float_format="%.10f")
        _write_report(report_path, run_id, comparison, gate)

        primary = comparison.loc[
            comparison["variant"] == config["variants"]["promotion_candidate"]
        ].iloc[0]
        component_passed = bool(gate.iloc[0]["component_gate_passed"])
        standalone = bool(gate.iloc[0]["standalone_submission_eligible"])
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "model_count": int(config["budget"]["expected_model_count"]),
                "feature_count": len(feature_columns),
                "primary_variant": config["variants"]["promotion_candidate"],
                "primary_long_g1_mape": float(primary["mape"]),
                "component_gate_passed": component_passed,
                "standalone_submission_eligible": standalone,
                "submission_eligible": False,
                "artifacts": {
                    "oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(),
                    "oof_predictions_sha256": sha256(oof_path),
                    "fit_summary": fit_path.relative_to(PROJECT_ROOT).as_posix(),
                    "fit_summary_sha256": sha256(fit_path),
                    "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                    "metrics_sha256": sha256(metrics_path),
                    "variant_comparison": comparison_path.relative_to(PROJECT_ROOT).as_posix(),
                    "variant_comparison_sha256": sha256(comparison_path),
                    "gate_results": gate_path.relative_to(PROJECT_ROOT).as_posix(),
                    "gate_results_sha256": sha256(gate_path),
                    "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                    "report_sha256": sha256(report_path),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "split_sha256": validated["split_manifest"]["assignment_sha256"],
                    "control_run_id": config["control_scientific_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": config["variants"]["promotion_candidate"],
                    "long_g1_mape": float(primary["mape"]),
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"component_gate_passed={component_passed}; standalone_submission_eligible={standalone}",
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", component_gate_passed=component_passed, standalone_submission_eligible=standalone)
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gate.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} "
            f"component_gate_passed={component_passed} standalone_submission_eligible={standalone}",
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
        _append_event(event_log, "run_failed", failure_type=type(exc).__name__, message=str(exc))
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
                    "control_run_id": config["control_scientific_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": config["variants"]["promotion_candidate"],
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
