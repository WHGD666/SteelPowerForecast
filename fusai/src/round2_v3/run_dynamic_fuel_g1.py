"""Run the pre-registered horizon-specific dynamic-fuel long-g1 experiment.

This runner creates scientific OOF evidence only. It reuses the frozen baseline
OOF as the control, fits every imputer/scaler/regression inside each time fold,
and never creates a platform submission.
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

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io_utils import sha256, write_json
from src.round2_v3.baseline_utils import build_supervised_long
from src.round2_v3.dynamic_fuel_utils import (
    build_dynamic_fuel_features,
    latest_mature_origin,
    recent_origin_window,
)
from src.round2_v3.event_ablation_utils import (
    attach_event_context,
    gate_results,
    summarize_ablation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "dynamic_fuel_g1_v1.yaml"
)
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "round2_v3" / "registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("dynamic_fuel_utils.py")
BASELINE_UTILS_SOURCE = RUNNER_SOURCE.with_name("baseline_utils.py")
EVENT_UTILS_SOURCE = RUNNER_SOURCE.with_name("event_ablation_utils.py")
METRICS_SOURCE = RUNNER_SOURCE.with_name("metrics.py")
CONTROL_MODEL = "lightgbm_shared_horizon"
RECENT_FOLDS = [
    "wf_03_bf3_active",
    "wf_04_late_september",
    "wf_05_holder1_active",
]


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
    return f"{stamp}_dynamic_fuel_g1_{digest.hexdigest()[:10]}"


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


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["partitions"] if item["partition"] == name]
    if len(matches) != 1:
        raise AssertionError(f"manifest has {len(matches)} {name} partitions")
    return matches[0]


def _validate_inputs(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    feature_manifest = _read_json(paths["feature_manifest"])
    label_manifest = _read_json(paths["label_manifest"])
    split_manifest = _read_json(paths["split_manifest"])
    control_manifest = _read_json(paths["control_manifest"])
    train_feature_path = PROJECT_ROOT / _partition(feature_manifest, "train")["output"]
    label_path = PROJECT_ROOT / label_manifest["output"]

    if sha256(train_feature_path) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("origin feature train artifact hash mismatch")
    if sha256(paths["feature_registry"]) != feature_manifest["registry_sha256"]:
        raise AssertionError("origin feature registry hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    if feature_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("origin features contain future observations")
    if feature_manifest["target_history_included"]:
        raise AssertionError("dynamic fuel experiment forbids target history")

    if control_manifest["run_id"] != config["control_run_id"]:
        raise AssertionError("control manifest run_id differs from config")
    if control_manifest["status"] != "completed":
        raise AssertionError("control run is not completed")
    expected_oof = control_manifest["artifacts"]["oof_predictions"]
    if paths["control_oof"] != PROJECT_ROOT / expected_oof:
        raise AssertionError("configured control OOF path differs from manifest")
    if sha256(paths["control_oof"]) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("control OOF hash mismatch")

    scope = config["frozen_scope"]
    if scope["period"] != "long" or scope["target"] != "generator_1":
        raise AssertionError("first dynamic fuel experiment must be long generator_1 only")
    if scope["short_outputs_changed"] or scope["generator_all_changed"]:
        raise AssertionError("short and generator_all outputs must remain frozen")
    if scope["true_target_history_allowed"] or scope["future_process_observations_allowed"]:
        raise AssertionError("forbidden information enabled in config")
    if config["training"]["estimator"] != "ordinary_least_squares":
        raise AssertionError("v1 estimator must remain ordinary least squares")
    if config["variants"]["broad_blend_search"]:
        raise AssertionError("broad blend search is forbidden in v1")
    if config["variants"]["promotion_candidate"] != "blend_dynamic70_control30":
        raise AssertionError("promotion candidate differs from pre-registration")

    episodes = pd.read_csv(paths["event_episodes"])
    return {
        "feature_manifest": feature_manifest,
        "label_manifest": label_manifest,
        "split_manifest": split_manifest,
        "control_manifest": control_manifest,
        "train_feature_path": train_feature_path,
        "label_path": label_path,
        "episodes": episodes,
    }


def _load_control_oof(path: Path, model_folds: set[str]) -> pd.DataFrame:
    columns = [
        "model",
        "period",
        "fold_id",
        "datetime",
        "interval_start",
        "target",
        "horizon_minutes",
        "actual",
        "prediction",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250_000):
        selected = chunk.loc[
            (chunk["model"] == CONTROL_MODEL)
            & (chunk["period"] == "long")
            & (chunk["target"] == "generator_1")
            & (chunk["fold_id"].isin(model_folds))
        ].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("control OOF contains no matching long generator_1 rows")
    result = pd.concat(parts, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["datetime"], errors="raise")
    result["interval_start"] = pd.to_datetime(result["interval_start"], errors="raise")
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    if result.duplicated(key).any():
        raise ValueError("control OOF contains duplicate prediction keys")
    return result


def _prediction_part(
    meta: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
    variant: str,
    fold_id: str,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    result = meta.copy()
    result["actual"] = actual
    result["prediction"] = prediction
    result["variant"] = variant
    result["fold_id"] = fold_id
    return attach_event_context(result, episodes, context_hours=24)[
        [
            "variant",
            "fold_id",
            "datetime",
            "interval_start",
            "horizon_minutes",
            "actual",
            "prediction",
            "origin_regime",
            "origin_episode_id",
            "target_in_event",
            "target_episode_id",
            "horizon_bucket",
        ]
    ]


def _apply_recent_fold_floor(
    gates: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = gates.copy()
    control = metrics.loc[
        (metrics["scope"] == "fold") & (metrics["variant"] == "control"),
        ["fold_id", "accuracy_1_minus_mape"],
    ].set_index("fold_id")["accuracy_1_minus_mape"]
    minimums: list[float] = []
    for variant in result["variant"]:
        candidate = metrics.loc[
            (metrics["scope"] == "fold") & (metrics["variant"] == variant),
            ["fold_id", "accuracy_1_minus_mape"],
        ].set_index("fold_id")["accuracy_1_minus_mape"]
        deltas = [
            (float(candidate.loc[fold]) - float(control.loc[fold])) * 100
            for fold in RECENT_FOLDS
        ]
        minimums.append(float(min(deltas)))
    result["recent_single_fold_min_gain_pct"] = minimums
    floor = float(config["selection_gate"]["minimum_recent_single_fold_gain_pct"])
    result["pass_recent_single_fold_floor"] = (
        result["recent_single_fold_min_gain_pct"] >= floor
    )
    eligible = config["variants"]["promotion_candidate"]
    result["promotion_eligible"] = result["variant"] == eligible
    result["promotion_gate_passed"] = (
        result["promotion_gate_passed"].astype(bool)
        & result["pass_recent_single_fold_floor"].astype(bool)
        & result["promotion_eligible"].astype(bool)
    )
    return result


def _write_report(
    path: Path,
    run_id: str,
    comparison: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# Horizon-specific dynamic fuel long-g1",
        "",
        f"- run_id: `{run_id}`",
        "- 角色：scientific OOF；不是 v29b 复现，不自动生成提交。",
        "",
        "## 总体结果",
        "",
        "| variant | MAPE | accuracy | delta accuracy (pct) |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.mape:.6f} | "
            f"{row.accuracy_1_minus_mape:.6f} | {row.delta_accuracy_pct:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 预登记门禁",
            "",
            "| variant | overall | recent mean | recent worst | episodes | pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gates.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.overall_accuracy_gain_pct:.4f} | "
            f"{row.recent_fold_mean_gain_pct:.4f} | "
            f"{row.recent_single_fold_min_gain_pct:.4f} | "
            f"{row.episode_improvement_fraction:.1%} | "
            f"{row.promotion_gate_passed} |"
        )
    lines.extend(
        [
            "",
            "只有预登记的 70% dynamic + 30% control 可以通过晋级门禁；",
            "standalone OLS 只用于机制诊断。通过本地门禁也不会自动生成提交。",
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
    paths = {key: PROJECT_ROOT / value for key, value in config["inputs"].items()}
    validated = _validate_inputs(config, paths)

    origin_features = pd.read_csv(validated["train_feature_path"], low_memory=False)
    feature_config = config["dynamic_fuel_features"]
    dynamic_features, dynamic_registry = build_dynamic_fuel_features(
        origin_features,
        feature_config["fuel_columns"],
        feature_config["lag_steps"],
        feature_config["rolling_mean_steps"],
        frequency_minutes=int(feature_config["origin_frequency_minutes"]),
        include_current=bool(feature_config["include_current"]),
    )
    dynamic_columns = dynamic_registry["feature_name"].tolist()
    model_columns = [*dynamic_columns, *feature_config["target_calendar_columns"]]
    dynamic_times = pd.DatetimeIndex(dynamic_features["datetime"])

    split_assignments = pd.read_csv(paths["split_assignments"])
    for column in (
        "train_end",
        "latest_eligible_train_origin",
        "validation_origin",
        "short_label_end",
        "long_label_end",
    ):
        split_assignments[column] = pd.to_datetime(
            split_assignments[column], errors="raise"
        )
    split_assignments = split_assignments.loc[
        split_assignments["role"] == "model_selection"
    ].copy()
    fold_count = int(split_assignments["fold_id"].nunique())
    horizon_count = len(config["frozen_scope"]["horizons_minutes"])
    expected_models = int(config["budget"]["expected_model_count"])
    if fold_count != int(config["budget"]["fold_count"]):
        raise AssertionError("model-selection fold count differs from budget")
    if fold_count * horizon_count != expected_models:
        raise AssertionError("configured dynamic fuel model budget is inconsistent")
    if args.preflight_only:
        frequency = int(feature_config["origin_frequency_minutes"])
        window_days = int(config["training"]["recent_window_days"])
        for _, fold in split_assignments.groupby("fold_id", sort=False):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            for horizon in config["frozen_scope"]["horizons_minutes"]:
                latest_origin = latest_mature_origin(
                    train_end, int(horizon), frequency_minutes=frequency
                )
                recent_origin_window(
                    dynamic_times,
                    latest_origin,
                    window_days=window_days,
                    frequency_minutes=frequency,
                )
        print(
            "PASS preflight "
            f"folds={fold_count} horizons={horizon_count} models={expected_models} "
            f"fuel_features={len(dynamic_columns)} model_features={len(model_columns)} "
            f"window_days={window_days} "
            f"max_source_offset={int(dynamic_registry['source_offset_max_minutes'].max())}"
        )
        return

    identity_inputs = [
        *paths.values(),
        RUNNER_SOURCE,
        UTILS_SOURCE,
        BASELINE_UTILS_SOURCE,
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
        "control_run_id": config["control_run_id"],
        "external_reference_artifact_id": config["external_reference_artifact_id"],
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"branch": git_branch, "commit": git_commit, "dirty": git_dirty},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
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
            for path in identity_inputs
        },
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False

    try:
        interval_targets = pd.read_csv(validated["label_path"], low_memory=False)
        interval_targets["datetime"] = pd.to_datetime(
            interval_targets["datetime"], errors="raise"
        )
        model_folds = set(split_assignments["fold_id"])
        control_oof = _load_control_oof(paths["control_oof"], model_folds)
        episodes = validated["episodes"].copy()
        horizons = [int(value) for value in config["frozen_scope"]["horizons_minutes"]]
        frequency = int(feature_config["origin_frequency_minutes"])
        window_days = int(config["training"]["recent_window_days"])
        blend = config["variants"]["blend_dynamic70_control30"]
        dynamic_weight = float(blend["dynamic_fuel_weight"])
        control_weight = float(blend["control_weight"])
        if not np.isclose(dynamic_weight + control_weight, 1.0):
            raise AssertionError("blend weights must sum to one")

        predictions: list[pd.DataFrame] = []
        coefficient_records: list[dict[str, object]] = []
        fold_groups = list(split_assignments.groupby("fold_id", sort=False))
        for fold_number, (fold_id, fold) in enumerate(fold_groups, start=1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            validation_origins = pd.DatetimeIndex(
                fold["validation_origin"]
            ).sort_values()
            print(
                f"[{fold_number}/{fold_count}] {fold_id}: "
                f"validation_origins={len(validation_origins)}",
                flush=True,
            )
            for horizon_number, horizon in enumerate(horizons, start=1):
                latest_origin = latest_mature_origin(
                    train_end, horizon, frequency_minutes=frequency
                )
                training_origins = recent_origin_window(
                    dynamic_times,
                    latest_origin,
                    window_days=window_days,
                    frequency_minutes=frequency,
                )
                x_train, y_train, _ = build_supervised_long(
                    dynamic_features,
                    interval_targets,
                    training_origins,
                    [horizon],
                    "generator_1",
                    dynamic_columns,
                )
                x_validation, y_validation, validation_meta = build_supervised_long(
                    dynamic_features,
                    interval_targets,
                    validation_origins,
                    [horizon],
                    "generator_1",
                    dynamic_columns,
                )
                if len(y_validation) != len(validation_origins):
                    raise AssertionError(
                        f"incomplete validation labels for {fold_id} h={horizon}"
                    )
                model = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                        ("regressor", LinearRegression()),
                    ]
                )
                model.fit(x_train[model_columns], y_train)
                dynamic_prediction = model.predict(x_validation[model_columns])

                control_part = control_oof.loc[
                    (control_oof["fold_id"] == fold_id)
                    & (control_oof["horizon_minutes"] == horizon)
                ].sort_values(["datetime", "interval_start"])
                expected_meta = validation_meta.sort_values(
                    ["datetime", "interval_start"]
                ).reset_index(drop=True)
                control_part = control_part.reset_index(drop=True)
                if len(control_part) != len(expected_meta):
                    raise AssertionError(
                        f"control row mismatch for {fold_id} h={horizon}: "
                        f"{len(control_part)} != {len(expected_meta)}"
                    )
                if not control_part[["datetime", "interval_start"]].equals(
                    expected_meta[["datetime", "interval_start"]]
                ):
                    raise AssertionError("control prediction keys differ from validation keys")
                np.testing.assert_allclose(
                    control_part["actual"].to_numpy(dtype=float),
                    y_validation,
                    rtol=0,
                    atol=1e-8,
                    err_msg="control and rebuilt validation labels differ",
                )
                control_prediction = control_part["prediction"].to_numpy(dtype=float)
                blend_prediction = (
                    dynamic_weight * dynamic_prediction
                    + control_weight * control_prediction
                )
                predictions.extend(
                    [
                        _prediction_part(
                            validation_meta,
                            y_validation,
                            control_prediction,
                            "control",
                            fold_id,
                            episodes,
                        ),
                        _prediction_part(
                            validation_meta,
                            y_validation,
                            dynamic_prediction,
                            "dynamic_fuel_ols",
                            fold_id,
                            episodes,
                        ),
                        _prediction_part(
                            validation_meta,
                            y_validation,
                            blend_prediction,
                            "blend_dynamic70_control30",
                            fold_id,
                            episodes,
                        ),
                    ]
                )

                imputer = model.named_steps["imputer"]
                scaler = model.named_steps["scaler"]
                regressor = model.named_steps["regressor"]
                for feature_index, feature_name in enumerate(model_columns):
                    coefficient_records.append(
                        {
                            "fold_id": fold_id,
                            "horizon_minutes": horizon,
                            "feature_name": feature_name,
                            "coefficient_standardized": float(
                                regressor.coef_[feature_index]
                            ),
                            "intercept": float(regressor.intercept_),
                            "imputer_median": float(imputer.statistics_[feature_index]),
                            "scaler_mean": float(scaler.mean_[feature_index]),
                            "scaler_scale": float(scaler.scale_[feature_index]),
                            "training_rows": int(len(y_train)),
                            "training_origin_start": training_origins.min(),
                            "training_origin_end": training_origins.max(),
                        }
                    )
                if horizon_number % 16 == 0 or horizon_number == horizon_count:
                    print(
                        f"  horizons={horizon_number}/{horizon_count} "
                        f"latest_h={horizon}",
                        flush=True,
                    )
            _append_event(
                event_log,
                "fold_completed",
                fold_id=fold_id,
                horizon_models=horizon_count,
                validation_origins=len(validation_origins),
            )

        oof = pd.concat(predictions, ignore_index=True)
        expected_control_rows = len(control_oof)
        actual_control_rows = int((oof["variant"] == "control").sum())
        if actual_control_rows != expected_control_rows:
            raise AssertionError(
                f"rebuilt control coverage differs: {actual_control_rows} != {expected_control_rows}"
            )
        oof_path = output_dir / "oof_predictions.csv"
        oof.to_csv(
            oof_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        coefficients = pd.DataFrame.from_records(coefficient_records)
        coefficients_path = output_dir / "coefficients.csv"
        coefficients.to_csv(
            coefficients_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        registry_path = output_dir / "dynamic_fuel_feature_registry.csv"
        dynamic_registry.to_csv(registry_path, index=False, encoding="utf-8")

        metrics = summarize_ablation(oof)
        metrics_path = output_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        overall = metrics.loc[metrics["scope"] == "overall"].copy()
        control_accuracy = float(
            overall.loc[
                overall["variant"] == "control", "accuracy_1_minus_mape"
            ].iloc[0]
        )
        overall["delta_accuracy_pct"] = (
            overall["accuracy_1_minus_mape"] - control_accuracy
        ) * 100
        comparison = overall[
            ["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]
        ].sort_values("mape")
        comparison_path = output_dir / "variant_comparison.csv"
        comparison.to_csv(
            comparison_path, index=False, encoding="utf-8", float_format="%.10f"
        )
        gates = gate_results(
            metrics,
            "control",
            ["dynamic_fuel_ols", "blend_dynamic70_control30"],
            config["selection_gate"],
            RECENT_FOLDS,
        )
        gates = _apply_recent_fold_floor(gates, metrics, config)
        gates_path = output_dir / "gate_results.csv"
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        report_path = output_dir / "report.md"
        _write_report(report_path, run_id, comparison, gates)

        candidate = comparison.loc[
            comparison["variant"] == config["variants"]["promotion_candidate"]
        ].iloc[0]
        passed = gates.loc[gates["promotion_gate_passed"], "variant"].tolist()
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "model_count": expected_models,
                "dynamic_feature_count": len(dynamic_columns),
                "model_feature_count": len(model_columns),
                "model_parameters": {
                    "estimator": "ordinary_least_squares",
                    "recent_window_days": window_days,
                    "dynamic_fuel_weight": dynamic_weight,
                    "control_weight": control_weight,
                },
                "promotion_candidate": config["variants"]["promotion_candidate"],
                "candidate_long_g1_mape": float(candidate["mape"]),
                "promotion_gate_passed_variants": passed,
                "submission_eligible": False,
                "artifacts": {
                    "oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(),
                    "oof_predictions_sha256": sha256(oof_path),
                    "coefficients": coefficients_path.relative_to(PROJECT_ROOT).as_posix(),
                    "coefficients_sha256": sha256(coefficients_path),
                    "feature_registry": registry_path.relative_to(PROJECT_ROOT).as_posix(),
                    "feature_registry_sha256": sha256(registry_path),
                    "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                    "metrics_sha256": sha256(metrics_path),
                    "variant_comparison": comparison_path.relative_to(PROJECT_ROOT).as_posix(),
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
            "control_run_id": config["control_run_id"],
            "hypothesis": config["hypothesis"],
            "primary_change": "horizon-specific 21-day dynamic fuel OLS for long generator_1",
            "model_name": "dynamic_fuel_ols_blend70_control30",
            "long_g1_mape": float(candidate["mape"]),
            "holdout_evaluated": False,
            "submission_eligible": False,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": manifest["duration_seconds"],
            "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "notes": (
                f"promotion_candidate={config['variants']['promotion_candidate']}; "
                f"promotion_passed={passed}; scientific OOF only, not v29b OOF"
            ),
        }
        if config["output"]["append_registry"]:
            _append_registry(registry_row)
            registry_written = True
        _append_event(
            event_log,
            "run_completed",
            duration_seconds=manifest["duration_seconds"],
            promotion_passed=passed,
        )
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} promotion_passed={passed}",
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
                    "control_run_id": config["control_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": "horizon-specific 21-day dynamic fuel OLS for long generator_1",
                    "model_name": "dynamic_fuel_ols_blend70_control30",
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

