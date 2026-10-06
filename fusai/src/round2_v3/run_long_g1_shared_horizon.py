"""Compare the v28 per-horizon trees with one shared-horizon long-g1 tree."""

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
)
from src.round2_v3.baseline_utils import TARGET_CALENDAR_COLUMNS, build_prediction_long
from src.round2_v3.event_ablation_utils import attach_event_context, summarize_ablation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/long_g1_shared_horizon_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("anomaly_weight_utils.py")
BASELINE_UTILS_SOURCE = RUNNER_SOURCE.with_name("baseline_utils.py")
EVENT_UTILS_SOURCE = RUNNER_SOURCE.with_name("event_ablation_utils.py")
METRICS_SOURCE = RUNNER_SOURCE.with_name("metrics.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _append_event(path: Path, event: str, **values: object) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {item["run_id"] for item in reader}
    if not fields or str(row["run_id"]) in existing:
        raise RuntimeError("invalid or duplicate experiment registry entry")
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(
            {field: row.get(field, "") for field in fields}
        )


def _identity(config_path: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *paths]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_long_g1_shared_"
        + digest.hexdigest()[:10]
    )


def _validate(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    raw_manifest = _read_json(paths["raw_fingerprints"])
    expected = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
    raw_tables = [PROJECT_ROOT / value for value in config["inputs"]["raw_train_tables"]]
    for path in raw_tables:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256(path) != expected.get(relative):
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
    training = config["training"]
    if not training["use_same_training_origin_contract_as_v28"]:
        raise AssertionError("training origin contract must remain v28 exact")
    if training["target_history_features_allowed"] or training["future_process_observations_allowed"]:
        raise AssertionError("future process/target-history features are forbidden")
    if list(training["global_tree_features"]) != list(TARGET_CALENDAR_COLUMNS):
        raise AssertionError("shared-horizon position features differ from frozen contract")
    fuel = config["fuel_recipe"]
    if float(fuel["near_weight"]) != 0.8 or float(fuel["far_weight"]) != 0.6:
        raise AssertionError("v28 fuel weights must remain 0.8/0.6")
    horizons = [int(value) for value in config["horizons_minutes"]]
    folds = list(config["validation"]["fold_ids"])
    if horizons != list(range(15, 1441, 15)):
        raise AssertionError("long horizons must remain 15..1440 by 15")
    if len(folds) != int(config["budget"]["expected_model_count"]):
        raise AssertionError("shared-horizon model budget must equal fold count")
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
    columns = [
        "variant", "fold_id", "datetime", "interval_start", "horizon_minutes",
        "actual_interval_mean", "actual_legacy_point", "prediction",
    ]
    control = pd.read_csv(path, usecols=columns, parse_dates=["datetime", "interval_start"])
    control = control.loc[
        (control["variant"] == "control_weight1") & control["fold_id"].isin(folds)
    ].copy()
    control["variant"] = "v28_per_horizon_control"
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    if control.duplicated(key).any():
        raise AssertionError("control OOF contains duplicate keys")
    return control


def _metric_tables(oof: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for label_variant, actual_column in (
        ("official_interval_mean", "actual_interval_mean"),
        ("legacy_point", "actual_legacy_point"),
    ):
        source = oof.copy()
        source["actual"] = source[actual_column]
        metrics = summarize_ablation(source)
        metrics.insert(0, "label_variant", label_variant)
        parts.append(metrics)
    return pd.concat(parts, ignore_index=True)


def _align_calendar_to_point_target(
    features: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Move shared calendar columns from interval start to the v28 point label."""
    result = features.copy()
    target_times = pd.DatetimeIndex(meta["datetime"]) + pd.TimedeltaIndex(
        pd.to_timedelta(meta["horizon_minutes"].to_numpy(), unit="min")
    )
    hour_angle = 2 * np.pi * (target_times.hour + target_times.minute / 60) / 24
    minute_angle = 2 * np.pi * target_times.minute / 60
    day_angle = 2 * np.pi * target_times.dayofweek / 7
    result["target_hour_sin"] = np.sin(hour_angle).astype(np.float32)
    result["target_hour_cos"] = np.cos(hour_angle).astype(np.float32)
    result["target_minute_sin"] = np.sin(minute_angle).astype(np.float32)
    result["target_minute_cos"] = np.cos(minute_angle).astype(np.float32)
    result["target_dayofweek_sin"] = np.sin(day_angle).astype(np.float32)
    result["target_dayofweek_cos"] = np.cos(day_angle).astype(np.float32)
    result["target_is_weekend"] = (target_times.dayofweek >= 5).astype(np.float32)
    result["target_month"] = target_times.month.astype(np.float32)
    result["target_day_of_month"] = target_times.day.astype(np.float32)
    return result


def _evaluate_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    official = metrics.loc[metrics["label_variant"] == "official_interval_mean"].copy()
    control = config["variants"]["control"]
    candidate = config["variants"]["promotion_candidate"]
    gate = config["selection_gate"]

    def accuracy(scope: str, variant: str, *, fold: str = "all", bucket: str = "all", regime: str = "all") -> float:
        selected = official.loc[
            (official["scope"] == scope)
            & (official["variant"] == variant)
            & (official["fold_id"].astype(str) == str(fold))
            & (official["horizon_bucket"].astype(str) == str(bucket))
            & (official["origin_regime"].astype(str) == str(regime))
        ]
        if len(selected) != 1:
            raise AssertionError(f"metric lookup not unique: {scope}/{variant}/{fold}/{bucket}/{regime}")
        return float(selected.iloc[0]["accuracy_1_minus_mape"])

    overall_gain = (accuracy("overall", candidate) - accuracy("overall", control)) * 100
    fold_gains = {
        fold: (accuracy("fold", candidate, fold=fold) - accuracy("fold", control, fold=fold)) * 100
        for fold in config["validation"]["fold_ids"]
    }
    ordinary_loss = (accuracy("origin_regime", control, regime="ordinary") - accuracy("origin_regime", candidate, regime="ordinary")) * 100
    bucket_gains = {
        bucket: (accuracy("horizon_bucket", candidate, bucket=bucket) - accuracy("horizon_bucket", control, bucket=bucket)) * 100
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440")
    }
    control_episodes = official.loc[(official["scope"] == "episode") & (official["variant"] == control), ["episode_id", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
    candidate_episodes = official.loc[(official["scope"] == "episode") & (official["variant"] == candidate), ["episode_id", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "candidate_accuracy"})
    episodes = control_episodes.merge(candidate_episodes, on="episode_id", how="inner", validate="one_to_one")
    episode_fraction = float((episodes["candidate_accuracy"] > episodes["control_accuracy"]).mean()) if len(episodes) else np.nan
    point = metrics.loc[
        (metrics["label_variant"] == "legacy_point") & (metrics["scope"] == "overall"),
        ["variant", "accuracy_1_minus_mape"],
    ].set_index("variant")["accuracy_1_minus_mape"]
    point_gain = (float(point[candidate]) - float(point[control])) * 100
    pass_component = overall_gain >= float(gate["minimum_component_long_g1_accuracy_gain_pct"])
    pass_folds = min(fold_gains.values()) >= float(gate["minimum_single_fold_gain_pct"])
    pass_ordinary = ordinary_loss <= float(gate["maximum_ordinary_origin_accuracy_loss_pct"])
    pass_episodes = (not gate["require_majority_episode_improvement"]) or (np.isfinite(episode_fraction) and episode_fraction > 0.5)
    pass_near_far = min(bucket_gains["h015_120"], bucket_gains["h735_1440"]) >= -float(gate["maximum_near_far_accuracy_loss_pct"])
    passed = bool(pass_component and pass_folds and pass_ordinary and pass_episodes and pass_near_far)
    return pd.DataFrame([{
        "variant": candidate,
        "overall_accuracy_gain_pct": overall_gain,
        "legacy_point_gain_pct": point_gain,
        "single_fold_min_gain_pct": min(fold_gains.values()),
        "single_fold_mean_gain_pct": float(np.mean(list(fold_gains.values()))),
        "ordinary_accuracy_loss_pct": ordinary_loss,
        "episode_improvement_fraction": episode_fraction,
        "evaluated_episode_count": len(episodes),
        "near_gain_pct": bucket_gains["h015_120"],
        "mid1_gain_pct": bucket_gains["h135_360"],
        "mid2_gain_pct": bucket_gains["h375_720"],
        "far_gain_pct": bucket_gains["h735_1440"],
        **{f"{fold}_gain_pct": value for fold, value in fold_gains.items()},
        "pass_component_gain": pass_component,
        "pass_folds": pass_folds,
        "pass_ordinary": pass_ordinary,
        "pass_episodes": pass_episodes,
        "pass_near_far": pass_near_far,
        "component_gate_passed": passed,
        "standalone_submission_eligible": bool(passed and overall_gain >= float(gate["minimum_standalone_long_g1_accuracy_gain_pct"])),
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {
        key: PROJECT_ROOT / value
        for key, value in config["inputs"].items()
        if key != "raw_train_tables"
    }
    validated = _validate(config, paths)
    raw = merge_training_tables(validated["raw_tables"])
    grid = to_v28_grid(raw)
    features, feature_columns = build_v28_features(grid)
    if args.preflight_only:
        print(
            f"PASS preflight rows={len(grid)} origin_features={len(feature_columns)} "
            f"model_features={len(feature_columns) + len(TARGET_CALENDAR_COLUMNS)} "
            f"folds={len(validated['folds'])} models={config['budget']['expected_model_count']}"
        )
        return

    identity_inputs = [
        *validated["raw_tables"], *paths.values(), RUNNER_SOURCE, UTILS_SOURCE,
        BASELINE_UTILS_SOURCE, EVENT_UTILS_SOURCE, METRICS_SOURCE,
    ]
    run_id = _identity(config_path, identity_inputs)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    event_log = output_dir / "events.jsonl"
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    git_commit = _git_value("rev-parse", "HEAD")
    git_dirty = bool(_git_value("status", "--porcelain"))
    manifest: dict[str, Any] = {
        "run_id": run_id, "status": "running", "task": config["task"],
        "experiment_role": config["experiment_role"], "protocol_version": "round2_v3",
        "experiment_version": config["experiment_version"], "control_artifact_id": config["control_artifact_id"],
        "control_scientific_run_id": config["control_scientific_run_id"], "started_at": started_at.isoformat(),
        "holdout_evaluated": False, "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"commit": git_commit, "dirty": git_dirty},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "dependencies": {"lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__}},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, *identity_inputs]},
        "failure": {"type": None, "message": None, "traceback": None}, "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False
    try:
        feature_frame = features.copy()
        feature_frame.insert(0, "datetime", grid["datetime"].to_numpy())
        point_series = pd.to_numeric(grid.set_index("datetime")["generator_1"], errors="coerce")
        control = _load_control(paths["control_oof"], validated["folds"])
        split = pd.read_csv(paths["split_assignments"], parse_dates=["train_end", "validation_origin"])
        split = split.loc[(split["role"] == "model_selection") & split["fold_id"].isin(validated["folds"])].copy()
        if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
            raise AssertionError("fold order/content differs from preregistration")
        episodes = pd.read_csv(paths["event_episodes"])
        training_start = pd.Timestamp(config["training"]["g1_tree_start"])
        candidate_parts: list[pd.DataFrame] = []
        fits: list[dict[str, object]] = []
        groups = list(split.groupby("fold_id", sort=False))
        for fold_number, (fold_id, fold) in enumerate(groups, 1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            training_origins = pd.DatetimeIndex(grid.loc[(grid["datetime"] >= training_start) & (grid["datetime"] <= train_end), "datetime"])
            validation_origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
            x_train, train_meta = build_prediction_long(feature_frame, training_origins, validated["horizons"], feature_columns)
            x_train = _align_calendar_to_point_target(x_train, train_meta)
            train_target_times = pd.DatetimeIndex(train_meta["datetime"]) + pd.TimedeltaIndex(
                pd.to_timedelta(train_meta["horizon_minutes"].to_numpy(), unit="min")
            )
            y_train = point_series.reindex(train_target_times).to_numpy(dtype=float)
            valid_train = np.isfinite(y_train) & (train_target_times <= train_end)
            x_train = x_train.loc[valid_train].reset_index(drop=True)
            y_train = y_train[valid_train]
            x_validation, validation_meta = build_prediction_long(feature_frame, validation_origins, validated["horizons"], feature_columns)
            x_validation = _align_calendar_to_point_target(x_validation, validation_meta)
            model = lgb.LGBMRegressor(**dict(config["lightgbm"]))
            print(f"[{fold_number}/{len(groups)}] {fold_id}: train_rows={len(y_train)} validation_rows={len(x_validation)} features={x_train.shape[1]}", flush=True)
            model.fit(x_train, y_train)
            tree_prediction = np.clip(model.predict(x_validation), 0.0, None)
            fuel_frame = grid.loc[(grid["datetime"] >= pd.Timestamp(config["fuel_recipe"]["fit_start"])) & (grid["datetime"] <= train_end)]
            coefficients = fit_linear_fuel_proxy(fuel_frame, "generator_1", FUELS_G1)
            validation_grid = grid.set_index("datetime").loc[validation_origins].reset_index()
            fuel_by_origin = pd.Series(
                predict_linear_fuel(validation_grid, FUELS_G1, coefficients),
                index=validation_origins,
            )
            fuel_prediction = fuel_by_origin.reindex(pd.DatetimeIndex(validation_meta["datetime"])).to_numpy(dtype=float)
            horizons = validation_meta["horizon_minutes"].to_numpy(dtype=int)
            weights = np.where(horizons <= int(config["fuel_recipe"]["near_horizon_max_minutes"]), float(config["fuel_recipe"]["near_weight"]), float(config["fuel_recipe"]["far_weight"]))
            prediction = (1.0 - weights) * tree_prediction + weights * fuel_prediction
            point_times = pd.DatetimeIndex(validation_meta["datetime"]) + pd.TimedeltaIndex(
                pd.to_timedelta(horizons, unit="min")
            )
            actual_point = point_series.reindex(point_times).to_numpy(dtype=float)
            control_fold = control.loc[control["fold_id"] == fold_id].sort_values(["datetime", "horizon_minutes"]).reset_index(drop=True)
            candidate = validation_meta.copy()
            candidate["fold_id"] = fold_id
            candidate["actual_legacy_point"] = actual_point
            candidate["prediction"] = prediction
            candidate["variant"] = config["variants"]["promotion_candidate"]
            key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
            candidate = candidate.merge(control_fold[key + ["actual_interval_mean"]], on=key, how="left", validate="one_to_one")
            if candidate["actual_interval_mean"].isna().any():
                raise AssertionError(f"missing official labels after control join: {fold_id}")
            control_point = control_fold.set_index(key)["actual_legacy_point"]
            candidate_point = candidate.set_index(key)["actual_legacy_point"]
            np.testing.assert_allclose(
                candidate_point.loc[control_point.index].to_numpy(dtype=float),
                control_point.to_numpy(dtype=float),
                rtol=0,
                atol=1e-10,
            )
            candidate_parts.append(candidate[["variant", *key, "actual_interval_mean", "actual_legacy_point", "prediction"]])
            fits.append({"fold_id": fold_id, "training_rows": len(y_train), "validation_rows": len(x_validation), "origin_feature_count": len(feature_columns), "model_feature_count": x_train.shape[1], "training_origin_start": training_origins.min(), "training_origin_end": train_end})
            _append_event(event_log, "fold_completed", fold_id=fold_id, training_rows=len(y_train), validation_rows=len(x_validation))
        oof = pd.concat([control, *candidate_parts], ignore_index=True)
        expected_rows = len(split) * len(validated["horizons"]) * 2
        if len(oof) != expected_rows:
            raise AssertionError(f"OOF row count {len(oof)} != {expected_rows}")
        oof = attach_event_context(oof, episodes, context_hours=24)
        metrics = _metric_tables(oof)
        gates = _evaluate_gate(metrics, config)
        official = metrics.loc[(metrics["label_variant"] == "official_interval_mean") & (metrics["scope"] == "overall")].copy()
        control_accuracy = float(official.loc[official["variant"] == config["variants"]["control"], "accuracy_1_minus_mape"].iloc[0])
        official["delta_accuracy_pct"] = (official["accuracy_1_minus_mape"] - control_accuracy) * 100
        comparison = official[["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]].sort_values("mape")
        frames = {"oof_predictions.csv": oof, "fit_summary.csv": pd.DataFrame(fits), "metrics.csv": metrics, "variant_comparison.csv": comparison, "gate_results.csv": gates}
        artifacts: dict[str, str] = {}
        for name, frame in frames.items():
            path = output_dir / name
            frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
            stem = name.removesuffix(".csv")
            artifacts[stem] = path.relative_to(PROJECT_ROOT).as_posix()
            artifacts[stem + "_sha256"] = sha256(path)
        gate_row = gates.iloc[0]
        candidate_mape = float(comparison.loc[comparison["variant"] == config["variants"]["promotion_candidate"], "mape"].iloc[0])
        ended_at = datetime.now(timezone.utc)
        manifest.update({"status": "completed", "ended_at": ended_at.isoformat(), "duration_seconds": round(time.perf_counter() - started_clock, 3), "model_count": len(groups), "origin_feature_count": len(feature_columns), "model_feature_count": len(feature_columns) + len(TARGET_CALENDAR_COLUMNS), "model_parameters": dict(config["lightgbm"]), "primary_variant": config["variants"]["promotion_candidate"], "primary_long_g1_mape": candidate_mape, "component_gate_passed": bool(gate_row["component_gate_passed"]), "standalone_submission_eligible": bool(gate_row["standalone_submission_eligible"]), "artifacts": artifacts})
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "label_version": validated["label_manifest"]["artifact_version"], "split_sha256": validated["split_manifest"]["assignment_sha256"], "control_run_id": config["control_scientific_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "v28_long_g1_shared_horizon", "long_g1_mape": candidate_mape, "holdout_evaluated": False, "submission_eligible": False, "git_commit": git_commit, "git_dirty": git_dirty, "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat(), "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"component_gate_passed={bool(gate_row['component_gate_passed'])}; scientific OOF only"})
            registry_written = True
        _append_event(event_log, "run_completed", duration_seconds=manifest["duration_seconds"], component_gate_passed=bool(gate_row["component_gate_passed"]))
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={bool(gate_row['component_gate_passed'])} standalone_submission_eligible={bool(gate_row['standalone_submission_eligible'])}", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "ended_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(time.perf_counter() - started_clock, 3), "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", error_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_scientific_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "v28_long_g1_shared_horizon", "holdout_evaluated": False, "submission_eligible": False, "git_commit": git_commit, "git_dirty": git_dirty, "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"failure={type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
