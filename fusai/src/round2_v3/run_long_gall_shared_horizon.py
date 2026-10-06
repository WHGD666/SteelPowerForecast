"""Isolate a shared-horizon tree inside the legacy v20 long-gall recipe."""

from __future__ import annotations

import argparse
import csv
import gc
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
    GALL_FUEL_COLUMNS,
    build_v16_legacy_features,
    causal_low_fuel_ratio,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
)
from src.round2_v3.baseline_utils import TARGET_CALENDAR_COLUMNS, build_prediction_long
from src.round2_v3.event_ablation_utils import attach_event_context, summarize_ablation
from src.round2_v3.run_long_g1_shared_horizon import _align_calendar_to_point_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/long_gall_shared_horizon_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("anomaly_weight_utils.py")
BASELINE_UTILS_SOURCE = RUNNER_SOURCE.with_name("baseline_utils.py")
EVENT_UTILS_SOURCE = RUNNER_SOURCE.with_name("event_ablation_utils.py")
SHARED_G1_SOURCE = RUNNER_SOURCE.with_name("run_long_g1_shared_horizon.py")
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
        + "_long_gall_shared_"
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
    training = config["training"]
    if training["feature_engine"] != "v16_legacy_columns":
        raise AssertionError("generator_all must remain on the safe old-column engine")
    if training["target_history_features_allowed"] or training["future_process_observations_allowed"]:
        raise AssertionError("future process/target-history features are forbidden")
    if list(training["global_tree_features"]) != list(TARGET_CALENDAR_COLUMNS):
        raise AssertionError("shared-horizon position features differ from contract")
    recipe = config["v20_recipe"]
    expected_weights = (0.5, 0.5, 0.225, 0.075, 0.7)
    actual_weights = tuple(float(recipe[name]) for name in (
        "near_tree_weight", "near_fuel_weight", "far_tree_weight",
        "far_climatology_weight", "far_fuel_weight",
    ))
    if actual_weights != expected_weights:
        raise AssertionError(f"v20 mixture weights differ: {actual_weights}")
    if int(recipe["near_max_step"]) != 32:
        raise AssertionError("v20 near/far boundary must remain 32 steps")
    if float(recipe["gate_trigger_absolute_ratio_delta"]) != 0.20:
        raise AssertionError("v20 gate threshold must remain 0.20")
    if (float(recipe["gate_clip_min"]), float(recipe["gate_clip_max"])) != (0.8, 1.2):
        raise AssertionError("v20 gate clip must remain [0.8, 1.2]")
    horizons = [int(value) for value in config["horizons_minutes"]]
    folds = list(config["validation"]["fold_ids"])
    if horizons != list(range(15, 1441, 15)):
        raise AssertionError("long horizons must remain 15..1440 by 15")
    expected_models = len(folds) * (len(horizons) + 1)
    if expected_models != int(config["budget"]["expected_model_count"]):
        raise AssertionError("model budget is inconsistent")
    return {
        "raw_tables": raw_tables,
        "label_manifest": label_manifest,
        "label_path": label_path,
        "split_manifest": split_manifest,
        "folds": folds,
        "horizons": horizons,
    }


def _climatology(frame: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
    source = frame[["datetime", "generator_all"]].dropna().copy()
    source["dow"] = source["datetime"].dt.dayofweek
    source["hour"] = source["datetime"].dt.hour
    profile = source.groupby(["dow", "hour"])["generator_all"].mean()
    hour_profile = source.groupby("hour")["generator_all"].mean()
    overall = float(source["generator_all"].mean())
    return np.asarray(
        [
            profile.get((value.dayofweek, value.hour), hour_profile.get(value.hour, overall))
            for value in origins
        ],
        dtype=float,
    )


def _v20_mix(
    tree_prediction: np.ndarray,
    fuel_prediction: np.ndarray,
    climatology_prediction: np.ndarray,
    horizons_minutes: np.ndarray,
    gate_ratio: np.ndarray,
    recipe: dict[str, Any],
) -> np.ndarray:
    tree = np.asarray(tree_prediction, dtype=float)
    fuel = np.asarray(fuel_prediction, dtype=float)
    climate = np.asarray(climatology_prediction, dtype=float)
    horizons = np.asarray(horizons_minutes, dtype=int)
    ratio = np.asarray(gate_ratio, dtype=float)
    if not (tree.shape == fuel.shape == climate.shape == horizons.shape == ratio.shape):
        raise ValueError("v20 mixture inputs have different shapes")
    steps = horizons // 15
    near = steps <= int(recipe["near_max_step"])
    prediction = np.empty_like(tree)
    prediction[near] = (
        float(recipe["near_tree_weight"]) * tree[near]
        + float(recipe["near_fuel_weight"]) * fuel[near]
    )
    prediction[~near] = (
        float(recipe["far_tree_weight"]) * tree[~near]
        + float(recipe["far_climatology_weight"]) * climate[~near]
        + float(recipe["far_fuel_weight"]) * fuel[~near]
    )
    fire = np.isfinite(ratio) & (
        np.abs(ratio - 1.0) > float(recipe["gate_trigger_absolute_ratio_delta"])
    )
    scale = np.clip(
        ratio,
        float(recipe["gate_clip_min"]),
        float(recipe["gate_clip_max"]),
    )
    prediction[fire] = prediction[fire] * scale[fire]
    return np.clip(prediction, 0.0, None)


def _metric_tables(oof: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for label_variant, column in (
        ("official_interval_mean", "actual_interval_mean"),
        ("legacy_point", "actual_legacy_point"),
    ):
        source = oof.copy()
        source["actual"] = source[column]
        metrics = summarize_ablation(source)
        metrics.insert(0, "label_variant", label_variant)
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def _evaluate_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    official = metrics.loc[metrics["label_variant"] == "official_interval_mean"].copy()
    control = config["variants"]["control"]
    candidate = config["variants"]["promotion_candidate"]
    gate = config["selection_gate"]

    def accuracy(scope: str, variant: str, *, fold: str = "all", bucket: str = "all", regime: str = "all") -> float:
        selected = official.loc[
            (official["scope"] == scope) & (official["variant"] == variant)
            & (official["fold_id"].astype(str) == str(fold))
            & (official["horizon_bucket"].astype(str) == str(bucket))
            & (official["origin_regime"].astype(str) == str(regime))
        ]
        if len(selected) != 1:
            raise AssertionError(f"metric lookup not unique: {scope}/{variant}/{fold}/{bucket}/{regime}")
        return float(selected.iloc[0]["accuracy_1_minus_mape"])

    gain = (accuracy("overall", candidate) - accuracy("overall", control)) * 100
    fold_gains = {
        fold: (accuracy("fold", candidate, fold=fold) - accuracy("fold", control, fold=fold)) * 100
        for fold in config["validation"]["fold_ids"]
    }
    ordinary_loss = (accuracy("origin_regime", control, regime="ordinary") - accuracy("origin_regime", candidate, regime="ordinary")) * 100
    bucket_gains = {
        bucket: (accuracy("horizon_bucket", candidate, bucket=bucket) - accuracy("horizon_bucket", control, bucket=bucket)) * 100
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440")
    }
    ce = official.loc[(official["scope"] == "episode") & (official["variant"] == control), ["episode_id", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "control"})
    pe = official.loc[(official["scope"] == "episode") & (official["variant"] == candidate), ["episode_id", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "candidate"})
    episodes = ce.merge(pe, on="episode_id", how="inner", validate="one_to_one")
    episode_fraction = float((episodes["candidate"] > episodes["control"]).mean()) if len(episodes) else np.nan
    point = metrics.loc[(metrics["label_variant"] == "legacy_point") & (metrics["scope"] == "overall"), ["variant", "accuracy_1_minus_mape"]].set_index("variant")["accuracy_1_minus_mape"]
    point_gain = (float(point[candidate]) - float(point[control])) * 100
    pass_component = gain >= float(gate["minimum_component_long_gall_accuracy_gain_pct"])
    pass_combination = gain >= float(gate["minimum_combination_long_gall_accuracy_gain_pct"])
    pass_folds = min(fold_gains.values()) >= float(gate["minimum_single_fold_gain_pct"])
    pass_ordinary = ordinary_loss <= float(gate["maximum_ordinary_origin_accuracy_loss_pct"])
    pass_episodes = (not gate["require_majority_episode_improvement"]) or (np.isfinite(episode_fraction) and episode_fraction > 0.5)
    pass_near_far = min(bucket_gains["h015_120"], bucket_gains["h735_1440"]) >= -float(gate["maximum_near_far_accuracy_loss_pct"])
    passed = bool(pass_component and pass_folds and pass_ordinary and pass_episodes and pass_near_far)
    return pd.DataFrame([{
        "variant": candidate, "overall_accuracy_gain_pct": gain,
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
        "pass_component_gain": pass_component, "pass_combination_gain": pass_combination,
        "pass_folds": pass_folds, "pass_ordinary": pass_ordinary,
        "pass_episodes": pass_episodes, "pass_near_far": pass_near_far,
        "component_gate_passed": passed,
        "combination_candidate_eligible": bool(passed and pass_combination),
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
    features, feature_columns = build_v16_legacy_features(grid)
    forbidden = {"blast_furnace_3", "air_heater_3"}
    if forbidden & set(feature_columns):
        raise AssertionError("unsafe restored gall columns entered the feature set")
    if args.preflight_only:
        print(
            f"PASS preflight rows={len(grid)} old_features={len(feature_columns)} "
            f"shared_features={len(feature_columns) + len(TARGET_CALENDAR_COLUMNS)} "
            f"models={config['budget']['expected_model_count']} restored_gall_columns=false"
        )
        return

    identity_inputs = [
        *validated["raw_tables"], *paths.values(), RUNNER_SOURCE, UTILS_SOURCE,
        BASELINE_UTILS_SOURCE, EVENT_UTILS_SOURCE, SHARED_G1_SOURCE, METRICS_SOURCE,
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
        "started_at": started_at.isoformat(), "holdout_evaluated": False,
        "submission_eligible": False,
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
        timestamps = pd.DatetimeIndex(grid["datetime"])
        positions = pd.Series(np.arange(len(grid), dtype=int), index=timestamps)
        point_series = pd.to_numeric(grid.set_index("datetime")["generator_all"], errors="coerce")
        interval = pd.read_csv(validated["label_path"], parse_dates=["datetime"])
        interval_series = pd.to_numeric(interval.set_index("datetime")["generator_all"], errors="coerce")
        split = pd.read_csv(paths["split_assignments"], parse_dates=["train_end", "validation_origin"])
        split = split.loc[(split["role"] == "model_selection") & split["fold_id"].isin(validated["folds"])].copy()
        if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
            raise AssertionError("fold order/content differs from preregistration")
        episodes = pd.read_csv(paths["event_episodes"])
        tree_start = pd.Timestamp(config["training"]["tree_start"])
        recipe = config["v20_recipe"]
        predictions: list[pd.DataFrame] = []
        fits: list[dict[str, object]] = []
        groups = list(split.groupby("fold_id", sort=False))
        for fold_number, (fold_id, fold) in enumerate(groups, 1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            cutoff = int(np.flatnonzero(timestamps <= train_end)[-1])
            origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
            validation_positions = positions.loc[origins].to_numpy(dtype=int)
            x_validation_shared, validation_meta = build_prediction_long(feature_frame, origins, validated["horizons"], feature_columns)
            x_validation_shared = _align_calendar_to_point_target(x_validation_shared, validation_meta)
            training_origins = pd.DatetimeIndex(grid.loc[(grid["datetime"] >= tree_start) & (grid["datetime"] <= train_end), "datetime"])
            x_train_shared, train_meta = build_prediction_long(feature_frame, training_origins, validated["horizons"], feature_columns)
            x_train_shared = _align_calendar_to_point_target(x_train_shared, train_meta)
            train_target_times = pd.DatetimeIndex(train_meta["datetime"]) + pd.TimedeltaIndex(pd.to_timedelta(train_meta["horizon_minutes"].to_numpy(), unit="min"))
            y_shared = point_series.reindex(train_target_times).to_numpy(dtype=float)
            valid_shared = np.isfinite(y_shared) & (train_target_times <= train_end)
            x_train_shared = x_train_shared.loc[valid_shared].reset_index(drop=True)
            y_shared = y_shared[valid_shared]
            params = dict(config["lightgbm"])
            print(f"[{fold_number}/{len(groups)}] {fold_id}: shared_train_rows={len(y_shared)} validation_rows={len(x_validation_shared)}", flush=True)
            shared_model = lgb.LGBMRegressor(**params)
            shared_model.fit(x_train_shared, y_shared)
            shared_tree = np.clip(shared_model.predict(x_validation_shared), 0.0, None)
            del x_train_shared, y_shared, shared_model
            gc.collect()

            control_by_horizon: dict[int, np.ndarray] = {}
            for number, horizon in enumerate(validated["horizons"], 1):
                step = horizon // 15
                eligible = np.flatnonzero((timestamps >= tree_start) & (np.arange(len(grid)) + step <= cutoff))
                y_train = grid["generator_all"].to_numpy(dtype=float)[eligible + step]
                valid = np.isfinite(y_train)
                eligible, y_train = eligible[valid], y_train[valid]
                model = lgb.LGBMRegressor(**params)
                model.fit(features.iloc[eligible][feature_columns], y_train)
                control_by_horizon[horizon] = np.clip(model.predict(features.iloc[validation_positions][feature_columns]), 0.0, None)
                if number % 16 == 0:
                    print(f"  control horizons={number}/96 latest_h={horizon}", flush=True)

            fuel_train = grid.loc[(grid["datetime"] >= pd.Timestamp(recipe["fuel_fit_start"])) & (grid["datetime"] <= train_end)]
            fuel_coefficients = fit_linear_fuel_proxy(fuel_train, "generator_all", GALL_FUEL_COLUMNS)
            validation_grid = grid.set_index("datetime").loc[origins].reset_index()
            fuel_origin = predict_linear_fuel(validation_grid, GALL_FUEL_COLUMNS, fuel_coefficients)
            climate_origin = _climatology(grid.iloc[: cutoff + 1], origins)
            gate_train = grid.loc[(grid["datetime"] >= pd.Timestamp(recipe["gate_fit_start"])) & (grid["datetime"] <= train_end)]
            gate_coefficients = fit_linear_fuel_proxy(gate_train, "generator_all", GALL_FUEL_COLUMNS)
            gate_timeline = grid.iloc[: int(validation_positions.max()) + 1]
            ratio_timeline = causal_low_fuel_ratio(
                gate_timeline,
                gate_coefficients,
                smooth_steps=int(recipe["gate_smooth_steps"]),
                baseline_steps=int(recipe["gate_baseline_steps"]),
                baseline_min_periods=int(recipe["gate_baseline_min_periods"]),
            )
            ratio_origin = ratio_timeline.iloc[validation_positions].to_numpy(dtype=float)

            horizons_flat = validation_meta["horizon_minutes"].to_numpy(dtype=int)
            fuel_flat = pd.Series(fuel_origin, index=origins).reindex(pd.DatetimeIndex(validation_meta["datetime"])).to_numpy(dtype=float)
            climate_flat = pd.Series(climate_origin, index=origins).reindex(pd.DatetimeIndex(validation_meta["datetime"])).to_numpy(dtype=float)
            ratio_flat = pd.Series(ratio_origin, index=origins).reindex(pd.DatetimeIndex(validation_meta["datetime"])).to_numpy(dtype=float)
            control_tree = np.concatenate([control_by_horizon[h] for h in validated["horizons"]]).reshape(len(validated["horizons"]), len(origins)).T.reshape(-1)
            if control_tree.shape != shared_tree.shape:
                raise AssertionError("control/shared tree shapes differ")
            actual_point_times = pd.DatetimeIndex(validation_meta["datetime"]) + pd.TimedeltaIndex(pd.to_timedelta(horizons_flat, unit="min"))
            actual_point = point_series.reindex(actual_point_times).to_numpy(dtype=float)
            actual_interval = interval_series.reindex(pd.DatetimeIndex(validation_meta["interval_start"])).to_numpy(dtype=float)
            for variant, tree_prediction in (
                (config["variants"]["control"], control_tree),
                (config["variants"]["promotion_candidate"], shared_tree),
            ):
                final_prediction = _v20_mix(tree_prediction, fuel_flat, climate_flat, horizons_flat, ratio_flat, recipe)
                predictions.append(pd.DataFrame({
                    "variant": variant, "fold_id": fold_id,
                    "datetime": validation_meta["datetime"],
                    "interval_start": validation_meta["interval_start"],
                    "horizon_minutes": horizons_flat,
                    "actual_interval_mean": actual_interval,
                    "actual_legacy_point": actual_point,
                    "prediction": final_prediction,
                }))
            fits.append({"fold_id": fold_id, "shared_training_rows": len(train_meta.loc[valid_shared]), "validation_rows": len(validation_meta), "old_origin_feature_count": len(feature_columns), "shared_model_feature_count": len(feature_columns) + len(TARGET_CALENDAR_COLUMNS), "gate_fire_origin_count": int((np.isfinite(ratio_origin) & (np.abs(ratio_origin - 1) > float(recipe["gate_trigger_absolute_ratio_delta"]))).sum()), "training_origin_start": tree_start, "training_origin_end": train_end})
            _append_event(event_log, "fold_completed", fold_id=fold_id, shared_training_rows=int(valid_shared.sum()), validation_rows=len(validation_meta))

        oof = pd.concat(predictions, ignore_index=True)
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
        manifest.update({"status": "completed", "ended_at": ended_at.isoformat(), "duration_seconds": round(time.perf_counter() - started_clock, 3), "model_count": int(config["budget"]["expected_model_count"]), "old_origin_feature_count": len(feature_columns), "shared_model_feature_count": len(feature_columns) + len(TARGET_CALENDAR_COLUMNS), "model_parameters": dict(config["lightgbm"]), "primary_variant": config["variants"]["promotion_candidate"], "primary_long_gall_mape": candidate_mape, "component_gate_passed": bool(gate_row["component_gate_passed"]), "combination_candidate_eligible": bool(gate_row["combination_candidate_eligible"]), "artifacts": artifacts})
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "label_version": validated["label_manifest"]["artifact_version"], "split_sha256": validated["split_manifest"]["assignment_sha256"], "control_run_id": config["control_artifact_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "v20_long_gall_shared_horizon", "long_gall_mape": candidate_mape, "holdout_evaluated": False, "submission_eligible": False, "git_commit": git_commit, "git_dirty": git_dirty, "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat(), "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"component_gate_passed={bool(gate_row['component_gate_passed'])}; combination_candidate_eligible={bool(gate_row['combination_candidate_eligible'])}; scientific OOF only"})
            registry_written = True
        _append_event(event_log, "run_completed", duration_seconds=manifest["duration_seconds"], component_gate_passed=bool(gate_row["component_gate_passed"]))
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={bool(gate_row['component_gate_passed'])} combination_candidate_eligible={bool(gate_row['combination_candidate_eligible'])}", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "ended_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(time.perf_counter() - started_clock, 3), "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", error_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_artifact_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "v20_long_gall_shared_horizon", "holdout_evaluated": False, "submission_eligible": False, "git_commit": git_commit, "git_dirty": git_dirty, "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"failure={type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
