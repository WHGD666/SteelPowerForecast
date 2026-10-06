"""IronFlow experiment v7/v8/v9 runner: target-definition modes and enriched features.

Supported controlled changes (exactly one per config, everything else frozen to v2):
- ``target_definition.mode``: ``level`` | ``residual_additive`` | ``ratio``.  The
  persistence baseline for training rows and validation origins is the origin-visible
  ``feat_current_{target}`` column only, so every transformation stays causal.
- ``features.feature_set``: ``v2_physics`` | ``v9_enriched``.  ``v9_enriched`` appends
  cross-target, multi-scale volatility, gas-holder linkage, generator-gas structure
  and intraday-harmonic features.  All features look backwards only.

Failure discipline: any exception keeps produced artifacts, writes ``failure.json``
into the failure run directory when it exists, appends a registry failure row and
exits non-zero.  A 30-minute wall-clock budget is enforced.  Immutable run ids and
registry uniqueness/monotonicity rules are identical to v2.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_baseline import (  # noqa: E402
    add_metric_rows,
    calculate_metrics,
    file_hash,
    git_state,
    json_hash,
    json_safe,
    validate_split_fingerprint,
)
from run_experiment_v2 import (  # noqa: E402
    make_feature_frame_v2,
)
from stage_config import build_horizon_groups, resolve_modeling_strategy  # noqa: E402

SUPPORTED_TARGET_MODES = ("level", "residual_additive", "ratio")
SUPPORTED_FEATURE_SETS = ("v2_physics", "v9_enriched")
# The effective wall-clock budget is resolved from config ``time_budget_seconds``
# (stage_config.resolve_modeling_strategy).  This constant is kept only as the
# documented prelim-v1 compatibility reference.
FAILURE_TIME_LIMIT_SECONDS = 30.0 * 60.0

REGISTRY_COLUMNS = [
    "run_id",
    "created_at_utc",
    "git_commit",
    "git_dirty",
    "config_sha256",
    "prepared_train_sha256",
    "split_sha256",
    "best_development_model",
    "best_development_mape",
    "best_development_score_1_mape",
    "duration_seconds",
    "artifact_directory",
    "sealed_holdout_evaluated",
]


def make_feature_frame_v9(
    frame: pd.DataFrame,
    prepared_feature_columns: list[str],
    targets: list[str],
    target_lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """v2 physics block plus the registered v9 enrichment, all backward-looking."""
    features, _ = make_feature_frame_v2(frame, prepared_feature_columns, targets, target_lags)
    new_columns: list[str] = []

    if {"generator_1", "generator_all"}.issubset(targets):
        cross = frame["generator_all"] - frame["generator_1"]
        features["feat_generator_all_minus_generator_1"] = cross
        features["feat_generator_all_minus_generator_1_diff_1"] = cross.diff(1)
        for window in (4, 16):
            features[f"feat_generator_all_minus_generator_1_roll_mean_{window}"] = cross.rolling(
                window, min_periods=1
            ).mean()
        new_columns.extend(
            [
                "feat_generator_all_minus_generator_1",
                "feat_generator_all_minus_generator_1_diff_1",
                "feat_generator_all_minus_generator_1_roll_mean_4",
                "feat_generator_all_minus_generator_1_roll_mean_16",
            ]
        )

    for target in targets:
        series = frame[target]
        for window in (8, 32):
            features[f"feat_{target}_roll_std_{window}"] = series.rolling(window, min_periods=1).std()
            new_columns.append(f"feat_{target}_roll_std_{window}")

    holder_column = "blast_furnace_gas_holder_2"
    if holder_column not in frame:
        raise AssertionError(f"missing gas-holder column for v9 linkage feature: {holder_column}")
    holder_delta = frame[holder_column].diff(4)
    for target in targets:
        target_delta = frame[target].diff(4)
        same_direction = (
            ((holder_delta > 0) & (target_delta > 0)) | ((holder_delta < 0) & (target_delta < 0))
        ).astype(float)
        features[f"feat_holder_vs_{target}_corr_proxy"] = same_direction.rolling(16, min_periods=1).mean()
        new_columns.append(f"feat_holder_vs_{target}_corr_proxy")

    for column in (
        "generator_use_blast_furnace_gas",
        "generator_use_coke_gas",
        "generator_use_converter_gas",
    ):
        if column not in frame:
            raise AssertionError(f"missing generator gas-use column for v9 proxy: {column}")
        for window in (4, 16):
            features[f"feat_{column}_roll_mean_{window}"] = frame[column].rolling(
                window, min_periods=1
            ).mean()
            new_columns.append(f"feat_{column}_roll_mean_{window}")

    minute_of_day = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    hours_since_day_start = minute_of_day / 60.0
    for harmonic, angle_scale in ((1, 1.0), (2, 2.0)):
        angle = 2.0 * np.pi * angle_scale * hours_since_day_start / 24.0
        features[f"feat_hours_since_day_start_sin_{harmonic}"] = np.sin(angle)
        features[f"feat_hours_since_day_start_cos_{harmonic}"] = np.cos(angle)
        new_columns.extend(
            [
                f"feat_hours_since_day_start_sin_{harmonic}",
                f"feat_hours_since_day_start_cos_{harmonic}",
            ]
        )

    missing_new = [column for column in new_columns if column not in features]
    if missing_new:
        raise AssertionError(f"v9 feature construction missing columns: {missing_new}")
    if not all(column.startswith("feat_") for column in new_columns):
        raise AssertionError("v9 enrichment produced non-prefixed feature names")
    non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(features[name])]
    if non_numeric:
        raise AssertionError(f"non-numeric model features: {non_numeric}")
    return features, list(features.columns)


def validate_config(config: dict[str, Any]) -> tuple[str, str]:
    target_definition = config.get("target_definition")
    if not isinstance(target_definition, dict):
        raise AssertionError("config must define a target_definition block")
    mode = target_definition.get("mode")
    if mode not in SUPPORTED_TARGET_MODES:
        raise AssertionError(
            f"unsupported target_definition.mode: {mode}; allowed {list(SUPPORTED_TARGET_MODES)}"
        )
    if target_definition.get("baseline") != "persistence":
        raise AssertionError("target_definition.baseline must be persistence in v7 runner")
    if target_definition.get("baseline_source") != "feat_current_target":
        raise AssertionError("target_definition.baseline_source must be feat_current_target")
    floor_epsilon = float(target_definition.get("floor_epsilon", 0.0))
    if not (floor_epsilon > 0):
        raise AssertionError("target_definition.floor_epsilon must be positive")
    feature_set = config["features"]["feature_set"]
    if feature_set not in SUPPORTED_FEATURE_SETS:
        raise AssertionError(
            f"unsupported features.feature_set: {feature_set}; allowed {list(SUPPORTED_FEATURE_SETS)}"
        )
    if (config.get("training") or {}).get("target_transform") not in (None,):
        raise AssertionError("v7 runner does not support training.target_transform; use target_definition")
    weight_policy = (config.get("sample_weight") or {}).get("policy")
    if weight_policy not in (None, "inverse_actual_clipped"):
        raise AssertionError(f"unsupported sample weight policy: {weight_policy}")
    if float((config.get("sample_weight") or {}).get("floor", 1.0)) <= 0:
        raise AssertionError("sample_weight.floor must be positive")
    return mode, feature_set


def append_registry_record(root: Path, record: dict[str, Any]) -> None:
    registry_path = root / "experiments" / "registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [column for column in REGISTRY_COLUMNS if column not in record]
    if missing:
        raise AssertionError(f"registry record missing columns: {missing}")
    row = pd.DataFrame([record], columns=REGISTRY_COLUMNS)
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        if record["run_id"] in set(registry["run_id"]):
            raise AssertionError(f"run id already exists in experiment registry: {record['run_id']}")
        observed_times = pd.to_datetime(registry["created_at_utc"], utc=True, format="mixed")
        current_time = pd.to_datetime([record["created_at_utc"]], utc=True, format="mixed")
        if current_time[0] <= observed_times.max():
            raise AssertionError("registry created_at_utc must be strictly increasing")
        registry = pd.concat([registry, row], ignore_index=True)
    else:
        registry = row
    registry.to_csv(registry_path, index=False, encoding="utf-8", lineterminator="\n")


def extract_assumption(config_path: Path) -> str | None:
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lstrip("#").strip()
            if "预登记假设" in stripped:
                return stripped
    except OSError:
        return None
    return None


def record_failure(
    root: Path,
    config_path: Path,
    context: dict[str, Any],
    started: float,
    exc: Exception,
) -> int:
    duration = time.perf_counter() - started
    run_id = context.get("run_id") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_nogit_failed"
    )
    output_dir = context.get("output_dir")
    artifact_directory = None
    if output_dir is not None and Path(output_dir).exists():
        artifact_directory = Path(output_dir).resolve().relative_to(root).as_posix()
        failure_payload = {
            "run_id": run_id,
            "status": "failure",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_version": (context.get("config") or {}).get("experiment_version"),
            "assumption": extract_assumption(config_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "completed_units": context.get("completed_units", 0),
            "duration_seconds": duration,
            "fingerprint_state": {
                "config_sha256": context.get("config_sha256"),
                "split_sha256": context.get("split_sha256"),
                "prepared_train_sha256": context.get("prepared_train_sha256"),
                "git_commit": context.get("git_commit"),
                "git_dirty": context.get("git_dirty"),
            },
        }
        (Path(output_dir) / "failure.json").write_text(
            json.dumps(failure_payload, ensure_ascii=False, indent=2, default=json_safe),
            encoding="utf-8",
        )
    record = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": context.get("git_commit"),
        "git_dirty": context.get("git_dirty"),
        "config_sha256": context.get("config_sha256"),
        "prepared_train_sha256": context.get("prepared_train_sha256"),
        "split_sha256": context.get("split_sha256"),
        "best_development_model": "experiment_failed",
        "best_development_mape": None,
        "best_development_score_1_mape": None,
        "duration_seconds": duration,
        "artifact_directory": artifact_directory,
        "sealed_holdout_evaluated": False,
    }
    try:
        append_registry_record(root, record)
    except Exception as registry_exc:  # noqa: BLE001
        print(f"FAILED registry append error: {registry_exc}")
        print(f"FAILED run_id={run_id} error_type={type(exc).__name__} error={exc}")
        return 1
    print(f"FAILED run_id={run_id} error_type={type(exc).__name__} error={exc}")
    print(f"FAILURE RECORDED artifact={artifact_directory or '<no run directory>'}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_v7_resid_additive.yaml"))
    parser.add_argument("--run-id", help="Optional immutable run id; must not already exist")
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    context: dict[str, Any] = {
        "run_id": None,
        "output_dir": None,
        "config": None,
        "config_sha256": None,
        "split_sha256": None,
        "prepared_train_sha256": None,
        "git_commit": None,
        "git_dirty": None,
        "completed_units": 0,
    }
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        context["config"] = config
        context["config_sha256"] = file_hash(config_path)
        target_mode, _feature_set = validate_config(config)

        split_config_path = root / config["split_config"]
        split_config = yaml.safe_load(split_config_path.read_text(encoding="utf-8"))
        split_path = root / config["split_assignment_file"]
        split_fingerprint_path = root / split_config["output"]["fingerprint_file"]
        split_fingerprint = validate_split_fingerprint(split_path, split_fingerprint_path)
        context["split_sha256"] = split_fingerprint
        assignments = pd.read_csv(
            split_path,
            parse_dates=["train_end", "origin", "max_target_time"],
        )
        development = assignments[assignments["role"] == "development"].copy()
        if development.empty:
            raise AssertionError("no development assignments found")

        prepared_dir = root / config["prepared_directory"]
        train_path = prepared_dir / config["prepared_train_file"]
        preparation_manifest_path = prepared_dir / config["preparation_manifest_file"]
        preparation_manifest = json.loads(preparation_manifest_path.read_text(encoding="utf-8"))
        expected_train_hash = preparation_manifest["outputs"][train_path.relative_to(root).as_posix()]
        observed_train_hash = file_hash(train_path)
        if observed_train_hash != expected_train_hash:
            raise AssertionError("prepared training table hash differs from preparation manifest")
        context["prepared_train_sha256"] = observed_train_hash

        frame = pd.read_csv(train_path, parse_dates=["datetime"], low_memory=False)
        if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
            raise AssertionError("prepared training timestamps must be unique and increasing")
        expected_delta = pd.Timedelta(minutes=int(split_config["frequency_minutes"]))
        if not frame["datetime"].diff().dropna().eq(expected_delta).all():
            raise AssertionError("prepared training table is not on the frozen 15-minute grid")

        targets = list(config["targets"])
        horizons = [int(value) for value in config["horizons_steps"]]
        strategy_spec = resolve_modeling_strategy(config)
        strategy = strategy_spec["strategy"]
        group_size = strategy_spec["horizon_group_size"]
        budget_seconds = strategy_spec["time_budget_seconds"]
        period = int(config["seasonal_day"]["period_steps"])
        target_lags = [int(value) for value in config["features"]["target_lags_steps"]]
        prepared_features = list(preparation_manifest["input_feature_columns"])
        if config["features"]["feature_set"] == "v9_enriched":
            features, feature_names = make_feature_frame_v9(frame, prepared_features, targets, target_lags)
        else:
            features, feature_names = make_feature_frame_v2(frame, prepared_features, targets, target_lags)
        time_to_position = pd.Series(frame.index.to_numpy(), index=frame["datetime"])

        missing_origins = sorted(set(development["origin"]) - set(time_to_position.index))
        missing_targets = sorted(set(development["max_target_time"]) - set(time_to_position.index))
        if missing_origins or missing_targets:
            raise AssertionError(
                f"development split falls outside prepared data: origins={missing_origins[:3]}, "
                f"targets={missing_targets[:3]}"
            )

        current_git = git_state(root)
        context["git_commit"] = current_git.get("commit")
        context["git_dirty"] = current_git.get("dirty")
        config_fingerprint = json_hash(config)
        default_run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + str(current_git.get("commit", "nogit"))[:8]
            + "_"
            + config_fingerprint[:8]
        )
        run_id = args.run_id or default_run_id
        output_dir = root / config["output"]["root"] / run_id
        if output_dir.exists():
            raise FileExistsError(f"immutable run directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        context["run_id"] = run_id
        context["output_dir"] = output_dir
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        lgbm_params = dict(config["lightgbm_direct"])
        lgbm_params.pop("strategy", None)
        weight_policy = (config.get("sample_weight") or {}).get("policy")
        weight_floor = float((config.get("sample_weight") or {}).get("floor", 1.0))
        floor_epsilon = float(config["target_definition"]["floor_epsilon"])
        prediction_records: list[dict[str, Any]] = []
        fit_records: list[dict[str, Any]] = []
        importance_records: list[dict[str, Any]] = []

        fold_ids = development["fold_id"].drop_duplicates().tolist()
        print(f"run_id={run_id} folds={len(fold_ids)} features={len(feature_names)} mode={target_mode}")
        for fold_number, fold_id in enumerate(fold_ids, start=1):
            fold = development[development["fold_id"] == fold_id].sort_values("origin")
            train_ends = fold["train_end"].drop_duplicates()
            if len(train_ends) != 1:
                raise AssertionError(f"{fold_id}: expected one training cutoff")
            train_end = pd.Timestamp(train_ends.iloc[0])
            origin_positions = time_to_position.loc[fold["origin"]].to_numpy(dtype=int)
            print(
                f"[{fold_number}/{len(fold_ids)}] {fold_id}: train_end={train_end} "
                f"origins={len(origin_positions)}"
            )

            for target in targets:
                current_column = f"feat_current_{target}"
                if current_column not in frame:
                    raise AssertionError(f"missing causal current target feature: {current_column}")
                horizon_samples: dict[int, dict[str, Any]] = {}
                for horizon in horizons:
                    if time.perf_counter() - started > budget_seconds:
                        raise TimeoutError(
                            f"run exceeded {budget_seconds:.0f}s wall-clock budget"
                        )
                    horizon_delta = horizon * expected_delta
                    actual_series = frame[target].shift(-horizon)
                    actual = actual_series.iloc[origin_positions].to_numpy(dtype=float)
                    if not np.isfinite(actual).all():
                        raise AssertionError(f"{fold_id}/{target}/h{horizon}: missing validation labels")

                    target_times = frame["datetime"].iloc[origin_positions] + horizon_delta
                    baseline_predictions = {
                        "persistence": frame[current_column].iloc[origin_positions].to_numpy(dtype=float),
                        "seasonal_day": frame[target]
                        .shift(period - horizon)
                        .iloc[origin_positions]
                        .to_numpy(dtype=float),
                    }

                    label_available = (frame["datetime"] + horizon_delta <= train_end) & actual_series.notna()
                    train_positions = np.flatnonzero(label_available.to_numpy())
                    if len(train_positions) == 0:
                        raise AssertionError(f"{fold_id}/{target}/h{horizon}: no training rows")
                    y_train = actual_series.iloc[train_positions].to_numpy(dtype=float)
                    current_train = frame[current_column].iloc[train_positions].to_numpy(dtype=float)
                    current_origin = frame[current_column].iloc[origin_positions].to_numpy(dtype=float)
                    if target_mode == "level":
                        fit_positions = train_positions
                        fit_y = y_train
                    else:
                        baseline_mask = np.isfinite(current_train)
                        if not baseline_mask.all():
                            train_positions = train_positions[baseline_mask]
                            y_train = y_train[baseline_mask]
                            current_train = current_train[baseline_mask]
                            if len(train_positions) == 0:
                                raise AssertionError(
                                    f"{fold_id}/{target}/h{horizon}: no training rows with finite "
                                    "persistence baseline"
                                )
                        fit_positions = train_positions
                        if target_mode == "residual_additive":
                            fit_y = y_train - current_train
                        else:
                            fit_y = y_train / np.maximum(current_train, floor_epsilon)
                        if not np.isfinite(fit_y).all():
                            raise AssertionError(
                                f"{fold_id}/{target}/h{horizon}: non-finite transformed training labels"
                            )
                    horizon_samples[horizon] = {
                        "target_times": target_times,
                        "baseline_predictions": baseline_predictions,
                        "fit_positions": fit_positions,
                        "y_train": y_train,
                        "fit_y": fit_y,
                        "current_origin": current_origin,
                        "actual": actual,
                    }

                if strategy == "horizon_grouped":
                    groups = build_horizon_groups(horizons, group_size)
                else:
                    groups = [[horizon] for horizon in horizons]

                for group in groups:
                    fit_started = time.perf_counter()
                    if len(group) == 1:
                        single_horizon = group[0]
                        sample = horizon_samples[single_horizon]
                        model = lgb.LGBMRegressor(**lgbm_params)
                        single_fit_kwargs: dict[str, Any] = {}
                        if weight_policy is not None:
                            single_fit_kwargs["sample_weight"] = 1.0 / np.maximum(
                                np.abs(sample["y_train"]), weight_floor
                            )
                        model.fit(
                            features.iloc[sample["fit_positions"]], sample["fit_y"], **single_fit_kwargs
                        )
                        model_inputs = {single_horizon: features.iloc[origin_positions]}
                    else:
                        x_parts: list[pd.DataFrame] = []
                        y_parts: list[np.ndarray] = []
                        weight_parts: list[np.ndarray] = []
                        for horizon in group:
                            sample = horizon_samples[horizon]
                            part = features.iloc[sample["fit_positions"]].copy()
                            part["feat_horizon_step"] = float(horizon)
                            x_parts.append(part)
                            y_parts.append(sample["fit_y"])
                            if weight_policy is not None:
                                weight_parts.append(
                                    1.0 / np.maximum(np.abs(sample["y_train"]), weight_floor)
                                )
                        grouped_features = pd.concat(x_parts, ignore_index=True)
                        grouped_labels = np.concatenate(y_parts)
                        model = lgb.LGBMRegressor(**lgbm_params)
                        grouped_fit_kwargs: dict[str, Any] = {}
                        if weight_policy is not None:
                            grouped_fit_kwargs["sample_weight"] = np.concatenate(weight_parts)
                        model.fit(grouped_features, grouped_labels, **grouped_fit_kwargs)
                        model_inputs = {}
                        for horizon in group:
                            part = features.iloc[origin_positions].copy()
                            part["feat_horizon_step"] = float(horizon)
                            model_inputs[horizon] = part
                    fit_seconds = time.perf_counter() - fit_started

                    for horizon in group:
                        sample = horizon_samples[horizon]
                        lightgbm_prediction = model.predict(model_inputs[horizon])
                        if target_mode == "residual_additive":
                            lightgbm_prediction = lightgbm_prediction + sample["current_origin"]
                        elif target_mode == "ratio":
                            lightgbm_prediction = lightgbm_prediction * np.maximum(
                                sample["current_origin"], floor_epsilon
                            )
                        if not np.isfinite(lightgbm_prediction).all():
                            raise AssertionError(
                                f"{fold_id}/{target}/h{horizon}: non-finite restored lightgbm predictions"
                            )
                        baseline_predictions = dict(sample["baseline_predictions"])
                        baseline_predictions["lightgbm_direct"] = lightgbm_prediction
                        importance_columns = list(model_inputs[horizon].columns)
                        fit_records.append(
                            {
                                "fold_id": fold_id,
                                "target": target,
                                "horizon_step": horizon,
                                "horizon_minutes": horizon * int(split_config["frequency_minutes"]),
                                "train_end": train_end,
                                "n_train": len(sample["fit_positions"]),
                                "n_features": len(importance_columns),
                                "fit_seconds": fit_seconds,
                            }
                        )
                        context["completed_units"] = len(fit_records)
                        for feature, importance in zip(importance_columns, model.feature_importances_):
                            importance_records.append(
                                {
                                    "fold_id": fold_id,
                                    "target": target,
                                    "horizon_step": horizon,
                                    "feature": feature,
                                    "importance_split": int(importance),
                                }
                            )

                        for model_name, predicted in baseline_predictions.items():
                            if not np.isfinite(predicted).all():
                                raise AssertionError(
                                    f"{fold_id}/{target}/h{horizon}/{model_name}: non-finite predictions"
                                )
                            for position, target_time, actual_value, predicted_value in zip(
                                origin_positions, sample["target_times"], sample["actual"], predicted
                            ):
                                prediction_records.append(
                                    {
                                        "run_id": run_id,
                                        "fold_id": fold_id,
                                        "origin": frame.at[position, "datetime"],
                                        "target_time": target_time,
                                        "target": target,
                                        "horizon_step": horizon,
                                        "horizon_minutes": horizon
                                        * int(split_config["frequency_minutes"]),
                                        "model": model_name,
                                        "actual": actual_value,
                                        "prediction": predicted_value,
                                    }
                                )

        predictions = pd.DataFrame.from_records(prediction_records)
        expected_prediction_rows = len(development) * len(targets) * len(horizons) * len(config["models"])
        if len(predictions) != expected_prediction_rows:
            raise AssertionError(
                f"prediction row count mismatch: expected {expected_prediction_rows}, got {len(predictions)}"
            )
        if predictions.duplicated(["fold_id", "origin", "target", "horizon_step", "model"]).any():
            raise AssertionError("duplicate validation prediction keys")

        metric_records: list[dict[str, Any]] = []
        add_metric_rows(predictions, ["model"], "overall_model", metric_records)
        add_metric_rows(predictions, ["fold_id", "model"], "fold_model", metric_records)
        add_metric_rows(predictions, ["model", "target"], "model_target", metric_records)
        add_metric_rows(
            predictions,
            ["model", "target", "horizon_step", "horizon_minutes"],
            "model_target_horizon",
            metric_records,
        )
        add_metric_rows(
            predictions,
            ["fold_id", "model", "target", "horizon_step", "horizon_minutes"],
            "fold_model_target_horizon",
            metric_records,
        )
        metrics = pd.DataFrame.from_records(metric_records)
        fits = pd.DataFrame.from_records(fit_records)
        importance = pd.DataFrame.from_records(importance_records)

        predictions_path = output_dir / "validation_predictions.csv"
        metrics_path = output_dir / "metrics.csv"
        fits_path = output_dir / "fit_summary.csv"
        importance_path = output_dir / "feature_importance.csv"
        predictions.to_csv(predictions_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8")
        fits.to_csv(fits_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
        importance.to_csv(importance_path, index=False, encoding="utf-8")

        overall = metrics[metrics["scope"] == "overall_model"].sort_values("mape")
        summary_path = output_dir / "summary.json"
        summary = {
            "run_id": run_id,
            "experiment_version": config["experiment_version"],
            "feature_set": config["features"]["feature_set"],
            "target_definition": config["target_definition"],
            "ranking_by_development_mape": overall[
                ["model", "n", "mape", "score_1_mape", "mae", "rmse", "zero_actual_count"]
            ].to_dict(orient="records"),
            "sealed_holdout_evaluated": False,
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8"
        )

        duration = time.perf_counter() - started
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
            "experiment_version": config["experiment_version"],
            "git": current_git,
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "sha256": file_hash(config_path),
                "resolved_sha256": config_fingerprint,
                "split_config_path": split_config_path.relative_to(root).as_posix(),
                "split_config_sha256": file_hash(split_config_path),
            },
            "data": {
                "prepared_train_path": train_path.relative_to(root).as_posix(),
                "prepared_train_sha256": observed_train_hash,
                "preparation_manifest_path": preparation_manifest_path.relative_to(root).as_posix(),
                "preparation_manifest_sha256": file_hash(preparation_manifest_path),
            },
            "split": {
                "assignment_path": split_path.relative_to(root).as_posix(),
                "assignment_sha256": split_fingerprint,
                "development_fold_ids": fold_ids,
                "development_origin_rows": len(development),
                "sealed_holdout_evaluated": False,
            },
            "model": {
                "names": list(config["models"]),
                "feature_set": config["features"]["feature_set"],
                "target_definition": config["target_definition"],
                "sample_weight_policy": weight_policy,
                "sample_weight_floor": weight_floor if weight_policy else None,
                "lightgbm_strategy": config["lightgbm_direct"]["strategy"],
                "lightgbm_params": lgbm_params,
                "feature_count": len(feature_names),
                "features": feature_names,
            },
            "metric_policy": config["metrics"],
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    package: importlib.metadata.version(package)
                    for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "pyyaml"]
                },
            },
            "outputs": {
                path.name: file_hash(path)
                for path in [
                    predictions_path,
                    metrics_path,
                    fits_path,
                    importance_path,
                    summary_path,
                    output_dir / "resolved_config.json",
                ]
            },
        }
        manifest_path = output_dir / "run_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8"
        )

        registry_record = {
            "run_id": run_id,
            "created_at_utc": manifest["created_at_utc"],
            "git_commit": current_git.get("commit"),
            "git_dirty": current_git.get("dirty"),
            "config_sha256": manifest["config"]["sha256"],
            "prepared_train_sha256": observed_train_hash,
            "split_sha256": split_fingerprint,
            "best_development_model": overall.iloc[0]["model"],
            "best_development_mape": overall.iloc[0]["mape"],
            "best_development_score_1_mape": overall.iloc[0]["score_1_mape"],
            "duration_seconds": duration,
            "artifact_directory": output_dir.relative_to(root).as_posix(),
            "sealed_holdout_evaluated": False,
        }
        append_registry_record(root, registry_record)
        print(overall[["model", "mape", "score_1_mape", "mae", "rmse"]].to_string(index=False))
        print(f"PASS output={output_dir} duration_seconds={duration:.2f} holdout_evaluated=false")
        return 0
    except Exception as exc:  # noqa: BLE001
        return record_failure(root, config_path, context, started, exc)


if __name__ == "__main__":
    raise SystemExit(main())