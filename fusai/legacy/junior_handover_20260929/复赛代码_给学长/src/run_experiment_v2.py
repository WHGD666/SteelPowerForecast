"""IronFlow experiment v2: causal physics feature block for LightGBM direct multi-step.

One controlled change versus baseline_v1: the feature matrix is extended with
gas-balance, gas-holder dynamics and target trend/rolling features.  Splits,
fold structure, LightGBM parameters and metrics stay identical to the frozen
baseline_v1 contract, so the comparison isolates the feature factor.

Causality: every engineered feature at row t uses observations with timestamp
<= t only (diffs, rolling windows ending at t, and t-row columns).  No future
information enters training or validation features.
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
    make_feature_frame,
    validate_split_fingerprint,
)

BALANCE_COMPONENTS = {
    "blast_furnace_production": ["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"],
    "air_heater_consumption": ["air_heater_1", "air_heater_2", "air_heater_4"],
    "blast_furnace_user_consumption": [
        "blast_furnace_user1",
        "blast_furnace_user2",
        "blast_furnace_user3",
        "blast_furnace_user4",
    ],
}

GENERATOR_GAS_USE_COLUMNS = [
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
]


def column_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AssertionError(f"balance component missing from prepared table: {missing}")
    return frame[columns].sum(axis=1)


def make_feature_frame_v2(
    frame: pd.DataFrame,
    prepared_feature_columns: list[str],
    targets: list[str],
    target_lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    features, _ = make_feature_frame(frame, prepared_feature_columns, targets, target_lags)

    blast_furnace_production = column_sum(frame, BALANCE_COMPONENTS["blast_furnace_production"])
    blast_furnace_sink = (
        column_sum(frame, BALANCE_COMPONENTS["air_heater_consumption"])
        + column_sum(frame, BALANCE_COMPONENTS["blast_furnace_user_consumption"])
        + frame["into_gas_mixed_blast_furnace"]
    )
    coke_balance = frame["coke_oven_1"] - frame["into_gas_mixed_coke"]
    converter_balance = frame["converter_1"] - (frame["converter_user2"] + frame["into_gas_mixed_converter"])
    blast_furnace_balance = blast_furnace_production - blast_furnace_sink
    total_balance = blast_furnace_balance + coke_balance + converter_balance

    balance_series = {
        "feat_blast_furnace_balance": blast_furnace_balance,
        "feat_coke_balance": coke_balance,
        "feat_converter_balance": converter_balance,
        "feat_total_gas_balance": total_balance,
    }
    for name, series in balance_series.items():
        for window in (4, 16, 96):
            features[f"{name}_roll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
        features[name] = series

    holder = frame["blast_furnace_gas_holder_2"]
    for lag in (1, 4, 16):
        features[f"feat_holder_delta_{lag}"] = holder.diff(lag)
    features["feat_holder_delta_roll_mean_4"] = holder.diff(1).rolling(4, min_periods=1).mean()

    for target in targets:
        series = frame[target]
        for lag in (1, 4):
            features[f"feat_{target}_diff_{lag}"] = series.diff(lag)
        for window in (4, 16, 96):
            features[f"feat_{target}_roll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
            features[f"feat_{target}_roll_std_{window}"] = series.rolling(window, min_periods=1).std()

    for column in GENERATOR_GAS_USE_COLUMNS:
        for lag in (1, 4):
            features[f"feat_{column}_diff_{lag}"] = frame[column].diff(lag)

    non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(features[name])]
    if non_numeric:
        raise AssertionError(f"non-numeric model features: {non_numeric}")
    return features, list(features.columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_v2_features.yaml"))
    parser.add_argument("--run-id", help="Optional immutable run id; must not already exist")
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    split_config_path = root / config["split_config"]
    split_config = yaml.safe_load(split_config_path.read_text(encoding="utf-8"))
    split_path = root / config["split_assignment_file"]
    split_fingerprint_path = root / split_config["output"]["fingerprint_file"]
    split_fingerprint = validate_split_fingerprint(split_path, split_fingerprint_path)
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

    frame = pd.read_csv(train_path, parse_dates=["datetime"], low_memory=False)
    if frame["datetime"].duplicated().any() or not frame["datetime"].is_monotonic_increasing:
        raise AssertionError("prepared training timestamps must be unique and increasing")
    expected_delta = pd.Timedelta(minutes=int(split_config["frequency_minutes"]))
    if not frame["datetime"].diff().dropna().eq(expected_delta).all():
        raise AssertionError("prepared training table is not on the frozen 15-minute grid")

    targets = list(config["targets"])
    horizons = [int(value) for value in config["horizons_steps"]]
    period = int(config["seasonal_day"]["period_steps"])
    target_lags = [int(value) for value in config["features"]["target_lags_steps"]]
    prepared_features = list(preparation_manifest["input_feature_columns"])
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
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lgbm_params = dict(config["lightgbm_direct"])
    lgbm_params.pop("strategy", None)
    weight_policy = (config.get("sample_weight") or {}).get("policy")
    weight_floor = float((config.get("sample_weight") or {}).get("floor", 1.0))
    if weight_policy not in (None, "inverse_actual_clipped"):
        raise AssertionError(f"unsupported sample weight policy: {weight_policy}")
    target_transform = (config.get("training") or {}).get("target_transform")
    if target_transform not in (None, "log", "delta"):
        raise AssertionError(f"unsupported target transform: {target_transform}")
    recency_half_life_days = (config.get("training") or {}).get("recency_half_life_days")
    if recency_half_life_days is not None:
        recency_half_life_days = float(recency_half_life_days)
        if recency_half_life_days <= 0:
            raise AssertionError("recency_half_life_days must be positive")
    prediction_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []

    fold_ids = development["fold_id"].drop_duplicates().tolist()
    print(f"run_id={run_id} folds={len(fold_ids)} features={len(feature_names)}")
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
            for horizon in horizons:
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
                model = lgb.LGBMRegressor(**lgbm_params)
                fit_started = time.perf_counter()
                fit_kwargs: dict[str, Any] = {}
                y_train = actual_series.iloc[train_positions].to_numpy(dtype=float)
                current_column = f"feat_current_{target}"
                current_train = frame[current_column].iloc[train_positions].to_numpy(dtype=float)
                current_origin = frame[current_column].iloc[origin_positions].to_numpy(dtype=float)
                if target_transform == "log":
                    fit_y = np.log(np.maximum(y_train, 1e-6))
                elif target_transform == "delta":
                    fit_y = y_train - current_train
                else:
                    fit_y = y_train
                if weight_policy is not None:
                    fit_kwargs["sample_weight"] = 1.0 / np.maximum(np.abs(y_train), weight_floor)
                if recency_half_life_days is not None:
                    ages = (
                        train_end - frame["datetime"].iloc[train_positions]
                    ).dt.total_seconds().to_numpy() / 86400.0
                    recency = 0.5 ** (ages / recency_half_life_days)
                    fit_kwargs["sample_weight"] = (
                        recency
                        if "sample_weight" not in fit_kwargs
                        else fit_kwargs["sample_weight"] * recency
                    )
                model.fit(features.iloc[train_positions], fit_y, **fit_kwargs)
                lightgbm_prediction = model.predict(features.iloc[origin_positions])
                if target_transform == "log":
                    lightgbm_prediction = np.exp(lightgbm_prediction)
                elif target_transform == "delta":
                    lightgbm_prediction = current_origin + lightgbm_prediction
                fit_seconds = time.perf_counter() - fit_started
                baseline_predictions["lightgbm_direct"] = lightgbm_prediction
                fit_records.append(
                    {
                        "fold_id": fold_id,
                        "target": target,
                        "horizon_step": horizon,
                        "horizon_minutes": horizon * int(split_config["frequency_minutes"]),
                        "train_end": train_end,
                        "n_train": len(train_positions),
                        "n_features": len(feature_names),
                        "fit_seconds": fit_seconds,
                    }
                )
                for feature, importance in zip(feature_names, model.feature_importances_):
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
                        origin_positions, target_times, actual, predicted
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
            "sample_weight_policy": weight_policy,
            "sample_weight_floor": weight_floor if weight_policy else None,
            "target_transform": target_transform,
            "recency_half_life_days": recency_half_life_days,
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

    registry_path = root / "experiments" / "registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_record = pd.DataFrame.from_records(
        [
            {
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
        ]
    )
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        if run_id in set(registry["run_id"]):
            raise AssertionError(f"run id already exists in experiment registry: {run_id}")
        registry = pd.concat([registry, registry_record], ignore_index=True)
    else:
        registry = registry_record
    registry.to_csv(registry_path, index=False, encoding="utf-8", lineterminator="\n")
    print(overall[["model", "mape", "score_1_mape", "mae", "rmse"]].to_string(index=False))
    print(f"PASS output={output_dir} duration_seconds={duration:.2f} holdout_evaluated=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
