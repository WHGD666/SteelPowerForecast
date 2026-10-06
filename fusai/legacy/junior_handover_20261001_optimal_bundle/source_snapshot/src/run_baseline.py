"""Run leakage-safe IronFlow development baselines on frozen rolling-origin folds.

The sealed holdout is deliberately rejected by this runner.  One LightGBM model
is fitted for every target/horizon pair and fold, using labels no later than the
fold training cutoff.  Validation features are evaluated at their rolling
origins and therefore never use observations after the prediction origin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Calculate diagnostics; zero actuals are excluded from MAPE and counted."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(actual) & np.isfinite(predicted)
    if not finite.any():
        raise ValueError("metric input contains no finite actual/prediction pairs")
    clean_actual = actual[finite]
    clean_predicted = predicted[finite]
    error = clean_predicted - clean_actual
    nonzero = clean_actual != 0.0
    zero_count = int((~nonzero).sum())
    mape = float(np.mean(np.abs(error[nonzero]) / np.abs(clean_actual[nonzero]))) if nonzero.any() else math.nan
    return {
        "n": int(finite.sum()),
        "mape_n": int(nonzero.sum()),
        "zero_actual_count": zero_count,
        "mape": mape,
        "score_1_mape": 1.0 - mape if math.isfinite(mape) else math.nan,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, encoding="utf-8", errors="replace"
        ).strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}


def make_feature_frame(
    frame: pd.DataFrame,
    prepared_feature_columns: list[str],
    targets: list[str],
    target_lags: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    missing = sorted(set(prepared_feature_columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"prepared input features missing from training table: {missing}")
    forbidden = sorted(set(prepared_feature_columns) & set(targets))
    if forbidden:
        raise AssertionError(f"raw future targets cannot be model features: {forbidden}")

    features = frame[prepared_feature_columns].copy()
    for target in targets:
        for lag in target_lags:
            features[f"feat_{target}_lag_{lag}"] = frame[target].shift(lag)

    timestamp = frame["datetime"]
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    day_angle = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    week_angle = 2.0 * np.pi * timestamp.dt.dayofweek / 7.0
    features["feat_hour_sin"] = np.sin(day_angle)
    features["feat_hour_cos"] = np.cos(day_angle)
    features["feat_day_of_week_sin"] = np.sin(week_angle)
    features["feat_day_of_week_cos"] = np.cos(week_angle)

    non_numeric = [column for column in features if not pd.api.types.is_numeric_dtype(features[column])]
    if non_numeric:
        raise AssertionError(f"non-numeric model features: {non_numeric}")
    return features, list(features.columns)


def add_metric_rows(
    predictions: pd.DataFrame,
    group_columns: list[str],
    scope: str,
    destination: list[dict[str, Any]],
) -> None:
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        grouped = predictions.groupby(group_columns, sort=True, dropna=False)
    else:
        grouped = [((), predictions)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record: dict[str, Any] = {"scope": scope}
        record.update(dict(zip(group_columns, keys)))
        record.update(calculate_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()))
        destination.append(record)


def validate_split_fingerprint(split_path: Path, fingerprint_path: Path) -> str:
    if not split_path.exists() or not fingerprint_path.exists():
        raise FileNotFoundError(
            "frozen split artifacts are missing; run src/build_splits.py before the baseline"
        )
    expected = fingerprint_path.read_text(encoding="ascii").split()[0]
    observed = file_hash(split_path)
    if observed != expected:
        raise AssertionError(f"split fingerprint mismatch: expected {expected}, observed {observed}")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v1.yaml"))
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
    if (assignments["role"] == "holdout").sum() == 0:
        raise AssertionError("split assignments must contain a sealed holdout")
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
    features, feature_names = make_feature_frame(frame, prepared_features, targets, target_lags)
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
                model.fit(features.iloc[train_positions], actual_series.iloc[train_positions])
                lightgbm_prediction = model.predict(features.iloc[origin_positions])
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
