"""Build the IronFlow semifinal (round-2 long-horizon) submission.

Semifinal specifics versus the preliminary builders:

- 96 horizons (t+15 ... t+1440 minutes) per target, 960 rolling origins on a
  15-minute grid over 2025-10-01 .. 2025-10-10.
- The semifinal test tables provide NO target observations at all
  (generator_1 / generator_all are entirely NaN), so every feature must be
  computable from covariates alone.  Target lags, target rolling stats and
  feat_current_* persistence baselines are therefore excluded from the model
  feature set; the covariate block (gas balances, holder dynamics, generator
  gas use, calendar) of the frozen v2 feature set is kept with identical
  formulas.
- Features are computed once on the concatenated train+test timeline so that
  rolling windows at the start of the test period continue from training
  history.  Every feature at row t uses observations with timestamp <= t only.

Modes:
  --mode eval   internal development estimate on a splits_v2 development fold
                (default dev_03), never an official score
  --mode build  full-training-window refit and submission package construction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import file_hash, git_state, validate_submission  # noqa: E402

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


def raw_covariate_columns(frame: pd.DataFrame, targets: list[str]) -> list[str]:
    return sorted(
        column
        for column in frame.columns
        if column != "datetime" and not column.startswith("feat_") and column not in targets
    )


def make_feature_frame_semi(frame: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Covariate-only semifinal feature block (no target-derived features)."""
    covariates = raw_covariate_columns(frame, targets)
    features = frame[covariates].copy()

    timestamp = frame["datetime"]
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    day_angle = 2.0 * np.pi * minute_of_day / (24.0 * 60.0)
    week_angle = 2.0 * np.pi * timestamp.dt.dayofweek / 7.0
    features["feat_hour_sin"] = np.sin(day_angle)
    features["feat_hour_cos"] = np.cos(day_angle)
    features["feat_day_of_week_sin"] = np.sin(week_angle)
    features["feat_day_of_week_cos"] = np.cos(week_angle)

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

    for column in GENERATOR_GAS_USE_COLUMNS:
        for lag in (1, 4):
            features[f"feat_{column}_diff_{lag}"] = frame[column].diff(lag)

    non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(features[name])]
    if non_numeric:
        raise AssertionError(f"non-numeric model features: {non_numeric}")
    return features, list(features.columns)


def pooled_mape(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int]:
    mask = actual != 0
    if not mask.any():
        raise AssertionError("no non-zero actual values for MAPE")
    mape = float(np.abs((actual[mask] - predicted[mask]) / actual[mask]).mean())
    return mape, int((~mask).sum())


def lightgbm_params(config: dict[str, Any]) -> dict[str, Any]:
    block = config["lightgbm_direct"]
    return {
        "objective": block["objective"],
        "n_estimators": block["n_estimators"],
        "learning_rate": block["learning_rate"],
        "num_leaves": block["num_leaves"],
        "max_depth": block["max_depth"],
        "min_child_samples": block["min_child_samples"],
        "subsample": block["subsample"],
        "subsample_freq": block["subsample_freq"],
        "colsample_bytree": block["colsample_bytree"],
        "reg_alpha": block["reg_alpha"],
        "reg_lambda": block["reg_lambda"],
        "random_state": block["random_state"],
        "n_jobs": block["n_jobs"],
        "verbosity": block["verbosity"],
        "deterministic": block["deterministic"],
        "force_col_wise": block["force_col_wise"],
    }


def load_prepared(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_dir = root / config["prepared_directory"]
    train_name = config.get("prepared_train_file", "train_prepared.csv")
    train = pd.read_csv(prepared_dir / train_name, parse_dates=["datetime"])
    test = pd.read_csv(prepared_dir / config["prepared_test_file"], parse_dates=["datetime"])
    if list(train.columns) != list(test.columns):
        raise AssertionError("prepared train/test schemas differ")
    return train, test


def combined_features(
    train: pd.DataFrame, test: pd.DataFrame, targets: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if train["datetime"].max() >= test["datetime"].min():
        raise AssertionError("train/test timelines overlap or are not ordered")
    gap = test["datetime"].min() - train["datetime"].max()
    if gap != pd.Timedelta(minutes=15):
        raise AssertionError(f"train/test timelines not contiguous: gap={gap}")
    combined = pd.concat([train, test], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    features, columns = make_feature_frame_semi(combined, targets)
    features["datetime"] = combined["datetime"]
    train_features = features[features["datetime"] <= train["datetime"].max()].reset_index(drop=True)
    test_features = features[features["datetime"] >= test["datetime"].min()].reset_index(drop=True)
    if len(train_features) != len(train) or len(test_features) != len(test):
        raise AssertionError("feature split does not match prepared partitions")
    return train_features, test_features, columns


def horizons_from_config(config: dict[str, Any]) -> tuple[list[int], list[int]]:
    steps = [int(step) for step in config["horizons_steps"]]
    minutes = [int(minute) for minute in config["horizons_minutes"]]
    if len(steps) != len(minutes) or minutes != [step * int(config["frequency_minutes"]) for step in steps]:
        raise AssertionError("horizon steps and minutes are inconsistent")
    return steps, minutes


def eval_feature_block(
    train: pd.DataFrame, targets: list[str], config: dict[str, Any], variant: str
) -> tuple[pd.DataFrame, list[str]]:
    if variant == "with_target_lags":
        from run_experiment_v2 import make_feature_frame_v2  # noqa: E402

        covariates = raw_covariate_columns(train, targets)
        lags = [int(step) for step in config["features"]["target_lags_steps"]]
        return make_feature_frame_v2(train, covariates, targets, lags)
    if variant == "covariate_only":
        return make_feature_frame_semi(train, targets)
    raise AssertionError(f"unknown eval variant: {variant}")


def run_eval(
    root: Path,
    config: dict[str, Any],
    fold_id: str,
    variant: str,
) -> dict[str, Any]:
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)
    train, _ = load_prepared(root, config)

    splits = pd.read_csv(root / "manifests/splits_v2.csv", parse_dates=["train_end", "origin"])
    fold = splits[(splits["fold_id"] == fold_id) & (splits["role"] == "development")]
    if fold.empty:
        raise AssertionError(f"development fold not found in splits_v2: {fold_id}")
    train_end = fold["train_end"].iloc[0]
    origins = fold["origin"].reset_index(drop=True)

    origin_mask = train["datetime"].isin(set(origins))
    origin_rows = train.loc[origin_mask].reset_index(drop=True)
    if len(origin_rows) != len(origins) or origin_rows["datetime"].tolist() != origins.tolist():
        raise AssertionError("fold origins missing or misordered in prepared train table")

    features, columns = eval_feature_block(train, targets, config, variant)
    features["datetime"] = train["datetime"].values
    train_features = features[train["datetime"] <= train_end].reset_index(drop=True)
    origin_features = features.loc[origin_mask].reset_index(drop=True)
    if origin_features["datetime"].tolist() != origins.tolist():
        raise AssertionError("origin feature rows misaligned")

    params = lightgbm_params(config)
    metrics: list[dict[str, Any]] = []
    predictions_long: list[pd.DataFrame] = []
    for target in targets:
        target_series = train[target]
        persistence_at_origin = origin_rows[target].to_numpy(dtype=float)
        for step in steps:
            label_all = target_series.shift(-step)
            train_labels = label_all[train["datetime"] <= train_end]
            mask = train_labels.notna()
            model = lgb.LGBMRegressor(**params)
            model.fit(train_features.loc[mask, columns], train_labels[mask])
            pred = np.clip(model.predict(origin_features[columns]), 0.0, None)
            actual = label_all.loc[origin_mask].to_numpy(dtype=float)
            valid = np.isfinite(actual)
            model_mape, zero_count = pooled_mape(actual[valid], pred[valid])
            persistence_mape, _ = pooled_mape(actual[valid], persistence_at_origin[valid])
            predictions_long.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "step": step,
                        "origin": origins[valid],
                        "actual": actual[valid],
                        "pred": pred,
                        "persistence": persistence_at_origin[valid],
                    }
                )
            )
            metrics.append(
                {
                    "target": target,
                    "step": step,
                    "model": f"lightgbm_{variant}_semi",
                    "mape": model_mape,
                    "persistence_mape": persistence_mape,
                    "n": int(valid.sum()),
                    "zero_actual_count": zero_count,
                }
            )
            print(f"eval {target} step {step:2d}: model {model_mape:.6f} | persistence {persistence_mape:.6f}", flush=True)

    long_frame = pd.concat(predictions_long, ignore_index=True)
    summary: dict[str, Any] = {}
    for target in targets:
        block = long_frame[long_frame["target"] == target]
        model_mape, _ = pooled_mape(block["actual"].to_numpy(), block["pred"].to_numpy())
        persistence_mape, _ = pooled_mape(block["actual"].to_numpy(), block["persistence"].to_numpy())
        summary[target] = {
            "model_mape": model_mape,
            "model_score_1_mape": 1.0 - model_mape,
            "persistence_mape": persistence_mape,
        }
    pooled_all, _ = pooled_mape(long_frame["actual"].to_numpy(), long_frame["pred"].to_numpy())
    pooled_persistence, _ = pooled_mape(long_frame["actual"].to_numpy(), long_frame["persistence"].to_numpy())
    return {
        "fold_id": fold_id,
        "train_end": train_end.isoformat(),
        "origins": int(len(origins)),
        "variant": variant,
        "per_target": summary,
        "pooled": {"model_mape": pooled_all, "persistence_mape": pooled_persistence},
        "metrics": metrics,
    }


def run_build(root: Path, config: dict[str, Any], submission_id: str | None) -> dict[str, Any]:
    started = time.perf_counter()
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)
    train, test = load_prepared(root, config)
    train_features, test_features, columns = combined_features(train, test, targets)

    params = lightgbm_params(config)
    prediction_matrix: dict[tuple[str, int], np.ndarray] = {}
    model_summaries: list[dict[str, Any]] = []
    for target in targets:
        target_series = train[target]
        for step in steps:
            labels = target_series.shift(-step)
            mask = labels.notna()
            model = lgb.LGBMRegressor(**params)
            model.fit(train_features.loc[mask, columns], labels[mask])
            pred = np.clip(model.predict(test_features[columns]), 0.0, None)
            prediction_matrix[(target, step)] = pred
            model_summaries.append(
                {
                    "target": target,
                    "horizon_step": step,
                    "horizon_minutes": step * int(config["frequency_minutes"]),
                    "n_train": int(mask.sum()),
                    "last_label_time": train["datetime"].max().isoformat(),
                }
            )
            print(f"fit {target} step {step:2d} done", flush=True)

    predictions = pd.DataFrame({"datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")})
    for target in targets:
        for step, minute in zip(steps, minutes):
            predictions[f"{target}_t+{minute}_pred"] = np.round(prediction_matrix[(target, step)], 6)

    # Official semifinal rule (group Q&A 2026-09): no input.csv is submitted; the
    # package carries two prefixed result files - short-period s_result.csv
    # (t+15..t+120, same contract as the preliminary) and long-period
    # l_result.csv (t+15..t+1440).  Short horizons are the first steps of the
    # same per-step models, so no refit is needed.
    short_steps = [step for step in steps if step * int(config["frequency_minutes"]) <= 120]
    short_minutes = [step * int(config["frequency_minutes"]) for step in short_steps]
    short_predictions = pd.DataFrame({"datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")})
    for target in targets:
        for step, minute in zip(short_steps, short_minutes):
            short_predictions[f"{target}_t+{minute}_pred"] = np.round(prediction_matrix[(target, step)], 6)

    prepared_input_path = root / config["prepared_directory"] / config["prepared_input_file"]
    submission_input = pd.read_csv(prepared_input_path)
    expected_datetimes = test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    validation_long = validate_submission(submission_input, predictions, expected_datetimes, targets, minutes)
    validation_short = validate_submission(submission_input, short_predictions, expected_datetimes, targets, short_minutes)

    output_root = root / config["output"]["root"]
    run_id = submission_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_semi_covprobe_v1"
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    input_path = out_dir / config["output"]["input_file"]
    result_path = out_dir / "l_result.csv"
    short_result_path = out_dir / "s_result.csv"
    submission_input.to_csv(input_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
    predictions.to_csv(result_path, index=False, encoding="utf-8")
    short_predictions.to_csv(short_result_path, index=False, encoding="utf-8")

    archive_path = out_dir / f"{config['team_name']}_{config['output']['archive_suffix']}"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(short_result_path, arcname="s_result.csv")
        archive.write(result_path, arcname="l_result.csv")

    manifest = {
        "submission_id": run_id,
        "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "competition_stage": "semi",
        "team_name": config["team_name"],
        "selection_policy": "covariate_only_lightgbm_direct_no_target_history_available_in_test",
        "per_target_model": {target: "lightgbm_direct_covariate_semi" for target in targets},
        "feature_set": "covariate_physics_calendar_no_target_features",
        "feature_columns": columns,
        "feature_column_count": len(columns),
        "target_feature_exclusion_reason": "semifinal test targets are entirely absent; target lags/rolls would be all-NaN at predict time",
        "models": model_summaries,
        "validation_long": validation_long,
        "validation_short": validation_short,
        "package_contents": ["s_result.csv (short t+15..t+120)", "l_result.csv (long t+15..t+1440)"],
        "package_format_note": "official group Q&A 2026-09: semifinal submission carries only the two prefixed result files, no input.csv; scoring uses 15-minute interval means",
        "input_rows": int(len(submission_input)),
        "input_columns": int(len(submission_input.columns)),
        "prediction_rows": int(len(predictions)),
        "prediction_columns": int(len(predictions.columns)),
        "short_prediction_columns": int(len(short_predictions.columns)),
        "artifacts": {
            archive_path.relative_to(root).as_posix(): file_hash(archive_path),
            short_result_path.relative_to(root).as_posix(): file_hash(short_result_path),
            result_path.relative_to(root).as_posix(): file_hash(result_path),
            input_path.relative_to(root).as_posix(): file_hash(input_path),
        },
        "package_size_bytes": archive_path.stat().st_size,
        "source_is_internal_validation_not_official_score": True,
        "sealed_holdout_evaluated": False,
        "git": git_state(root),
    }
    manifest_path = out_dir / "submission_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": out_dir.as_posix(), "zip": archive_path.name, "size": archive_path.stat().st_size}, ensure_ascii=False))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/submission_semi_composed.yaml"))
    parser.add_argument("--mode", choices=("eval", "build"), default="build")
    parser.add_argument("--eval-fold", default="dev_03")
    parser.add_argument(
        "--variant",
        choices=("covariate_only", "with_target_lags"),
        default="covariate_only",
        help="eval only: with_target_lags is a diagnostic ceiling (target history is absent in the semifinal test period and cannot be used for submission)",
    )
    parser.add_argument("--submission-id", help="Optional immutable output directory name")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if args.mode == "eval":
        result = run_eval(root, config, args.eval_fold, args.variant)
        print(json.dumps({key: result[key] for key in ("fold_id", "per_target", "pooled")}, ensure_ascii=False, indent=2))
    else:
        run_build(root, config, args.submission_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
