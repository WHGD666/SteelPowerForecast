"""Build the semifinal submission on true 1-minute rolling origins.

Follow-up to the 14400-origin broadcast probe (min1_broadcast_v3): instead of
broadcasting the 960-origin 15-minute-grid predictions, this retrains the
frozen LightGBM direct multi-step design natively on the 1-minute timeline:

- origins: every minute of 2025-10-01 00:00 .. 2025-10-10 23:59 (14400);
- horizons: t+15 .. t+1440 minutes, 96 steps per target;
- features: covariate-only semifinal block with windows scaled to the 1-minute
  grid (rolling means 60/240/1440 minutes = the frozen 4/16/96 x 15min;
  holder deltas 15/60/240 minutes; generator gas use diffs 15/60 minutes),
  computed on the concatenated train+test timeline so windows continue across
  the boundary; every feature uses observations with timestamp <= t only;
- training: per (target x horizon) LightGBM with the frozen parameters on the
  full 1-minute training timeline (2025-05-03 .. 2025-09-30).

Causality note: the semifinal test tables contain no target observations, so
no target-derived features are used (same as semi_covprobe_v1).
"""

from __future__ import annotations

import argparse
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
from build_submission_semi import (  # noqa: E402
    BALANCE_COMPONENTS,
    GENERATOR_GAS_USE_COLUMNS,
    column_sum,
    lightgbm_params,
    raw_covariate_columns,
)


def make_feature_frame_semi_min1(frame: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Covariate-only semifinal features with windows scaled to 1-minute rows."""
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
        for window in (60, 240, 1440):
            features[f"{name}_roll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
        features[name] = series

    holder = frame["blast_furnace_gas_holder_2"]
    for lag in (15, 60, 240):
        features[f"feat_holder_delta_{lag}"] = holder.diff(lag)
    features["feat_holder_delta_roll_mean_15"] = holder.diff(1).rolling(15, min_periods=1).mean()

    for column in GENERATOR_GAS_USE_COLUMNS:
        for lag in (15, 60):
            features[f"feat_{column}_diff_{lag}"] = frame[column].diff(lag)

    non_numeric = [name for name in features if not pd.api.types.is_numeric_dtype(features[name])]
    if non_numeric:
        raise AssertionError(f"non-numeric model features: {non_numeric}")
    return features, list(features.columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/submission_semi_composed.yaml"))
    parser.add_argument("--submission-id", default="20260926T_min1_native_v4")
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    config = yaml.safe_load((root / args.config).resolve().read_text(encoding="utf-8"))
    targets = list(config["targets"])
    minutes = [int(m) for m in config["horizons_minutes"]]
    short_minutes = [m for m in minutes if m <= 120]

    prepared_dir = root / "outputs/prepared_v2_min1"
    train = pd.read_csv(prepared_dir / "train_prepared.csv", parse_dates=["datetime"])
    test = pd.read_csv(prepared_dir / "test_prepared.csv", parse_dates=["datetime"])
    if train["datetime"].max() >= test["datetime"].min():
        raise AssertionError("train/test timelines overlap")
    if test["datetime"].diff().dropna().ne(pd.Timedelta(minutes=1)).any():
        raise AssertionError("test origins are not a contiguous 1-minute grid")
    if len(test) != 14400:
        raise AssertionError(f"expected 14400 test origins, got {len(test)}")

    combined = pd.concat([train, test], ignore_index=True)
    features_all, columns = make_feature_frame_semi_min1(combined, targets)
    train_features = features_all.iloc[: len(train)].reset_index(drop=True)
    test_features = features_all.iloc[len(train):].reset_index(drop=True)
    if len(test_features) != len(test):
        raise AssertionError("feature split mismatch")

    params = lightgbm_params(config)
    prediction_matrix: dict[int, dict[str, np.ndarray]] = {h: {} for h in minutes}
    model_summaries: list[dict[str, Any]] = []
    for target in targets:
        series = train[target]
        for h in minutes:
            labels = series.shift(-h)
            mask = labels.notna()
            model = lgb.LGBMRegressor(**params)
            model.fit(train_features.loc[mask, columns], labels[mask])
            pred = np.clip(model.predict(test_features[columns]), 0.0, None)
            prediction_matrix[h][target] = pred
            model_summaries.append({"target": target, "horizon_minutes": h, "n_train": int(mask.sum())})
            print(f"fit {target} t+{h}min done", flush=True)

    def build_frame(horizon_list: list[int]) -> pd.DataFrame:
        frame = pd.DataFrame({"datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")})
        for target in targets:
            for h in horizon_list:
                frame[f"{target}_t+{h}_pred"] = np.round(prediction_matrix[h][target], 6)
        return frame

    long_out = build_frame(minutes)
    short_out = build_frame(short_minutes)

    submission_input = pd.read_csv(prepared_dir / "input.csv")
    expected = test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    validation_long = validate_submission(submission_input, long_out, expected, targets, minutes)
    validation_short = validate_submission(submission_input, short_out, expected, targets, short_minutes)

    out_dir = root / config["output"]["root"] / args.submission_id
    out_dir.mkdir(parents=True, exist_ok=False)
    short_path = out_dir / "s_result.csv"
    long_path = out_dir / "l_result.csv"
    input_path = out_dir / "input.csv"
    short_out.to_csv(short_path, index=False, encoding="utf-8")
    long_out.to_csv(long_path, index=False, encoding="utf-8")
    submission_input.to_csv(input_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")

    archive_path = out_dir / f"{config['team_name']}_{config['output']['archive_suffix']}"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(short_path, arcname="s_result.csv")
        archive.write(long_path, arcname="l_result.csv")

    manifest = {
        "submission_id": args.submission_id,
        "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "competition_stage": "semi",
        "team_name": config["team_name"],
        "selection_policy": "lightgbm_direct_covariate_semi_native_min1_origins",
        "origin_grid": "1-minute rolling, 14400 origins",
        "feature_set": "covariate_physics_calendar_min1_windows_no_target_features",
        "feature_column_count": len(columns),
        "models": model_summaries,
        "validation_long": validation_long,
        "validation_short": validation_short,
        "package_contents": ["s_result.csv (short t+15..t+120)", "l_result.csv (long t+15..t+1440)"],
        "artifacts": {
            archive_path.relative_to(root).as_posix(): file_hash(archive_path),
            short_path.relative_to(root).as_posix(): file_hash(short_path),
            long_path.relative_to(root).as_posix(): file_hash(long_path),
            input_path.relative_to(root).as_posix(): file_hash(input_path),
        },
        "package_size_bytes": archive_path.stat().st_size,
        "sealed_holdout_evaluated": False,
        "git": git_state(root),
    }
    (out_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "out": out_dir.as_posix(), "size": archive_path.stat().st_size}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
