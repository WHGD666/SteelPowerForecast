"""Recursive semifinal model (v8): one rolling prediction stream per target.

Motivation: the independent per-step models produce a per-step accuracy curve
that oscillates (47/95 and 40/95 monotonicity violations on the sealed
holdout), which risks the official statistical review ("the closer to the
origin, the more accurate").  A recursive stream - where horizon h's features
include the model's own earlier predictions - has accuracy that decays
naturally with horizon by construction, and gives far horizons access to
recent predicted levels instead of only origin-time covariates.

Mechanics (15-minute grid, fully causal):
- Origins are processed in chronological order.  A working series W holds
  actual targets up to the last training timestamp and the model's own
  predictions afterwards: W[s] for a test timestamp s is the prediction made
  at the previous origin (s - 15min, step 1); the first test timestamp is
  forward-filled from the last training actual (causal fill of our own best
  estimate).
- For each origin t and each target, features are the frozen v2 physics block
  computed on the combined covariate timeline (origin-time covariates) plus
  target lags / diffs / rolling stats computed from W with timestamps <= t.
  Training uses exactly the same feature definition with W = actuals, so
  train and test-time feature semantics match.
- Models: the frozen LightGBM direct parameters, one per (target, horizon).

An honest validation is provided by simulating the recursion on a development
fold: W is cut off at the fold's train_end and refilled by the model's own
predictions, exactly as in the test period.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission_semi import (  # noqa: E402
    BALANCE_COMPONENTS,
    GENERATOR_GAS_USE_COLUMNS,
    column_sum,
    horizons_from_config,
    lightgbm_params,
    raw_covariate_columns,
)

TARGET_LAG_STEPS = [1, 4, 8, 16, 32, 96, 192, 672]


def physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[raw_covariate_columns(frame, ["generator_1", "generator_all"])].copy()
    timestamp = frame["datetime"]
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    features["feat_hour_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    features["feat_hour_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    week_angle = 2.0 * np.pi * timestamp.dt.dayofweek / 7.0
    features["feat_day_of_week_sin"] = np.sin(week_angle)
    features["feat_day_of_week_cos"] = np.cos(week_angle)

    bf_prod = column_sum(frame, BALANCE_COMPONENTS["blast_furnace_production"])
    bf_sink = (column_sum(frame, BALANCE_COMPONENTS["air_heater_consumption"])
               + column_sum(frame, BALANCE_COMPONENTS["blast_furnace_user_consumption"])
               + frame["into_gas_mixed_blast_furnace"])
    balances = {
        "feat_blast_furnace_balance": bf_prod - bf_sink,
        "feat_coke_balance": frame["coke_oven_1"] - frame["into_gas_mixed_coke"],
        "feat_converter_balance": frame["converter_1"] - (frame["converter_user2"] + frame["into_gas_mixed_converter"]),
    }
    balances["feat_total_gas_balance"] = sum(balances.values())
    for name, series in balances.items():
        for window in (4, 16, 96):
            features[f"{name}_roll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
        features[name] = series
    holder = frame["blast_furnace_gas_holder_2"]
    for lag in (1, 4, 16):
        features[f"feat_holder_delta_{lag}"] = holder.diff(lag)
    for column in GENERATOR_GAS_USE_COLUMNS:
        for lag in (1, 4):
            features[f"feat_{column}_diff_{lag}"] = frame[column].diff(lag)
    return features


def target_features(series: pd.Series) -> pd.DataFrame:
    """Target-history features from the working series W (causal, one row per timestamp)."""
    out = pd.DataFrame(index=series.index)
    for lag in TARGET_LAG_STEPS:
        out[f"feat_wlag_{lag}"] = series.shift(lag)
    for lag in (1, 4):
        out[f"feat_wdiff_{lag}"] = series.diff(lag)
    for window in (4, 16, 96):
        out[f"feat_wroll_mean_{window}"] = series.rolling(window, min_periods=1).mean()
        out[f"feat_wroll_std_{window}"] = series.rolling(window, min_periods=2).std()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("dev", "build"), default="dev")
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)
    params = lightgbm_params(config)

    prepared = pd.read_csv(root / "outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
    physics = physics_features(prepared)

    if args.mode == "dev":
        # honest recursion simulation on the dev_03 fold: W is cut at Aug 31 23:45
        cutoff = pd.Timestamp("2025-08-31 23:45:00")
        origins = pd.date_range("2025-09-01 00:00:00", "2025-09-02 23:45:00", freq="15min")
        train_mask = prepared["datetime"] <= cutoff
        sim_mask = prepared["datetime"] <= origins.max() + pd.Timedelta(minutes=15 * 96)
        sim = prepared.loc[sim_mask].reset_index(drop=True)
        sim_physics = physics.loc[sim_mask].reset_index(drop=True)
        W = {t: sim[t].where(sim["datetime"] <= cutoff).ffill() for t in targets}
        # models trained on data <= cutoff with W = actuals (train region semantics)
        models: dict[tuple[str, int], lgb.LGBMRegressor] = {}
        feature_columns: list[str] | None = None
        for target in targets:
            W_train = prepared.loc[train_mask].set_index("datetime")[target]
            tf_train = target_features(W_train)
            phys_train = physics.loc[train_mask].reset_index(drop=True)
            phys_train.index = W_train.index
            feats = pd.concat([phys_train, tf_train], axis=1)
            feature_columns = list(feats.columns)
            for step in steps:
                label = pd.Series(prepared.loc[train_mask, target].to_numpy(), index=W_train.index).shift(-step)
                mask = label.notna()
                model = lgb.LGBMRegressor(**params)
                model.fit(feats.loc[mask], label[mask])
                models[(target, step)] = model
        print("models trained", flush=True)

        # recursive rollout over dev origins
        origin_pos = {t: int(np.where(sim["datetime"] == t)[0][0]) for t in origins}
        preds = {(t, s): np.full(len(origins), np.nan) for t in targets for s in steps}
        for k, t_origin in enumerate(origins):
            p = origin_pos[t_origin]
            for target in targets:
                # W currently holds predictions up to t_origin (filled by previous origins)
                tf_row = target_features(W[target]).iloc[p]
                row = pd.concat([sim_physics.loc[p], tf_row])
                for step in steps:
                    value = float(np.clip(models[(target, step)].predict(row.to_numpy(float).reshape(1, -1))[0], 0.0, None))
                    preds[(target, step)][k] = value
                    target_time = p + step
                    if target_time < len(sim) and sim["datetime"].iloc[target_time] > t_origin:
                        W[target].iloc[target_time] = value
            if k % 48 == 0:
                print(f"rollout {k}/{len(origins)}", flush=True)

        # evaluate against interval-mean actuals
        minute_src = pd.read_csv(root / "data/Semi_load.csv", parse_dates=["datetime"]).set_index("datetime")
        from holdout_calibration_semi import pooled_mape
        summary = {}
        for target in targets:
            actuals = []
            for step in steps:
                times = origins + pd.Timedelta(minutes=step * 15)
                actuals.append(np.array([minute_src[target].loc[t - pd.Timedelta(minutes=14): t].mean() for t in times]))
            actual_matrix = np.column_stack(actuals)
            pred_matrix = np.column_stack([preds[(target, s)] for s in steps])
            valid = actual_matrix != 0
            mape = pooled_mape(actual_matrix[valid], pred_matrix[valid])
            per_step = [pooled_mape(actual_matrix[:, j][valid[:, j]], pred_matrix[:, j][valid[:, j]]) for j in range(len(steps))]
            violations = int(sum(1 for j in range(1, len(per_step)) if per_step[j] < per_step[j - 1]))
            summary[target] = {"mape": mape, "score_1_mape": 1 - mape, "violations": violations,
                               "per_step_first8": [round(v, 4) for v in per_step[:8]]}
            print(target, json.dumps(summary[target]), flush=True)
        (root / "outputs/recursive_dev03_v8.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"done in {time.perf_counter()-started:.0f}s -> outputs/recursive_dev03_v8.json")
        return 0

    if args.mode == "build":
        # full-training fit + rollout over the 960 test origins with the same
        # recursive semantics as the dev simulation, using O(1) incremental
        # window statistics (positions <= p are final when origin p is processed)
        from build_submission import file_hash, git_state, validate_submission

        test = pd.read_csv(root / "outputs/prepared_v2/test_prepared.csv", parse_dates=["datetime"])
        comb = pd.concat([prepared, test], ignore_index=True)
        physics_all = physics_features(comb)
        n_train = len(prepared)

        W_arr = {t: comb[t].to_numpy(float).copy() for t in targets}
        for t in targets:
            W_arr[t][n_train:] = W_arr[t][n_train - 1]  # stale fill, overwritten during rollout

        models: dict[tuple[str, int], lgb.LGBMRegressor] = {}
        feature_columns: list[str] | None = None
        for target in targets:
            W_series = pd.Series(W_arr[target][:n_train])
            tf_train = target_features(W_series)
            phys_train = physics_all.iloc[:n_train].reset_index(drop=True)
            feats = pd.concat([phys_train, tf_train], axis=1)
            feature_columns = list(feats.columns)
            for step in steps:
                label = comb[target].shift(-step).iloc[:n_train]
                mask = label.notna()
                model = lgb.LGBMRegressor(**params)
                model.fit(feats.loc[mask], label[mask])
                models[(target, step)] = model
            print(f"trained {target}", flush=True)

        # incremental window statistics over the finalized prefix of W
        predictions: dict[tuple[str, int], np.ndarray] = {(t, s): np.zeros(960) for t in targets for s in steps}
        roll_windows = (4, 16, 96)
        state: dict[str, dict[int, dict[str, object]]] = {
            t: {w: {"sum": 0.0, "sum2": 0.0, "queue": []} for w in roll_windows} for t in targets
        }

        test_rows = physics_all.iloc[n_train:].reset_index(drop=True)
        out_index = list(test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"))
        for k in range(960):
            p = n_train + k
            for target in targets:
                w = W_arr[target]
                # commit W[p] (final: written by origin p-1 step 1, or the stale fill at k=0)
                wp = w[p]
                for win in roll_windows:
                    st = state[target][win]
                    st["sum"] += wp
                    st["sum2"] += wp * wp
                    st["queue"].append(wp)
                    if len(st["queue"]) > win:
                        old = st["queue"].pop(0)
                        st["sum"] -= old
                        st["sum2"] -= old * old
                row_vals = {}
                for lag in TARGET_LAG_STEPS:
                    row_vals[f"feat_wlag_{lag}"] = w[p - lag]
                row_vals["feat_wdiff_1"] = wp - w[p - 1]
                row_vals["feat_wdiff_4"] = wp - w[p - 4]
                for win in roll_windows:
                    st = state[target][win]
                    n = len(st["queue"])
                    mean = st["sum"] / n
                    if n < 2:
                        std = float("nan")
                    else:
                        var = max(st["sum2"] / n - mean * mean, 0.0) * n / (n - 1)
                        std = float(np.sqrt(var))
                    row_vals[f"feat_wroll_mean_{win}"] = mean
                    row_vals[f"feat_wroll_std_{win}"] = std
                tf_row = pd.Series(row_vals)
                row = pd.concat([test_rows.loc[k], tf_row])
                X = row.to_numpy(float).reshape(1, -1)
                for step in steps:
                    value = float(np.clip(models[(target, step)].predict(X)[0], 0.0, None))
                    predictions[(target, step)][k] = value
                    future = p + step
                    if future < len(w):
                        w[future] = value  # may be overwritten later by a fresher origin's step-1
            if k % 96 == 0:
                print(f"rollout {k}/960", flush=True)

        def build_frame(horizon_list: list[int]) -> pd.DataFrame:
            frame = pd.DataFrame({"datetime": out_index})
            for target in targets:
                for h in horizon_list:
                    frame[f"{target}_t+{h}_pred"] = np.round(predictions[(target, h // 15)], 6)
            return frame

        long_out = build_frame(minutes)
        short_out = build_frame([m for m in minutes if m <= 120])
        predictions.clear()

        submission_input = pd.read_csv(root / "outputs/prepared_v2/input.csv")
        expected = pd.Series(out_index)
        validation_long = validate_submission(submission_input, long_out, expected, targets, minutes)
        validation_short = validate_submission(submission_input, short_out, expected, targets, [m for m in minutes if m <= 120])

        out_dir = root / config["output"]["root"] / "20260927T_recursive_v8"
        out_dir.mkdir(parents=True, exist_ok=False)
        short_path, long_path = out_dir / "s_result.csv", out_dir / "l_result.csv"
        short_out.to_csv(short_path, index=False, encoding="utf-8")
        long_out.to_csv(long_path, index=False, encoding="utf-8")
        archive_path = out_dir / f"{config['team_name']}_{config['output']['archive_suffix']}"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(short_path, arcname="s_result.csv")
            archive.write(long_path, arcname="l_result.csv")
        manifest = {
            "submission_id": "20260927T_recursive_v8",
            "status": "verified",
            "competition_stage": "semi",
            "team_name": config["team_name"],
            "selection_policy": "recursive_w_feature_stream_frozen_lightgbm_direct",
            "dev03_interval_mean_1_mape": json.loads((root / "outputs/recursive_dev03_v8.json").read_text(encoding="utf-8")),
            "validation_long": validation_long,
            "validation_short": validation_short,
            "artifacts": {
                archive_path.relative_to(root).as_posix(): file_hash(archive_path),
                short_path.relative_to(root).as_posix(): file_hash(short_path),
                long_path.relative_to(root).as_posix(): file_hash(long_path),
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

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
