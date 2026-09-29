"""Stage-2 design check: do target-time (tau) covariate features beat origin-time only?

The long file degrades far more from September to October (holdout 90.65 ->
official 82.93) than the short file (-> 89.08), pointing at covariate
staleness: the current models describe every horizon with origin-time
covariates, so a t+1440 prediction is driven by 24-hour-old gas system state.

This check trains per-step models whose features include a tau-block
(gas balances, generator gas use, holder level evaluated AT the target
timestamp, plus balance rolling means along the actual history ending at tau)
on top of the origin-time block, and compares holdout interval-mean MAPE
against the origin-only baseline.  Paths here are TRUE covariate values, so
the question answered is purely "does tau information help", decoupled from
stage-1 prediction noise.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
    make_feature_frame_semi,
)
from holdout_calibration_semi import pooled_mape  # noqa: E402


def balance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    bf_prod = column_sum(frame, BALANCE_COMPONENTS["blast_furnace_production"])
    bf_sink = (column_sum(frame, BALANCE_COMPONENTS["air_heater_consumption"])
               + column_sum(frame, BALANCE_COMPONENTS["blast_furnace_user_consumption"])
               + frame["into_gas_mixed_blast_furnace"])
    out = pd.DataFrame(index=frame.index)
    out["bal_bf"] = bf_prod - bf_sink
    out["bal_coke"] = frame["coke_oven_1"] - frame["into_gas_mixed_coke"]
    out["bal_converter"] = frame["converter_1"] - (frame["converter_user2"] + frame["into_gas_mixed_converter"])
    out["bal_total"] = out["bal_bf"] + out["bal_coke"] + out["bal_converter"]
    return out


def main() -> int:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)
    params = lightgbm_params(config)

    train = pd.read_csv(root / "outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
    balances = balance_frame(train)
    origin_features, origin_columns = make_feature_frame_semi(train, targets)
    origin_features["datetime"] = train["datetime"].values

    train_end = pd.Timestamp("2025-09-27 23:45:00")
    origins = pd.date_range("2025-09-28 00:00:00", "2025-09-29 23:45:00", freq="15min")
    origin_mask = train["datetime"].isin(set(origins))
    train_mask = train["datetime"] <= train_end

    minute_src = pd.read_csv(root / "data/Semi_load.csv", parse_dates=["datetime"]).set_index("datetime")

    origin_positions = np.flatnonzero(origin_mask.to_numpy())
    train_positions = np.flatnonzero(train_mask.to_numpy())

    report: dict = {"mode": "tau_block_true_paths", "origins": int(len(origins))}
    for target in targets:
        base_preds, tau_preds, actuals = [], [], []
        for step in steps:
            # origin-time block (identical for every step)
            X_origin = origin_features.loc[origin_mask, origin_columns].to_numpy(float)

            # tau-block from TRUE values at t+h (honest for the design question)
            tau_pos = origin_positions + step
            tau_valid = tau_pos < len(train)
            bal = balances.iloc[tau_pos[tau_valid]]
            use_tau = train.iloc[tau_pos[tau_valid]][GENERATOR_GAS_USE_COLUMNS]
            holder_tau = train.iloc[tau_pos[tau_valid]]["blast_furnace_gas_holder_2"]
            # rolling means of balances along actual history ending at tau
            roll_frames = {}
            for window in (4, 16, 96):
                rolled = balances.rolling(window, min_periods=1).mean()
                roll_frames[f"bal_roll_{window}"] = rolled.iloc[tau_pos[tau_valid]]
            holder_roll = train["blast_furnace_gas_holder_2"].rolling(15, min_periods=1).mean().iloc[tau_pos[tau_valid]]

            tau_block = np.column_stack([
                bal.to_numpy(float),
                use_tau.to_numpy(float),
                holder_tau.to_numpy(float),
                holder_roll.to_numpy(float),
                *[roll_frames[f"bal_roll_{w}"].to_numpy(float) for w in (4, 16, 96)],
            ])

            label_all = train[target].shift(-step)
            labels = label_all[train_mask]
            mask = labels.notna()
            X_train_origin = origin_features.loc[train_mask, origin_columns].to_numpy(float)
            tau_pos_train = train_positions + step
            ok = tau_pos_train < len(train)
            mask = mask & pd.Series(ok, index=labels.index).fillna(False)
            bal_tr = balances.iloc[tau_pos_train[ok]]
            use_tr = train.iloc[tau_pos_train[ok]][GENERATOR_GAS_USE_COLUMNS]
            holder_tr = train.iloc[tau_pos_train[ok]]["blast_furnace_gas_holder_2"]
            holder_roll_tr = train["blast_furnace_gas_holder_2"].rolling(15, min_periods=1).mean().iloc[tau_pos_train[ok]]
            roll_tr = {w: balances.rolling(window, min_periods=1).mean().iloc[tau_pos_train[ok]] for w in (4, 16, 96)}
            tau_block_train = np.column_stack([
                bal_tr.to_numpy(float),
                use_tr.to_numpy(float),
                holder_tr.to_numpy(float),
                holder_roll_tr.to_numpy(float),
                *[roll_tr[w].to_numpy(float) for w in (4, 16, 96)],
            ])
            X_train = np.column_stack([X_train_origin[mask.to_numpy()], tau_block_train])

            m_base = lgb.LGBMRegressor(**params)
            m_base.fit(X_train_origin[mask.to_numpy()], labels[mask])
            base_preds.append(np.clip(m_base.predict(X_origin), 0.0, None))

            m_tau = lgb.LGBMRegressor(**params)
            m_tau.fit(X_train, labels[mask])
            X_pred_tau = np.column_stack([X_origin, tau_block])
            tau_preds.append(np.clip(m_tau.predict(X_pred_tau), 0.0, None))

            times = origins + pd.Timedelta(minutes=15 * step)
            a = np.array([minute_src[target].loc[t - pd.Timedelta(minutes=14): t].mean() for t in times])
            a_full = np.full(len(origins), np.nan)
            a_full[tau_valid] = a
            actuals.append(a_full)

        A = np.column_stack(actuals)
        B = np.column_stack(base_preds)
        T = np.column_stack(tau_preds)
        for name, M in (("origin_only", B), ("origin_plus_tau", T)):
            per_step = []
            for j in range(M.shape[1]):
                v = ~np.isnan(A[:, j]) & (A[:, j] != 0)
                per_step.append(pooled_mape(A[v, j], M[v, j]))
            mape = float(np.mean(per_step))
            violations = int(sum(1 for j in range(1, 96) if per_step[j] < per_step[j - 1]))
            report.setdefault(target, {})[name] = {"mape": mape, "acc": 1 - mape, "violations": violations}
            print(f"{target} {name}: MAPE {mape:.4f} acc {1-mape:.4f} violations {violations}", flush=True)

    (root / "outputs/tau_block_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter()-started:.0f}s -> outputs/tau_block_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
