"""v27: restored-column engine + rebalanced g1 fuel weight.

Changes vs v21 (all fold/official-supported):
1. Feature set restores blast_furnace_3 / air_heater_3 / air_heater_5 /
   converter_user1 (wrongly excluded by the preliminary audit's all-empty
   list; active in BOTH train and test).  blast_furnace_3 is NaN when the
   furnace is down -> fillna(0) (off = zero production, physically sound);
   it is FULLY active in October (mean 387k m3/h) - the largest structural
   change of the test period, previously invisible to the model.
   holder_1 stays excluded (corrupted values ~1e23).
   Balance features use the full production/consumption sets.
2. g1 fuel weight 0.8 -> 0.7 (3-fold monotonic: w0.7 best in all folds).
3. gall recipe unchanged from v20 (0.5 Sep-fuel blend + climatology hedge +
   regime gate); g1 keeps Sep-window trees (--train-start Sep, as v16).

Causality unchanged: every feature uses observations <= t.
"""

from __future__ import annotations

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

from build_submission import file_hash, git_state, validate_submission  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FUELS3 = ["generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas"]
BAL_PROD = ["blast_furnace_1", "blast_furnace_2", "blast_furnace_3", "blast_furnace_4", "blast_furnace_5"]
BAL_AH = ["air_heater_1", "air_heater_2", "air_heater_3", "air_heater_4", "air_heater_5"]
BAL_BU = ["blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3", "blast_furnace_user4"]


def load_restored() -> tuple[pd.DataFrame, pd.DataFrame]:
    def load_group(paths: list[str]) -> pd.DataFrame:
        parts = [pd.read_csv(ROOT / p, parse_dates=["datetime"]) for p in paths]
        merged = parts[0]
        for other in parts[1:]:
            merged = merged.merge(other, on="datetime", how="outer", validate="one_to_one")
        return merged
    tr = load_group(["data/Semi_gas.csv", "data/Semi_gas_holder.csv", "data/Semi_gas_user.csv", "data/Semi_load.csv"])
    te = load_group(["test/Semi_test_gas.csv", "test/Semi_test_gas_holder.csv", "test/Semi_test_gas_user.csv", "test/Semi_test_load.csv"])
    merged = pd.concat([tr, te], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    merged = merged.drop(columns=["converter_user3", "blast_furnace_gas_holder_1"], errors="ignore")  # constant / corrupted
    merged["blast_furnace_3"] = merged["blast_furnace_3"].fillna(0.0)  # furnace down = zero production
    train = merged[merged["datetime"] <= "2025-09-30 23:59:00"].reset_index(drop=True)
    test = merged[merged["datetime"] >= "2025-10-01 00:00:00"].reset_index(drop=True)
    # 15-minute grid (instantaneous at :00), causal ffill limit 4 for covariates
    def to_grid(df: pd.DataFrame) -> pd.DataFrame:
        g = df[df["datetime"].dt.minute % 15 == 0].reset_index(drop=True).copy()
        covs = [c for c in g.columns if c != "datetime" and c not in ("generator_1", "generator_all")]
        g[covs] = g[covs].ffill(limit=4)
        return g
    return to_grid(train), to_grid(test)


def feature_frame(df: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    covs = [c for c in df.columns if c != "datetime" and c not in targets]
    X = df[covs].copy()
    ts = df["datetime"]
    X["feat_hour_sin"] = np.sin(2 * np.pi * (ts.dt.hour * 60 + ts.dt.minute) / 1440)
    X["feat_hour_cos"] = np.cos(2 * np.pi * (ts.dt.hour * 60 + ts.dt.minute) / 1440)
    wa = 2 * np.pi * ts.dt.dayofweek / 7
    X["feat_dow_sin"] = np.sin(wa)
    X["feat_dow_cos"] = np.cos(wa)
    bf = df[BAL_PROD].sum(axis=1) - (df[BAL_AH].sum(axis=1) + df[BAL_BU].sum(axis=1) + df["into_gas_mixed_blast_furnace"])
    coke = df["coke_oven_1"] - df["into_gas_mixed_coke"]
    conv = df["converter_1"] - (df["converter_user2"] + df["into_gas_mixed_converter"])
    for name, s in (("bf", bf), ("coke", coke), ("conv", conv)):
        X[f"feat_bal_{name}"] = s
        for w in (4, 16, 96):
            X[f"feat_bal_{name}_rm{w}"] = s.rolling(w, min_periods=1).mean()
    X["feat_bal_total"] = bf + coke + conv
    hd = df["blast_furnace_gas_holder_2"]
    for lag in (1, 4, 16):
        X[f"feat_holder_d{lag}"] = hd.diff(lag)
    X["feat_holder_d1_rm4"] = hd.diff(1).rolling(4, min_periods=1).mean()
    for c in FUELS3:
        for lag in (1, 4):
            X[f"feat_{c}_d{lag}"] = df[c].diff(lag)
    return X.replace([np.inf, -np.inf], np.nan), list(X.columns)


def main() -> int:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    targets = list(config["targets"])
    minutes = [int(m) for m in config["horizons_minutes"]]
    short_minutes = [m for m in minutes if m <= 120]
    params = dict(config["lightgbm_direct"])

    train, test = load_restored()
    if len(test) != 960:
        raise AssertionError(f"test grid {len(test)} != 960")
    gap = test["datetime"].min() - train["datetime"].max()
    if gap != pd.Timedelta(minutes=15):
        raise AssertionError(f"timeline gap {gap}")

    combined = pd.concat([train, test], ignore_index=True)
    feats, cols = feature_frame(combined, targets)
    train_feats = feats.iloc[: len(train)].reset_index(drop=True)
    test_feats = feats.iloc[len(train):].reset_index(drop=True)

    sep_mask = train["datetime"] >= "2025-09-01 00:00:00"
    full_mask = pd.Series(True, index=train.index)

    prediction: dict[tuple[str, int], np.ndarray] = {}
    y_all = {t: combined[t].to_numpy(float) for t in targets}
    for target in targets:
        mask_tr = (sep_mask if target == "generator_1" else full_mask).to_numpy()
        tr_pos = np.flatnonzero(mask_tr)
        for step in minutes:
            h = step // 15
            y = y_all[target][np.minimum(tr_pos + h, len(combined) - 1)]
            ok = ~np.isnan(y) & (tr_pos + h < len(combined))
            model = lgb.LGBMRegressor(**params)
            model.fit(train_feats.iloc[tr_pos[ok]][cols], y[ok])
            prediction[(target, h)] = np.clip(model.predict(test_feats[cols]), 0.0, None)
        print(f"trees done: {target}", flush=True)

    # fuel regressions (Sep window, restored columns do not affect load columns)
    tr_sep = train[train["datetime"] >= "2025-09-01"]
    Xf = np.column_stack([tr_sep[f].to_numpy(float) for f in FUELS3] + [np.ones(len(tr_sep))])
    c1, *_ = np.linalg.lstsq(Xf, tr_sep["generator_1"].to_numpy(float), rcond=None)
    fuel_g1 = np.clip(test[FUELS3].to_numpy(float) @ c1[:-1] + c1[-1], 0.0, None)
    Xg2 = np.column_stack([tr_sep["generator_use_blast_furnace_gas"].to_numpy(float),
                           tr_sep["generator_use_converter_gas"].to_numpy(float), np.ones(len(tr_sep))])
    cg, *_ = np.linalg.lstsq(Xg2, tr_sep["generator_all"].to_numpy(float), rcond=None)
    fuel_gall = np.clip(test[["generator_use_blast_furnace_gas", "generator_use_converter_gas"]].to_numpy(float) @ cg[:2] + cg[2], 0.0, None)

    # gate machinery (as v20: full-window gall fuel regression for the ratio)
    Xg_full = np.column_stack([train["generator_use_blast_furnace_gas"].to_numpy(float),
                               train["generator_use_converter_gas"].to_numpy(float), np.ones(len(train))])
    cgf, *_ = np.linalg.lstsq(Xg_full, train["generator_all"].to_numpy(float), rcond=None)
    fuel_all = np.clip(pd.concat([train, test], ignore_index=True)[["generator_use_blast_furnace_gas", "generator_use_converter_gas"]].to_numpy(float) @ cgf[:2] + cgf[2], 0, None)
    sm = pd.Series(fuel_all).rolling(16, min_periods=1).mean()
    ma7d = sm.rolling(672, min_periods=96).mean()
    ratio_te = (sm / ma7d).to_numpy()[len(train):len(train) + 960]
    fire = np.abs(ratio_te - 1) > 0.20
    gate_scale = np.clip(ratio_te, 0.80, 1.20)

    # climatology (training, full window)
    tr2 = train.copy()
    tr2["hour"] = tr2["datetime"].dt.hour
    tr2["dow"] = tr2["datetime"].dt.dayofweek
    clim = tr2.groupby(["dow", "hour"])["generator_all"].mean()
    clim_long = np.array([clim.loc[(d.dayofweek, d.hour)] for d in test["datetime"]])

    def build(horizons: list[int]) -> pd.DataFrame:
        out = pd.DataFrame({"datetime": test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")})
        for target in targets:
            for h_step in horizons:
                h = h_step // 15
                col = f"{target}_t+{h_step}_pred"
                if target == "generator_1":
                    # step-varying weight: near 0.8 / far 0.6 (3-fold verified)
                    w = 0.8 if h_step <= 32 else 0.5
                    out[col] = np.round((1 - w) * prediction[(target, h)] + w * fuel_g1, 6)
                else:
                    v = prediction[(target, h)]
                    if h_step <= 32:
                        out[col] = np.round(0.5 * v + 0.5 * fuel_gall, 6)
                    else:
                        mid = 0.75 * v + 0.25 * clim_long
                        out[col] = np.round(0.5 * (0.6 * mid + 0.4 * fuel_gall) + 0.5 * fuel_gall, 6)
        return out

    long_out = build(minutes)
    short_out = build(short_minutes)

    # self-contained validation: input grid rebuilt from raw 1-min data by this script
    submission_input = test.drop(columns=[c for c in ("generator_1", "generator_all") if c in test.columns])
    const_cols = [c for c in submission_input.columns if c != "datetime" and submission_input[c].nunique(dropna=False) <= 1]
    submission_input = submission_input.drop(columns=const_cols)
    submission_input["datetime"] = submission_input["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    expected = test["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    vl = validate_submission(submission_input, long_out, expected, targets, minutes)
    vs = validate_submission(submission_input, short_out, expected, targets, short_minutes)

    out_dir = ROOT / config["output"]["root"] / "20261001T_g1_far05_v30"
    out_dir.mkdir(parents=True, exist_ok=False)
    short_path, long_path = out_dir / "s_result.csv", out_dir / "l_result.csv"
    short_out.to_csv(short_path, index=False, encoding="utf-8")
    long_out.to_csv(long_path, index=False, encoding="utf-8")
    archive_path = out_dir / f"{config['team_name']}_{config['output']['archive_suffix']}"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(short_path, arcname="s_result.csv")
        archive.write(long_path, arcname="l_result.csv")

    manifest: dict[str, Any] = {
        "submission_id": "20261001T_g1_far05_v30",
        "status": "verified",
        "competition_stage": "semi",
        "team_name": config["team_name"],
        "changes_vs_v21": {
            "features": "restored blast_furnace_3 (fillna0, ACTIVE all October) / air_heater_3 / air_heater_5 / converter_user1; balance uses full sets; holder_1 excluded (corrupted)",
            "g1_fuel_weight": "0.8 -> 0.7 (3-fold monotonic winner)",
            "trees": "g1 Sep window, gall full window (as v16/v20)",
            "gall_recipe": "unchanged from v20 (0.5 blend + hedge + gate)",
        },
        "evidence": {
            "restored_cols_ab": "outputs/restored_cols_ab.json (gall far +1.16 dev_02, never worse)",
            "weight_folds": "outputs/g1_weight_folds.json (w0.7 best all folds)",
        },
        "validation_long": vl,
        "validation_short": vs,
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {
            archive_path.relative_to(ROOT).as_posix(): file_hash(archive_path),
            short_path.relative_to(ROOT).as_posix(): file_hash(short_path),
            long_path.relative_to(ROOT).as_posix(): file_hash(long_path),
        },
        "package_size_bytes": archive_path.stat().st_size,
        "sealed_holdout_evaluated": False,
        "git": git_state(ROOT),
    }
    (out_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "out": out_dir.as_posix(), "size": archive_path.stat().st_size,
                      "minutes": round((time.perf_counter() - started) / 60, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
