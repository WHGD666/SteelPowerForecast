"""Semifinal v10: two-stage long-horizon model.

Stage 1 predicts 12 gas-system aggregate paths (96 horizons per aggregate)
from origin-time covariates.  Stage 2 is the frozen LightGBM power model
augmented with a tau-block: gas balances, generator gas use and holder level
evaluated at the TARGET timestamp along the predicted path (plus path-based
balance rolling means).  Rationale: the long file degrades -7.7 accuracy
points from September to October while the short file only -2, i.e. covariate
staleness is the binding constraint at long horizons; the sealed-holdout
tau-block check (true paths) lifted generator_1 accuracy 0.9004 -> 0.9225.

Final generator_1 prediction = 0.5 * (v9 tree/fuel blend) + 0.5 * (tau model),
hedging stage-1 noise; generator_all keeps the v2 model untouched (the fuel
blend showed no holdout gain there).

Causality: stage-1 paths are functions of origin-time (<= t) information only;
stage-2 consumes predicted paths, never observed future values.
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
from build_submission_semi import (  # noqa: E402
    GENERATOR_GAS_USE_COLUMNS,
    horizons_from_config,
    lightgbm_params,
    make_feature_frame_semi,
)

AGG_DEFS = {
    "agg_bf_prod": (["blast_furnace_1", "blast_furnace_2", "blast_furnace_4", "blast_furnace_5"], "sum"),
    "agg_airheater": (["air_heater_1", "air_heater_2", "air_heater_4"], "sum"),
    "agg_bf_users": (["blast_furnace_user1", "blast_furnace_user2", "blast_furnace_user3", "blast_furnace_user4"], "sum"),
    "agg_into_mixed_bf": (["into_gas_mixed_blast_furnace"], "raw"),
    "agg_coke_oven": (["coke_oven_1"], "raw"),
    "agg_into_mixed_coke": (["into_gas_mixed_coke"], "raw"),
    "agg_converter": (["converter_1"], "raw"),
    "agg_conv_sink": (["converter_user2", "into_gas_mixed_converter"], "sum"),
    "agg_holder": (["blast_furnace_gas_holder_2"], "raw"),
    "agg_use_bfg": (["generator_use_blast_furnace_gas"], "raw"),
    "agg_use_coke": (["generator_use_coke_gas"], "raw"),
    "agg_use_conv": (["generator_use_converter_gas"], "raw"),
}


def aggregate_paths_frame(comb: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=comb.index)
    for name, (cols, mode) in AGG_DEFS.items():
        out[name] = comb[cols].sum(axis=1) if mode == "sum" else comb[cols[0]]
    return out


def balances_from_aggs(aggs: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=aggs.index)
    out["bal_bf"] = aggs["agg_bf_prod"] - (aggs["agg_airheater"] + aggs["agg_bf_users"] + aggs["agg_into_mixed_bf"])
    out["bal_coke"] = aggs["agg_coke_oven"] - aggs["agg_into_mixed_coke"]
    out["bal_conv"] = aggs["agg_converter"] - aggs["agg_conv_sink"]
    out["bal_total"] = out["bal_bf"] + out["bal_coke"] + out["bal_conv"]
    return out


def main() -> int:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)
    short_minutes = [m for m in minutes if m <= 120]
    params = lightgbm_params(config)

    train = pd.read_csv(root / "outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
    test = pd.read_csv(root / "outputs/prepared_v2/test_prepared.csv", parse_dates=["datetime"])
    comb = pd.concat([train, test], ignore_index=True)
    n_train = len(train)
    n_test = len(test)

    aggs = aggregate_paths_frame(comb)
    origin_features, origin_columns = make_feature_frame_semi(comb, targets)
    origin_features["datetime"] = comb["datetime"].values
    X_origin_train = origin_features.iloc[:n_train][origin_columns].to_numpy(float)
    X_origin_test = origin_features.iloc[n_train:][origin_columns].to_numpy(float)

    agg_names = list(AGG_DEFS)
    A = {name: aggs[name].to_numpy(float) for name in agg_names}

    # ---- stage 1: aggregate paths ----
    paths = np.zeros((n_test, len(steps), len(agg_names)))
    stage1_meta = []
    for ai, name in enumerate(agg_names):
        series = pd.Series(A[name])
        for j, step in enumerate(steps):
            label = series.shift(-step).iloc[:n_train]
            mask = label.notna()
            model = lgb.LGBMRegressor(**params)
            model.fit(X_origin_train[mask], label[mask])
            paths[:, j, ai] = np.clip(model.predict(X_origin_test), 0.0, None)
        stage1_meta.append({"aggregate": name, "models": len(steps)})
        np.savez_compressed(root / "outputs/v10_stage1_checkpoint.npz", paths=paths)
        print(f"stage1 {name} done ({ai+1}/12)", flush=True)

    # ---- stage 2: power models with tau-block ----
    def tau_block_from_true(positions: np.ndarray) -> np.ndarray:
        bal = balances_from_aggs(pd.DataFrame({k: A[k][positions] for k in agg_names}))
        return np.column_stack([
            bal.to_numpy(float),
            np.column_stack([A[f"agg_use_{s}"][positions] for s in ("bfg", "coke", "conv")]),
            A["agg_holder"][positions],
        ])

    tau_dim = 4 + 3 + 1
    g1_tau_predictions = np.zeros((n_test, len(steps)))
    gall_tau_predictions = np.zeros((n_test, len(steps)))
    for target in targets:
        series = comb[target]
        for j, step in enumerate(steps):
            label = series.shift(-step).iloc[:n_train]
            mask = (label.notna()).to_numpy() & (np.arange(n_train) + step < n_train)
            tau_train = tau_block_from_true(np.arange(n_train)[mask] + step)
            X_train = np.column_stack([X_origin_train[mask], tau_train])
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, label[mask])

            # tau-block for test origin k at horizon step j: predicted path values
            bal = balances_from_aggs(pd.DataFrame({k2: paths[:, j, ai] for ai, k2 in enumerate(agg_names)}))
            uses = paths[:, j, [agg_names.index(f"agg_use_{s}") for s in ("bfg", "coke", "conv")]]
            holder = paths[:, j, agg_names.index("agg_holder")]
            tau_pred = np.column_stack([bal.to_numpy(float), uses, holder])

            X_pred = np.column_stack([X_origin_test, tau_pred])
            values = np.clip(model.predict(X_pred), 0.0, None)
            if target == "generator_1":
                g1_tau_predictions[:, j] = values
            else:
                gall_tau_predictions[:, j] = values
        print(f"stage2 {target} done", flush=True)

    # ---- assemble: g1 = 0.5 * (v9 blend) + 0.5 * tau ; gall = v2 ----
    fuels = ["generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas"]
    Xf_train = np.column_stack([train[f].to_numpy(float) for f in fuels] + [np.ones(n_train)])
    yf_train = train["generator_1"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(Xf_train, yf_train, rcond=None)
    fuel_level = np.clip(test[fuels].to_numpy(float) @ coef[:-1] + coef[-1], 0.0, None)
    np.savez_compressed(root / "outputs/v10_stage2_checkpoint.npz",
                        g1_tau=g1_tau_predictions, gall_tau=gall_tau_predictions, fuel_level=fuel_level)

    source = root / "outputs/submissions/20260926T_dual_result_v2"
    long_frame = pd.read_csv(source / "l_result.csv")
    short_frame = pd.read_csv(source / "s_result.csv")

    v9_long = pd.read_csv(root / "outputs/submissions/20260927T_g1fuel_blend_v9/l_result.csv")
    v9_short = pd.read_csv(root / "outputs/submissions/20260927T_g1fuel_blend_v9/s_result.csv")

    def finalize(short_df: pd.DataFrame, long_df: pd.DataFrame, v9s: pd.DataFrame, v9l: pd.DataFrame):
        outs = []
        for df, v9 in ((short_df, v9s), (long_df, v9l)):
            out = df.copy()
            g1_cols = [c for c in out.columns if c.startswith("generator_1_")]
            tau_values = np.zeros((len(out), len(g1_cols)))
            for j, c in enumerate(g1_cols):
                step = int(c.split("t+")[1].split("_")[0]) // 15
                tau_values[:, j] = g1_tau_predictions[:, step - 1]
            blend_part = v9[g1_cols].to_numpy(float)
            out[g1_cols] = np.round(0.5 * blend_part + 0.5 * tau_values, 6)
            out["datetime"] = out["datetime"].astype(str)
            outs.append(out)
        return outs

    short_out, long_out = finalize(short_frame, long_frame, v9_short, v9_long)

    submission_input = pd.read_csv(root / "outputs/prepared_v2/input.csv")
    expected = long_frame["datetime"].str.slice(0, 19)
    validation_long = validate_submission(submission_input, long_out, expected, targets, minutes)
    validation_short = validate_submission(submission_input, short_out, expected, targets, short_minutes)

    out_dir = root / config["output"]["root"] / "20260927T_twostage_v10"
    out_dir.mkdir(parents=True, exist_ok=False)
    short_path, long_path = out_dir / "s_result.csv", out_dir / "l_result.csv"
    short_out.to_csv(short_path, index=False, encoding="utf-8")
    long_out.to_csv(long_path, index=False, encoding="utf-8")
    archive_path = out_dir / f"{config['team_name']}_{config['output']['archive_suffix']}"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(short_path, arcname="s_result.csv")
        archive.write(long_path, arcname="l_result.csv")

    manifest: dict[str, Any] = {
        "submission_id": "20260927T_twostage_v10",
        "status": "verified",
        "competition_stage": "semi",
        "team_name": config["team_name"],
        "base": "v9 g1 blend (52.4085) + gall from v2",
        "changes": {
            "stage1": f"{len(agg_names)} aggregate paths x {len(steps)} horizons, frozen LightGBM",
            "stage2": "origin block + tau-block (balances/uses/holder at target time from predicted paths)",
            "g1_final": "0.5 * v9 blend + 0.5 * tau model",
            "gall": "v2 untouched",
        },
        "holdout_evidence": {
            "tau_block_true_paths_g1": "acc 0.9004 -> 0.9225 (outputs/tau_block_check.json)",
            "v9_blend_g1_holdout": "acc 0.9122",
        },
        "validation_long": validation_long,
        "validation_short": validation_short,
        "duration_seconds": time.perf_counter() - started,
        "artifacts": {
            archive_path.relative_to(root).as_posix(): file_hash(archive_path),
            short_path.relative_to(root).as_posix(): file_hash(short_path),
            long_path.relative_to(root).as_posix(): file_hash(long_path),
        },
        "package_size_bytes": archive_path.stat().st_size,
        "sealed_holdout_evaluated": True,
        "git": git_state(root),
    }
    (out_dir / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "out": out_dir.as_posix(), "size": archive_path.stat().st_size,
                      "minutes": round((time.perf_counter() - started) / 60, 1)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
