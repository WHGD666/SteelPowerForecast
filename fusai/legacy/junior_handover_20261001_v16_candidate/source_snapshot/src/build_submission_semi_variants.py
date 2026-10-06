"""Rebuild the submitted semifinal packages (v9 / v11 / v12) byte-identically.

Every official submission must be regenerable from the repository.  This
script consolidates the package recipes that were originally executed as
one-off commands, so each submitted zip can be reproduced and its CSV
checksums compared against the archived originals.

Recipes (all start from the v2 base package outputs, which are themselves
deterministic LightGBM runs with fixed seeds):

- v9  : g1 columns = 0.5*tree + 0.5*fuel_linear  (fuel regression fit on the
        full 15-minute training table, three generator gas-use columns)
- v11 : v9 base, g1 = 0.3*tree + 0.7*fuel; long-file gall columns beyond
        step 32 blended 0.75*pred + 0.25*training (dow,hour) climatology
- v12 : v11 base, g1 = 0.2*tree + 0.8*fuel with the fuel regression refit on
        Aug 1 - Sep 30 only (recency bet)

Usage: python build_submission_semi_variants.py --variant v9 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

FUELS = ["generator_use_blast_furnace_gas", "generator_use_coke_gas", "generator_use_converter_gas"]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--variant", choices=("v9", "v11", "v12"), required=True)
    parser.add_argument("--verify", action="store_true", help="compare regenerated CSVs with the archived submission")
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    minutes = [int(m) for m in config["horizons_minutes"]]
    short_minutes = [m for m in minutes if m <= 120]

    base = root / "outputs/submissions/20260926T_dual_result_v2"
    s = pd.read_csv(base / "s_result.csv", parse_dates=["datetime"])
    l = pd.read_csv(base / "l_result.csv", parse_dates=["datetime"])
    train = pd.read_csv(root / "outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
    test = pd.read_csv(root / "outputs/prepared_v2/test_prepared.csv", parse_dates=["datetime"])

    def fit_fuel(tr: pd.DataFrame):
        X = np.column_stack([tr[f].to_numpy(float) for f in FUELS] + [np.ones(len(tr))])
        y = tr["generator_1"].to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        level = np.clip(test[FUELS].to_numpy(float) @ coef[:-1] + coef[-1], 0.0, None)
        return {d: v for d, v in zip(test["datetime"], level)}

    train = train.copy()
    train["hour"] = train["datetime"].dt.hour
    train["dow"] = train["datetime"].dt.dayofweek
    clim = train.groupby(["dow", "hour"])["generator_all"].mean()

    def apply_recipe(s_df: pd.DataFrame, l_df: pd.DataFrame):
        if args.variant == "v9":
            fuel = fit_fuel(train)
            def g1(df, n):
                vals = df.drop(columns="datetime").to_numpy(float).copy()
                fuel_col = np.array([fuel[d] for d in df["datetime"]])
                vals[:, :n] = np.round(0.5 * vals[:, :n] + 0.5 * np.repeat(fuel_col[:, None], n, axis=1), 6)
                out = df.copy(); out.iloc[:, 1:] = vals; return out
            return g1(s_df, len(short_minutes)), g1(l_df, len(minutes))
        fuel_span = train if args.variant == "v11" else train[train["datetime"] >= "2025-08-01 00:00:00"]
        fuel = fit_fuel(fuel_span)
        w_tree = 0.3 if args.variant == "v11" else 0.2
        def g1(df, n):
            vals = df.drop(columns="datetime").to_numpy(float).copy()
            fuel_col = np.array([fuel[d] for d in df["datetime"]])
            vals[:, :n] = np.round(w_tree * vals[:, :n] + (1 - w_tree) * np.repeat(fuel_col[:, None], n, axis=1), 6)
            out = df.copy(); out.iloc[:, 1:] = vals; return out
        def gall_far(df, n):
            if args.variant == "v9":
                return df
            vals = df.drop(columns="datetime").to_numpy(float).copy()
            climv = np.array([clim.loc[(d.dayofweek, d.hour)] for d in df["datetime"]])
            for j in range(n, 2 * n):
                if (j - n) + 1 > 32:
                    vals[:, j] = np.round(0.75 * vals[:, j] + 0.25 * climv, 6)
            out = df.copy(); out.iloc[:, 1:] = vals; return out
        s_out = g1(s_df, len(short_minutes))
        l_out = gall_far(g1(l_df, len(minutes)), len(minutes))
        return s_out, l_out

    s_out, l_out = apply_recipe(s.copy(), l.copy())
    for df in (s_out, l_out):
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    archive_names = {"v9": "20260927T_g1fuel_blend_v9", "v11": "20260929T_g1_70fuel_gall_farclim_v11", "v12": "20260929T_g1_80fuel_recent_v12"}
    archived = root / "outputs/submissions" / archive_names[args.variant]
    status = {"variant": args.variant}
    if args.verify:
        for name, new_df in (("s_result.csv", s_out), ("l_result.csv", l_out)):
            old = pd.read_csv(archived / name)
            same = old.equals(new_df)
            status[name] = "IDENTICAL" if same else "DIFFERS"
            if not same:
                diff = (old.astype(str) != new_df.astype(str)).sum().sum()
                status[name + "_cells_differing"] = int(diff)
    else:
        out_dir = root / "outputs/submissions" / f"reproduce_{args.variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        s_out.to_csv(out_dir / "s_result.csv", index=False)
        l_out.to_csv(out_dir / "l_result.csv", index=False)
        status["written"] = out_dir.as_posix()
        status["s_sha256"] = file_sha(out_dir / "s_result.csv")
        status["l_sha256"] = file_sha(out_dir / "l_result.csv")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import json
    raise SystemExit(main())
