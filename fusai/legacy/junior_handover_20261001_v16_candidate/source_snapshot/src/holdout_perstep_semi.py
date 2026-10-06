"""Per-step holdout diagnostics + trajectory-consistency candidates.

Question: does the per-step accuracy curve of the independent per-step models
violate the "closer to the origin must be more accurate" statistical review
(official rule 4-(4))?  And do simple trajectory-consistency fixes reduce
pooled interval-mean MAPE without hurting anything else?

Outputs (outputs/holdout_perstep_semi.json):
- per-step interval-mean MAPE for both targets (base model, trained <= Sep 27)
- monotonicity violation count
- candidate variants evaluated on the same holdout:
    base                     : current per-step predictions
    shrink_gall_096          : gall x 0.96
    smooth3                  : per-origin moving average over adjacent steps
    smooth3_plus_shrink      : combination
    far_climatology_blend    : far horizons blended toward training diurnal means
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission_semi import (  # noqa: E402
    horizons_from_config,
    lightgbm_params,
    make_feature_frame_semi,
)
from holdout_calibration_semi import pooled_mape  # noqa: E402


def smooth_across_steps(pred: np.ndarray, window: int = 3) -> np.ndarray:
    """Per-origin moving average along the step axis (edges: shrinking window)."""
    out = pred.copy()
    n = pred.shape[1]
    for j in range(n):
        lo, hi = max(0, j - window // 2), min(n, j + window // 2 + 1)
        out[:, j] = pred[:, lo:hi].mean(axis=1)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)

    train = pd.read_csv(root / "outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
    features, columns = make_feature_frame_semi(train, targets)
    features["datetime"] = train["datetime"].values

    train_end = pd.Timestamp("2025-09-27 23:45:00")
    origins = pd.date_range("2025-09-28 00:00:00", "2025-09-29 23:45:00", freq="15min")
    origin_mask = train["datetime"].isin(set(origins))
    train_mask = train["datetime"] <= train_end
    train_features = features[train_mask].reset_index(drop=True)
    origin_features = features.loc[origin_mask].reset_index(drop=True)

    params = lightgbm_params(config)
    minute_src = pd.read_csv(root / "data/Semi_load.csv", parse_dates=["datetime"]).set_index("datetime")

    cache = root / "outputs/holdout_perstep_cache.npz"
    if cache.exists():
        blob = np.load(cache)
        base_matrices = {t: blob[t] for t in targets}
        actual_matrices = {t: blob[t + "_actual"] for t in targets}
    else:
        base_matrices: dict[str, np.ndarray] = {}
        actual_matrices: dict[str, np.ndarray] = {}
        for target in targets:
            series = train[target]
            preds, actuals = [], []
            for step in steps:
                label_all = series.shift(-step)
                labels = label_all[train_mask]
                mask = labels.notna()
                model = lgb.LGBMRegressor(**params)
                model.fit(train_features.loc[mask, columns], labels[mask])
                preds.append(np.clip(model.predict(origin_features[columns]), 0.0, None))
                target_times = origins + pd.Timedelta(minutes=step * 15)
                actuals.append(np.array([minute_src[target].loc[t - pd.Timedelta(minutes=14): t].mean() for t in target_times]))
            base_matrices[target] = np.column_stack(preds)
            actual_matrices[target] = np.column_stack(actuals)
            print(f"retrained {target}", flush=True)
        np.savez_compressed(cache, **{t: base_matrices[t] for t in targets},
                            **{t + "_actual": actual_matrices[t] for t in targets})

    report: dict = {"origins": int(len(origins)), "per_step": {}, "candidates": {}}
    for target in targets:
        per_step = [pooled_mape(actual_matrices[target][:, j][actual_matrices[target][:, j] != 0],
                                base_matrices[target][:, j][actual_matrices[target][:, j] != 0])
                    for j in range(len(steps))]
        violations = int(sum(1 for j in range(1, len(per_step)) if per_step[j] < per_step[j - 1]))
        report["per_step"][target] = {"mape": [round(v, 5) for v in per_step], "monotonic_violations": violations}
        print(f"{target}: per-step computed, violations={violations}", flush=True)

    origin_keys = list(zip(origins.dayofweek, origins.hour))
    train["hour"] = train["datetime"].dt.hour
    train["dow"] = train["datetime"].dt.dayofweek
    clim = train.groupby(["dow", "hour"])[targets].mean()
    clim_matrix = {t: clim.loc[origin_keys, t].to_numpy(float) for t in targets}

    def evaluate(name: str, matrices: dict[str, np.ndarray]) -> None:
        entry = {}
        for t in targets:
            entry[t] = pooled_mape(actual_matrices[t][actual_matrices[t] != 0], matrices[t][actual_matrices[t] != 0])
        entry["file_avg_1_mape"] = float(np.mean([1 - v for v in entry.values()]))
        report["candidates"][name] = entry
        print(name, {k: round(v, 4) for k, v in entry.items()}, flush=True)

    evaluate("base", base_matrices)

    shrunk = {t: (m * 0.96 if t == "generator_all" else m) for t, m in base_matrices.items()}
    evaluate("shrink_gall_096", shrunk)

    smooth = {t: smooth_across_steps(m) for t, m in base_matrices.items()}
    evaluate("smooth3", smooth)

    combo = {t: (smooth_across_steps(base_matrices[t]) * (0.96 if t == "generator_all" else 1.0)) for t in targets}
    evaluate("smooth3_plus_shrink", combo)

    for w in (0.2, 0.35, 0.5):
        blend = {}
        for t in targets:
            m = base_matrices[t].copy()
            far = slice(32, 96)  # horizons beyond 8h blend toward climatology
            m[:, far] = (1 - w) * m[:, far] + w * clim_matrix[t][:, None]
            blend[t] = m
        evaluate(f"far_climatology_blend_w{w}", blend)

    (root / "outputs/holdout_perstep_semi.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved -> outputs/holdout_perstep_semi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
