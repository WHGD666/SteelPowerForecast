"""Sealed-holdout calibration for the semifinal candidate (one-time unsealing).

Purpose (per the frozen discipline: the holdout is consumed exactly once, to
calibrate/freeze the final candidate before submission):
- retrain the covariate-only semifinal model on data <= 2025-09-27 23:45;
- predict the sealed holdout origins (2025-09-28 00:00 .. 2025-09-29 23:45);
- report per-target pooled MAPE against (a) 15-minute point actuals and
  (b) 15-minute interval-mean actuals built from the 1-minute source;
- scan a per-target multiplicative shrinkage factor c in [0.85, 1.00] and
  report the MAPE-optimal c (MAPE-optimal forecasts sit below the conditional
  mean, so a mild downward correction is expected if the model is unbiased).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

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


def pooled_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.abs((actual[mask] - predicted[mask]) / actual[mask]).mean())


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
    origin_rows = train.loc[origin_mask].reset_index(drop=True)
    if origin_features["datetime"].tolist() != list(origins):
        raise AssertionError("holdout origin rows misaligned")

    params = lightgbm_params(config)
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for target in targets:
        series = train[target]
        for step in steps:
            label_all = series.shift(-step)
            labels = label_all[train_mask]
            mask = labels.notna()
            model = lgb.LGBMRegressor(**params)
            model.fit(train_features.loc[mask, columns], labels[mask])
            pred = np.clip(model.predict(origin_features[columns]), 0.0, None)
            predictions[(target, step)] = pred
        print(f"holdout retrain done for {target}", flush=True)

    # actuals: point values at origin + step, and 15-minute interval means of the 1-minute source
    minute_src = pd.read_csv(root / "data/Semi_load.csv", parse_dates=["datetime"]).set_index("datetime")
    report: dict[str, Any] = {"sealed_holdout_evaluated": True, "origins": int(len(origins))}
    for target in targets:
        point_block = []
        mean_block = []
        for step in steps:
            actual_point = series.shift(-step).loc[origin_mask].to_numpy(dtype=float)
            # interval mean over the 15 one-minute samples ending at the target time
            target_times = origins + pd.Timedelta(minutes=step * 15)
            interval_means = np.array(
                [minute_src[target].loc[t - pd.Timedelta(minutes=14): t].mean() for t in target_times]
            )
            point_block.append(actual_point)
            mean_block.append(interval_means)
        point_actual = np.column_stack(point_block)
        mean_actual = np.column_stack(mean_block)
        pred_matrix = np.column_stack([predictions[(target, step)] for step in steps])

        valid_point = point_actual != 0
        valid_mean = mean_actual != 0
        base_point = pooled_mape(point_actual[valid_point], pred_matrix[valid_point])
        base_mean = pooled_mape(mean_actual[valid_mean], pred_matrix[valid_mean])

        scan = {}
        for c in np.round(np.arange(0.85, 1.001, 0.01), 2):
            scan[float(c)] = pooled_mape(point_actual[valid_point], (c * pred_matrix)[valid_point])
        best_c = min(scan, key=scan.get)

        report[target] = {
            "mape_point_actual": base_point,
            "mape_interval_mean_actual": base_mean,
            "score_1_mape_point": 1.0 - base_point,
            "shrinkage_scan": scan,
            "best_shrinkage_c": best_c,
            "mape_at_best_c": scan[best_c],
        }
        print(f"{target}: point-MAPE {base_point:.4f} | interval-mean-MAPE {base_mean:.4f} | best c {best_c} -> {scan[best_c]:.4f}")

    (root / "outputs/holdout_calibration_semi.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved -> outputs/holdout_calibration_semi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
