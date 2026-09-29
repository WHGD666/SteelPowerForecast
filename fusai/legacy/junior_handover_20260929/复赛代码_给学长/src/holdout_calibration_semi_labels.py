"""Sealed-holdout comparison: instantaneous-point labels vs 15-minute interval-mean labels.

The official scoring uses 15-minute interval means of the 1-minute actuals.
generator_1 is a step-jumpy signal, so a model trained on instantaneous
15-minute samples fits sample noise rather than the scored quantity.  This
script retrains the same frozen LightGBM design with interval-mean labels and
compares holdout MAPE against the point-label model (point-trained numbers
from holdout_calibration_semi.json: g1 0.0996 / gall 0.0874 on interval-mean
actuals; shrinkage scanned against the same metric here).
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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/submission_semi_composed.yaml").read_text(encoding="utf-8"))
    targets = list(config["targets"])
    steps, minutes = horizons_from_config(config)

    train = pd.read_csv(root / "outputs/prepared_v2/train_prepared.csv", parse_dates=["datetime"])
    # interval-mean labels from the 1-minute source on the same 15-minute grid
    minute_load = pd.read_csv(root / "data/Semi_load.csv", parse_dates=["datetime"]).set_index("datetime")
    label_15 = minute_load[targets].resample("15min").mean()
    train = train.set_index("datetime")
    for target in targets:
        train[f"ilabel_{target}"] = label_15[target].reindex(train.index)
    train = train.reset_index()

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
    report: dict = {"label_mode": "interval_mean_15min"}
    for target in targets:
        series = train[f"ilabel_{target}"]
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
        pred_matrix = np.column_stack(preds)
        actual_matrix = np.column_stack(actuals)
        valid = actual_matrix != 0
        base = pooled_mape(actual_matrix[valid], pred_matrix[valid])
        scan = {}
        for c in np.round(np.arange(0.88, 1.001, 0.01), 2):
            scan[float(c)] = pooled_mape(actual_matrix[valid], (c * pred_matrix)[valid])
        best_c = min(scan, key=scan.get)
        report[target] = {"mape_interval_mean": base, "score_1_mape": 1.0 - base, "best_c": best_c, "mape_at_best_c": scan[best_c]}
        print(f"{target}: interval-mean MAPE {base:.4f} (1-MAPE {1-base:.4f}) | best c {best_c} -> {scan[best_c]:.4f}", flush=True)

    (root / "outputs/holdout_calibration_semi_labels.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved -> outputs/holdout_calibration_semi_labels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
