"""Read-only integrity and error diagnostics for a completed baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io_utils import sha256
from src.round2_v3.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "outputs" / "round2_v3" / "baseline_runs"


def horizon_band(period: str, horizon: int) -> str:
    if period == "short":
        if horizon <= 30:
            return "short_15_30"
        if horizon <= 60:
            return "short_45_60"
        return "short_75_120"
    if horizon <= 120:
        return "long_0_2h"
    if horizon <= 360:
        return "long_2_6h"
    if horizon <= 720:
        return "long_6_12h"
    return "long_12_24h"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = RUN_ROOT / args.run_id
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise AssertionError("diagnostics require a completed run")
    oof_path = PROJECT_ROOT / manifest["artifacts"]["oof_predictions"]
    metrics_path = PROJECT_ROOT / manifest["artifacts"]["metrics"]
    if sha256(oof_path) != manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("OOF hash mismatch")
    if sha256(metrics_path) != manifest["artifacts"]["metrics_sha256"]:
        raise AssertionError("metric hash mismatch")

    oof = pd.read_csv(oof_path, parse_dates=["datetime", "interval_start"])
    key = ["model", "period", "fold_id", "datetime", "target", "horizon_minutes"]
    if oof[key].duplicated().any():
        raise AssertionError("duplicate OOF prediction keys")
    if not np.isfinite(oof[["actual", "prediction"]].to_numpy(float)).all():
        raise AssertionError("OOF contains missing or non-finite values")
    expected_models = {"calendar_profile", "lightgbm_shared_horizon"}
    if set(oof["model"]) != expected_models:
        raise AssertionError("unexpected OOF model set")

    lgbm = oof.loc[oof["model"] == "lightgbm_shared_horizon"].copy()
    lgbm["horizon_band"] = [
        horizon_band(period, int(horizon))
        for period, horizon in zip(lgbm["period"], lgbm["horizon_minutes"], strict=True)
    ]

    group_records = []
    for (period, fold_id, target), part in lgbm.groupby(
        ["period", "fold_id", "target"], sort=False
    ):
        metrics = regression_metrics(part["actual"], part["prediction"])
        group_records.append(
            {
                "period": period,
                "fold_id": fold_id,
                "target": target,
                "accuracy": metrics["accuracy_1_minus_mape"],
                "mape": metrics["mape"],
                "signed_bias": float((part["prediction"] - part["actual"]).mean()),
                "prediction_to_actual_mean_ratio": float(
                    part["prediction"].mean() / part["actual"].mean()
                ),
            }
        )
    group_frame = pd.DataFrame(group_records)

    band_records = []
    for (period, target, band), part in lgbm.groupby(
        ["period", "target", "horizon_band"], sort=False
    ):
        metrics = regression_metrics(part["actual"], part["prediction"])
        band_records.append(
            {
                "period": period,
                "target": target,
                "band": band,
                "accuracy": metrics["accuracy_1_minus_mape"],
                "mae": metrics["mae"],
            }
        )
    band_frame = pd.DataFrame(band_records)

    physical = lgbm.pivot(
        index=["period", "fold_id", "datetime", "horizon_minutes"],
        columns="target",
        values="prediction",
    )
    violations = {
        "generator_1_nonpositive": int((physical["generator_1"] <= 0).sum()),
        "generator_all_nonpositive": int((physical["generator_all"] <= 0).sum()),
        "generator_1_above_200": int((physical["generator_1"] > 200).sum()),
        "generator_all_above_440": int((physical["generator_all"] > 440).sum()),
        "generator_1_above_generator_all": int(
            (physical["generator_1"] > physical["generator_all"]).sum()
        ),
    }

    paired = oof.pivot(
        index=["period", "fold_id", "datetime", "target", "horizon_minutes", "actual"],
        columns="model",
        values="prediction",
    ).reset_index()
    oracle_blend = []
    for (period, target), part in paired.groupby(["period", "target"], sort=False):
        candidates = []
        for weight in np.linspace(0.0, 1.0, 21):
            prediction = (
                weight * part["lightgbm_shared_horizon"]
                + (1.0 - weight) * part["calendar_profile"]
            )
            mape = float(np.mean(np.abs((prediction - part["actual"]) / part["actual"])))
            candidates.append((mape, float(weight)))
        best_mape, best_weight = min(candidates)
        oracle_blend.append(
            {
                "period": period,
                "target": target,
                "best_lightgbm_weight": best_weight,
                "same_oof_mape": best_mape,
                "same_oof_accuracy": 1.0 - best_mape,
                "role": "diagnostic_only_not_cross_fitted",
            }
        )

    print(
        f"PASS run_id={args.run_id} oof_rows={len(oof)} "
        f"lgbm_rows={len(lgbm)} hashes=true"
    )
    print("\nFOLD_TARGET")
    print(group_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nHORIZON_BANDS")
    print(band_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nPHYSICAL_VIOLATIONS")
    print(json.dumps(violations, ensure_ascii=False, sort_keys=True))
    print("\nORACLE_BLEND_DIAGNOSTIC")
    print(
        pd.DataFrame(oracle_blend).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )


if __name__ == "__main__":
    main()
