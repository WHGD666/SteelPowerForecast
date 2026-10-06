import numpy as np
import pandas as pd

from src.round2_v3.run_causal_overlap_reconciliation import causal_overlap_reconcile


def _frame() -> pd.DataFrame:
    rows = []
    origins = pd.date_range("2025-09-01", periods=3, freq="15min")
    values = [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]
    for origin, predictions in zip(origins, values, strict=True):
        for horizon, prediction in zip([15, 30, 45], predictions, strict=True):
            rows.append({"fold_id": "fold", "datetime": origin, "horizon_minutes": horizon, "prediction": prediction})
    return pd.DataFrame(rows)


def test_causal_overlap_reconcile_uses_previous_output_for_same_target() -> None:
    result = causal_overlap_reconcile(_frame(), alpha_current=0.5, horizons=[15, 30, 45], frequency_minutes=15)
    pivot = result.pivot(index="datetime", columns="horizon_minutes", values="prediction").sort_index()
    np.testing.assert_allclose(pivot.iloc[0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(pivot.iloc[1], [15.5, 25.5, 31.0])
    np.testing.assert_allclose(pivot.iloc[2], [18.75, 26.5, 32.0])


def test_causal_overlap_reconcile_resets_on_time_gap() -> None:
    frame = _frame()
    frame.loc[frame["datetime"] == frame["datetime"].max(), "datetime"] += pd.Timedelta(minutes=15)
    result = causal_overlap_reconcile(frame, alpha_current=0.5, horizons=[15, 30, 45], frequency_minutes=15)
    pivot = result.pivot(index="datetime", columns="horizon_minutes", values="prediction").sort_index()
    np.testing.assert_allclose(pivot.iloc[-1], [12.0, 22.0, 32.0])

