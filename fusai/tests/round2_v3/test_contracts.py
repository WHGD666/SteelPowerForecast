from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.contracts import (
    LONG_HORIZONS_MINUTES,
    SHORT_HORIZONS_MINUTES,
    build_interval_target_matrix,
    expected_result_columns,
    interval_bounds,
    interval_mean_series,
    validate_prediction_frame,
)


def synthetic_load(periods: int = 1500) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="1min")
    values = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "datetime": timestamps,
            "generator_1": values,
            "generator_all": values + 200.0,
        }
    )


@pytest.mark.parametrize(
    ("horizon", "start", "end"),
    [
        (15, "2025-01-01 10:00:00", "2025-01-01 10:14:00"),
        (30, "2025-01-01 10:15:00", "2025-01-01 10:29:00"),
        (120, "2025-01-01 11:45:00", "2025-01-01 11:59:00"),
        (1440, "2025-01-02 09:45:00", "2025-01-02 09:59:00"),
    ],
)
def test_interval_bounds(horizon: int, start: str, end: str) -> None:
    bounds = interval_bounds("2025-01-01 10:00:00", horizon)
    assert bounds.start == pd.Timestamp(start)
    assert bounds.end == pd.Timestamp(end)


def test_t_plus_15_uses_origin_through_origin_plus_14() -> None:
    frame = synthetic_load()
    origins = [pd.Timestamp("2025-01-01 10:00:00")]
    matrix = build_interval_target_matrix(frame, origins, [15, 30])
    assert matrix.loc[0, "generator_1_t+15"] == pytest.approx(np.mean(np.arange(600, 615)))
    assert matrix.loc[0, "generator_1_t+30"] == pytest.approx(np.mean(np.arange(615, 630)))


def test_incomplete_interval_is_missing_not_silently_filled() -> None:
    frame = synthetic_load(60).drop(index=7).reset_index(drop=True)
    aggregated = interval_mean_series(frame, "generator_1")
    assert pd.isna(aggregated.loc[pd.Timestamp("2025-01-01 00:00:00")])
    assert aggregated.loc[pd.Timestamp("2025-01-01 00:15:00")] == pytest.approx(np.mean(np.arange(15, 30)))


def test_result_schema_sizes_and_order() -> None:
    short = expected_result_columns(SHORT_HORIZONS_MINUTES)
    long = expected_result_columns(LONG_HORIZONS_MINUTES)
    assert len(short) == 17
    assert len(long) == 193
    assert short[1] == "generator_1_t+15_pred"
    assert short[8] == "generator_1_t+120_pred"
    assert short[9] == "generator_all_t+15_pred"
    assert long[-1] == "generator_all_t+1440_pred"


def test_prediction_contract_accepts_consistent_physical_values() -> None:
    origins = pd.date_range("2025-10-01", periods=3, freq="15min")
    horizons = [15, 30]
    frame = pd.DataFrame({"datetime": origins})
    for horizon in horizons:
        frame[f"generator_1_t+{horizon}_pred"] = 120.0
    for horizon in horizons:
        frame[f"generator_all_t+{horizon}_pred"] = 330.0
    result = validate_prediction_frame(frame, origins, horizons)
    assert result["rows"] == 3
    assert result["columns"] == 5


def test_prediction_contract_rejects_cross_target_violation() -> None:
    origins = pd.date_range("2025-10-01", periods=1, freq="15min")
    frame = pd.DataFrame(
        {
            "datetime": origins,
            "generator_1_t+15_pred": [180.0],
            "generator_all_t+15_pred": [170.0],
        }
    )
    with pytest.raises(AssertionError, match="physical contract violations"):
        validate_prediction_frame(frame, origins, [15])
