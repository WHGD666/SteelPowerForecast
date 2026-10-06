import numpy as np
import pandas as pd

from src.round2_v3.build_submission_holiday_long_g1_probe import (
    apply_holiday_fuel_correction,
    holiday_target_mask,
)


def _projection() -> dict[str, float | bool]:
    return {
        "generator_1_min": 0.001,
        "generator_1_max": 200.0,
        "generator_all_min": 0.001,
        "generator_all_max": 440.0,
        "enforce_generator_1_lte_generator_all": True,
    }


def test_holiday_mask_uses_target_interval_start_boundaries() -> None:
    origins = pd.DatetimeIndex(["2025-09-30 23:45", "2025-10-08 23:45"])
    mask = holiday_target_mask(
        origins,
        [15, 30],
        pd.Timestamp("2025-10-01 00:00"),
        pd.Timestamp("2025-10-08 23:59:59"),
    )
    assert mask.tolist() == [[False, True], [True, False]]


def test_holiday_correction_freezes_nonholiday_and_generator_all() -> None:
    origins = pd.DatetimeIndex(["2025-09-30 23:45", "2025-10-08 23:45"])
    horizons = [15, 30]
    base = pd.DataFrame(
        {
            "datetime": origins,
            "generator_1_t+15_pred": [100.0, 100.0],
            "generator_1_t+30_pred": [100.0, 100.0],
            "generator_all_t+15_pred": [300.0, 300.0],
            "generator_all_t+30_pred": [300.0, 300.0],
        }
    )
    candidate, report = apply_holiday_fuel_correction(
        base,
        origins,
        np.asarray([50.0, 60.0]),
        horizons,
        scale=0.95,
        interval_start=pd.Timestamp("2025-10-01 00:00"),
        interval_end=pd.Timestamp("2025-10-08 23:59:59"),
        projection=_projection(),
    )
    expected_g1 = np.asarray([[100.0, 98.0], [97.6, 100.0]])
    assert np.allclose(candidate.iloc[:, 1:3].to_numpy(float), expected_g1)
    assert np.array_equal(candidate.iloc[:, 3:5].to_numpy(float), np.full((2, 2), 300.0))
    assert report["holiday_g1_cells"] == 2
    assert report["changed_g1_cells"] == 2
    assert report["nonholiday_long_generator_1_exactly_frozen"] is True
