from __future__ import annotations

import numpy as np
import pandas as pd

from src.round2_v3.build_submission_dynamic_fuel_probe import (
    replace_long_generator_1,
)


def test_replace_long_g1_freezes_all_other_blocks() -> None:
    horizons = [15, 30]
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-10-01", periods=2, freq="15min"),
            "generator_1_t+15_pred": [100.0, 110.0],
            "generator_1_t+30_pred": [101.0, 111.0],
            "generator_all_t+15_pred": [200.0, 210.0],
            "generator_all_t+30_pred": [201.0, 211.0],
        }
    )
    baseline = frame.copy()
    baseline[["generator_1_t+15_pred", "generator_1_t+30_pred"]] = 80.0
    dynamic = np.asarray([[120.0, 130.0], [140.0, 150.0]])
    projection = {
        "generator_1_min": 0.001,
        "generator_1_max": 200.0,
        "generator_all_min": 0.001,
        "generator_all_max": 440.0,
        "enforce_generator_1_lte_generator_all": True,
    }
    result, report = replace_long_generator_1(
        frame, baseline, dynamic, horizons, 0.7, 0.3, projection
    )
    expected = 0.7 * dynamic + 0.3 * 80.0
    np.testing.assert_allclose(
        result[["generator_1_t+15_pred", "generator_1_t+30_pred"]], expected
    )
    pd.testing.assert_frame_equal(
        result[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
        frame[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
    )
    assert report["generator_all_exactly_frozen"] is True

