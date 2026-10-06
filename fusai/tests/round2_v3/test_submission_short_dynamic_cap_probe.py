from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.build_submission_short_dynamic_cap_probe import (
    replace_short_generator_1_with_capped_dynamic,
)


PROJECTION = {
    "generator_1_min": 0.001,
    "generator_1_max": 200.0,
    "generator_all_min": 0.001,
    "generator_all_max": 440.0,
    "enforce_generator_1_lte_generator_all": True,
}


def _base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-10-01", periods=2, freq="15min"),
            "generator_1_t+15_pred": [100.0, 100.0],
            "generator_1_t+30_pred": [100.0, 100.0],
            "generator_all_t+15_pred": [300.0, 300.0],
            "generator_all_t+30_pred": [300.0, 300.0],
        }
    )


def test_capped_dynamic_changes_only_short_generator_1() -> None:
    base = _base()
    dynamic = np.asarray([[110.0, 80.0], [104.0, 98.0]])
    result, report = replace_short_generator_1_with_capped_dynamic(
        base,
        dynamic,
        [15, 30],
        dynamic_weight=0.70,
        correction_cap_mw=5.0,
        projection=PROJECTION,
    )
    np.testing.assert_allclose(
        result[["generator_1_t+15_pred", "generator_1_t+30_pred"]],
        np.asarray([[105.0, 95.0], [102.8, 98.6]]),
    )
    pd.testing.assert_frame_equal(
        result[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
        base[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
    )
    assert report["source_cells_exceeding_cap"] == 2
    assert report["final_max_abs_correction_mw"] == pytest.approx(5.0)
    assert report["generator_all_exactly_frozen"] is True


def test_capped_dynamic_rejects_wrong_shape() -> None:
    with pytest.raises(AssertionError, match="dynamic matrix shape"):
        replace_short_generator_1_with_capped_dynamic(
            _base(),
            np.ones((2, 1)),
            [15, 30],
            dynamic_weight=0.70,
            correction_cap_mw=5.0,
            projection=PROJECTION,
        )
