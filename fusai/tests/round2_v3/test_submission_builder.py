from __future__ import annotations

import numpy as np
import pandas as pd

from src.round2_v3.build_submission_baseline import (
    apply_physical_projection,
    predictions_to_wide,
)
from src.round2_v3.verify_submission import validate_submission_status


def test_prediction_reshape_preserves_origin_major_order() -> None:
    origins = pd.date_range("2025-10-01", periods=2, freq="15min")
    horizons = [15, 30]
    frame = predictions_to_wide(
        origins,
        horizons,
        {
            "generator_1": np.array([1.0, 2.0, 3.0, 4.0]),
            "generator_all": np.array([10.0, 20.0, 30.0, 40.0]),
        },
    )
    assert frame["generator_1_t+15_pred"].tolist() == [1.0, 3.0]
    assert frame["generator_1_t+30_pred"].tolist() == [2.0, 4.0]
    assert frame["generator_all_t+15_pred"].tolist() == [10.0, 30.0]


def test_physical_projection_enforces_submission_bounds() -> None:
    frame = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2025-10-01")],
            "generator_1_t+15_pred": [250.0],
            "generator_all_t+15_pred": [180.0],
        }
    )
    projected, before = apply_physical_projection(
        frame,
        [15],
        {
            "generator_1_min": 0.001,
            "generator_1_max": 200.0,
            "generator_all_min": 0.001,
            "generator_all_max": 440.0,
            "enforce_generator_1_lte_generator_all": True,
        },
    )
    assert before["generator_1_above_generator_all"] == 1
    assert projected.loc[0, "generator_1_t+15_pred"] == 180.0


def test_submission_verifier_accepts_pre_and_post_upload_states() -> None:
    validate_submission_status("ready_for_upload")
    validate_submission_status("submitted")
