from __future__ import annotations

import numpy as np
import pandas as pd

from src.round2_v3.run_long_g1_shared_horizon import (
    _align_calendar_to_point_target,
)


def test_calendar_alignment_uses_point_target_not_interval_start() -> None:
    features = pd.DataFrame(
        {
            "horizon_minutes": [15.0],
            "horizon_step": [1.0],
            "target_hour_sin": [0.0],
            "target_hour_cos": [1.0],
            "target_minute_sin": [0.0],
            "target_minute_cos": [1.0],
            "target_dayofweek_sin": [0.0],
            "target_dayofweek_cos": [1.0],
            "target_is_weekend": [0.0],
            "target_month": [1.0],
            "target_day_of_month": [1.0],
        }
    )
    meta = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2025-09-01 23:45:00")],
            "interval_start": [pd.Timestamp("2025-09-01 23:45:00")],
            "horizon_minutes": [15],
        }
    )
    result = _align_calendar_to_point_target(features, meta)
    assert result.loc[0, "target_month"] == 9
    assert result.loc[0, "target_day_of_month"] == 2
    assert result.loc[0, "target_is_weekend"] == 0
    assert np.isclose(result.loc[0, "target_hour_sin"], 0.0, atol=1e-7)
    assert np.isclose(result.loc[0, "target_hour_cos"], 1.0, atol=1e-7)
