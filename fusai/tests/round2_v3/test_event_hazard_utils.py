from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.event_hazard_utils import (
    onset_within_horizon,
    quantile_alert_threshold,
)


def test_onset_label_is_strict_after_origin_and_inclusive_at_horizon() -> None:
    origins = pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 02:00"]
    )
    starts = pd.to_datetime(["2025-01-01 01:00", "2025-01-01 03:00"])
    result = onset_within_horizon(origins, starts, 60)
    np.testing.assert_array_equal(result, np.array([1, 0, 1], dtype=np.int8))


def test_onset_label_does_not_use_event_at_origin() -> None:
    origins = pd.to_datetime(["2025-01-01 01:00"])
    starts = pd.to_datetime(["2025-01-01 01:00"])
    np.testing.assert_array_equal(
        onset_within_horizon(origins, starts, 120), np.array([0], dtype=np.int8)
    )


def test_quantile_alert_threshold_respects_budget() -> None:
    values = np.arange(100, dtype=float) / 100
    threshold = quantile_alert_threshold(values, 0.10)
    assert threshold == pytest.approx(np.quantile(values, 0.90))
    with pytest.raises(ValueError):
        quantile_alert_threshold(values, 1.0)
