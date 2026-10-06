from __future__ import annotations

import numpy as np

from src.round2_v3.holiday_effect_utils import mape, mape_optimal_scale


def test_mape_optimal_scale_recovers_common_ratio() -> None:
    base = np.array([50.0, 100.0, 150.0, 200.0])
    actual = 0.9 * base
    scale = mape_optimal_scale(actual, base)
    assert scale == 0.9
    assert mape(actual, scale * base) < 1e-12

