from __future__ import annotations

import numpy as np
import pytest

from src.round2_v3.mape_objective_utils import normalized_mape_weights


def test_normalized_mape_weights_match_mape_denominator() -> None:
    target = np.array([50.0, 100.0, 200.0])
    weights = normalized_mape_weights(target, denominator_floor=1.0)
    np.testing.assert_allclose(weights.mean(), 1.0)
    np.testing.assert_allclose(weights[0] / weights[1], 2.0)
    np.testing.assert_allclose(weights[1] / weights[2], 2.0)


def test_normalized_mape_weights_apply_floor_and_validate() -> None:
    weights = normalized_mape_weights(
        np.array([0.0, 5.0, 10.0]), denominator_floor=5.0
    )
    np.testing.assert_allclose(weights[0], weights[1])
    with pytest.raises(ValueError):
        normalized_mape_weights(np.array([1.0]), denominator_floor=0.0)
    with pytest.raises(ValueError):
        normalized_mape_weights(np.array([np.nan]), denominator_floor=1.0)

