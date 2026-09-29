from __future__ import annotations

import numpy as np

from src.round2_v3.run_hierarchical import derive_small_unit_prediction


PROJECTION = {
    "generator_1_min": 0.001,
    "generator_1_max": 200.0,
    "large_units_min": 0.0,
    "require_generator_1_lte_generator_all": True,
}


def test_raw_hierarchical_reconstruction() -> None:
    result = derive_small_unit_prediction(
        np.array([300.0, 250.0]), np.array([220.0, 270.0]), False, PROJECTION
    )
    assert result.tolist() == [80.0, -20.0]


def test_constrained_reconstruction_enforces_physical_bounds() -> None:
    result = derive_small_unit_prediction(
        np.array([300.0, 150.0, 180.0]),
        np.array([220.0, 170.0, -10.0]),
        True,
        PROJECTION,
    )
    assert result.tolist() == [80.0, 0.001, 180.0]
    assert np.all(result > 0)
    assert np.all(result <= np.array([300.0, 150.0, 180.0]))
    assert np.all(result <= 200.0)
