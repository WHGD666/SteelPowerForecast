from __future__ import annotations

import numpy as np

from src.round2_v3.run_long_gall_shared_horizon import _v20_mix


def _recipe() -> dict[str, float | int]:
    return {
        "near_max_step": 32,
        "near_tree_weight": 0.5,
        "near_fuel_weight": 0.5,
        "far_tree_weight": 0.225,
        "far_climatology_weight": 0.075,
        "far_fuel_weight": 0.7,
        "gate_trigger_absolute_ratio_delta": 0.20,
        "gate_clip_min": 0.80,
        "gate_clip_max": 1.20,
    }


def test_v20_near_far_formula_and_gate() -> None:
    result = _v20_mix(
        tree_prediction=np.array([100.0, 100.0, 100.0]),
        fuel_prediction=np.array([80.0, 80.0, 80.0]),
        climatology_prediction=np.array([60.0, 60.0, 60.0]),
        horizons_minutes=np.array([480, 495, 495]),
        gate_ratio=np.array([1.0, 1.0, 0.7]),
        recipe=_recipe(),
    )
    assert np.isclose(result[0], 90.0)
    assert np.isclose(result[1], 83.0)
    assert np.isclose(result[2], 83.0 * 0.8)


def test_v20_gate_is_strict_at_threshold() -> None:
    result = _v20_mix(
        tree_prediction=np.array([100.0, 100.0]),
        fuel_prediction=np.array([100.0, 100.0]),
        climatology_prediction=np.array([100.0, 100.0]),
        horizons_minutes=np.array([15, 15]),
        gate_ratio=np.array([0.8, 1.21]),
        recipe=_recipe(),
    )
    assert np.isclose(result[0], 100.0)
    assert np.isclose(result[1], 120.0)
