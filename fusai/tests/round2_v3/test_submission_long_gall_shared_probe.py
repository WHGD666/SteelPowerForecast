import numpy as np
import pandas as pd

from src.round2_v3.build_submission_long_gall_shared_probe import replace_long_generator_all


def test_replace_long_generator_all_freezes_g1_and_projects_gall_upward() -> None:
    base = pd.DataFrame(
        {
            "datetime": ["2025-10-01 00:00:00", "2025-10-01 00:15:00"],
            "generator_1_t+15_pred": [100.0, 110.0],
            "generator_all_t+15_pred": [200.0, 210.0],
            "generator_1_t+30_pred": [120.0, 130.0],
            "generator_all_t+30_pred": [220.0, 230.0],
        }
    )
    raw = np.array([[90.0, 500.0], [205.0, 125.0]])
    result, report = replace_long_generator_all(
        base,
        raw,
        [15, 30],
        {
            "generator_all_min": 0.001,
            "generator_all_max": 440.0,
            "enforce_generator_1_lte_generator_all": True,
        },
    )
    np.testing.assert_array_equal(
        result[["generator_1_t+15_pred", "generator_1_t+30_pred"]],
        base[["generator_1_t+15_pred", "generator_1_t+30_pred"]],
    )
    np.testing.assert_allclose(
        result[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
        [[100.0, 440.0], [205.0, 130.0]],
    )
    assert report["long_generator_1_exactly_frozen"] is True
    assert report["raw_above_440_cells"] == 1
    assert report["projected_up_to_frozen_g1_cells"] == 2


def test_replace_long_generator_all_rejects_wrong_shape() -> None:
    base = pd.DataFrame(
        {
            "datetime": ["2025-10-01 00:00:00"],
            "generator_1_t+15_pred": [100.0],
            "generator_all_t+15_pred": [200.0],
        }
    )
    try:
        replace_long_generator_all(
            base,
            np.array([200.0]),
            [15],
            {"generator_all_min": 0.001, "generator_all_max": 440.0, "enforce_generator_1_lte_generator_all": True},
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("wrong-shaped generator_all matrix was accepted")

