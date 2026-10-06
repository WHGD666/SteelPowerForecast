import numpy as np
import pandas as pd

from src.round2_v3.build_submission_long_g1_shared_probe import replace_long_generator_1


def test_replace_long_generator_1_freezes_gall_and_projects_g1() -> None:
    base = pd.DataFrame(
        {
            "datetime": ["2025-10-01 00:00:00", "2025-10-01 00:15:00"],
            "generator_1_t+15_pred": [100.0, 110.0],
            "generator_all_t+15_pred": [200.0, 210.0],
            "generator_1_t+30_pred": [120.0, 130.0],
            "generator_all_t+30_pred": [220.0, 230.0],
        }
    )
    raw = np.array([[-5.0, 500.0], [205.0, 240.0]])
    result, report = replace_long_generator_1(
        base,
        raw,
        [15, 30],
        {"generator_1_min": 0.001, "generator_1_max": 200.0, "enforce_generator_1_lte_generator_all": True},
    )
    np.testing.assert_array_equal(
        result[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
        base[["generator_all_t+15_pred", "generator_all_t+30_pred"]],
    )
    np.testing.assert_allclose(
        result[["generator_1_t+15_pred", "generator_1_t+30_pred"]],
        [[0.001, 200.0], [200.0, 200.0]],
    )
    assert report["long_generator_all_exactly_frozen"] is True
    assert report["raw_nonpositive_cells"] == 1
    assert report["raw_above_200_cells"] == 3


def test_replace_long_generator_1_rejects_wrong_shape() -> None:
    base = pd.DataFrame({"datetime": ["2025-10-01"], "generator_1_t+15_pred": [100.0], "generator_all_t+15_pred": [200.0]})
    try:
        replace_long_generator_1(base, np.array([100.0]), [15], {"generator_1_min": 0.001, "generator_1_max": 200.0, "enforce_generator_1_lte_generator_all": True})
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("wrong-shaped generator_1 matrix was accepted")

