from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.analog_trajectory_utils import (
    build_future_trajectory_matrix,
    predict_analog_trajectories,
    robust_standardize,
    select_diverse_neighbors,
)


def test_future_trajectory_uses_official_interval_starts() -> None:
    labels = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01 10:00", periods=4, freq="15min"),
            "generator_1": [10.0, 20.0, 30.0, 40.0],
        }
    )
    matrix = build_future_trajectory_matrix(
        labels, [pd.Timestamp("2025-01-01 10:00")], [15, 30, 45], "generator_1"
    )
    np.testing.assert_allclose(matrix, [[10.0, 20.0, 30.0]])


def test_future_trajectory_preserves_incomplete_scoring_cell() -> None:
    labels = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01 10:00", periods=2, freq="15min"),
            "generator_1": [10.0, np.nan],
        }
    )
    matrix = build_future_trajectory_matrix(
        labels, [pd.Timestamp("2025-01-01 10:00")], [15, 30], "generator_1"
    )
    assert matrix[0, 0] == 10.0
    assert np.isnan(matrix[0, 1])


def test_robust_standardize_fits_library_only_and_imputes() -> None:
    library = np.asarray([[1.0, np.nan], [2.0, 10.0], [3.0, 20.0]])
    query = np.asarray([[4.0, np.nan]])
    lib, qry, fitted = robust_standardize(library, query, scale_floor=1e-6)
    assert np.isfinite(lib).all() and np.isfinite(qry).all()
    np.testing.assert_allclose(fitted["median"], [2.0, 15.0])
    assert qry[0, 1] == pytest.approx(0.0)


def test_neighbor_selection_enforces_day_diversity() -> None:
    distances = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    days = np.asarray(["a", "a", "a", "b", "c"])
    selected = select_diverse_neighbors(
        distances, days, neighbor_count=3, maximum_per_day=1
    )
    assert selected.tolist() == [0, 3, 4]


def test_analog_prediction_is_inverse_distance_weighted() -> None:
    library = np.asarray([[0.0], [2.0], [10.0]])
    query = np.asarray([[1.0]])
    trajectories = np.asarray([[10.0, 20.0], [30.0, 40.0], [100.0, 100.0]])
    days = np.asarray(["a", "b", "c"])
    prediction, diagnostics = predict_analog_trajectories(
        library,
        query,
        trajectories,
        days,
        neighbor_count=2,
        maximum_per_day=1,
        distance_epsilon=1e-6,
    )
    np.testing.assert_allclose(prediction, [[20.0, 30.0]])
    assert diagnostics.loc[0, "unique_source_days"] == 2
