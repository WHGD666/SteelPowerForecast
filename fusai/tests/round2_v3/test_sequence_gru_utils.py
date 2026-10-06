from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.round2_v3.sequence_gru_utils import (
    DirectMultiStepGRU,
    add_step_calendar,
    build_sequence_windows,
    fit_log_target_normalization,
    fit_transform_time_features,
    inverse_log_target_normalization,
)


def test_direct_gru_output_shape_and_unidirectional_contract() -> None:
    model = DirectMultiStepGRU(
        input_size=5,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
        output_size=4,
        layer_norm=True,
    )
    output = model(torch.zeros((3, 6, 5), dtype=torch.float32))
    assert output.shape == (3, 4)
    assert model.gru.bidirectional is False


def test_sequence_windows_end_at_origin_inclusive() -> None:
    grid = np.arange(20, dtype=np.float32).reshape(10, 2)
    windows = build_sequence_windows(grid, np.asarray([3, 5]), window_steps=3)
    np.testing.assert_allclose(windows[0], grid[1:4])
    np.testing.assert_allclose(windows[1], grid[3:6])


def test_sequence_windows_reject_insufficient_history() -> None:
    with pytest.raises(ValueError, match="complete history"):
        build_sequence_windows(np.ones((5, 2)), np.asarray([1]), window_steps=3)


def test_time_feature_scaler_fits_only_training_rows() -> None:
    values = np.asarray([[1.0, np.nan], [2.0, 10.0], [3.0, 20.0], [1000.0, 1000.0]])
    transformed, fitted = fit_transform_time_features(
        values, np.asarray([True, True, True, False]), scale_floor=1e-6
    )
    np.testing.assert_allclose(fitted["median"], [2.0, 15.0])
    assert np.isfinite(transformed).all()
    assert transformed[-1, 0] > 100.0


def test_log_target_normalization_round_trip() -> None:
    targets = np.asarray([[80.0, 100.0], [120.0, 150.0], [100.0, 125.0]])
    standardized, fitted = fit_log_target_normalization(targets)
    restored = inverse_log_target_normalization(
        standardized, fitted, prediction_min=0.001, prediction_max=200.0
    )
    np.testing.assert_allclose(restored, targets, rtol=1e-6)


def test_step_calendar_adds_four_causal_columns() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    times = pd.DatetimeIndex(["2025-01-06 00:00", "2025-01-06 06:00"])
    values, names = add_step_calendar(
        frame,
        times,
        {"add_step_hour_cyclic": True, "add_step_dayofweek_cyclic": True},
    )
    assert values.shape == (2, 5)
    assert names[-4:] == [
        "step_hour_sin",
        "step_hour_cos",
        "step_dayofweek_sin",
        "step_dayofweek_cos",
    ]
