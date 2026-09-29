from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.metrics import evaluate_wide_targets, regression_metrics


def test_regression_metrics_matches_hand_calculation() -> None:
    result = regression_metrics([100.0, 200.0], [110.0, 180.0])
    assert result["mape"] == pytest.approx(0.1)
    assert result["accuracy_1_minus_mape"] == pytest.approx(0.9)
    assert result["mae"] == pytest.approx(15.0)
    assert result["rmse"] == pytest.approx(np.sqrt(250.0))
    assert result["valid_count"] == 2


def test_zero_and_missing_actuals_are_excluded_and_counted() -> None:
    result = regression_metrics([0.0, np.nan, 100.0], [12.0, 20.0, 90.0])
    assert result["mape"] == pytest.approx(0.1)
    assert result["valid_count"] == 1
    assert result["zero_actual_count"] == 1
    assert result["missing_actual_count"] == 1


def test_wide_metrics_pool_cells_instead_of_averaging_group_mapes() -> None:
    actual = pd.DataFrame(
        {
            "generator_1_t+15": [100.0, 100.0],
            "generator_1_t+30": [200.0, 200.0],
            "generator_all_t+15": [300.0, 300.0],
            "generator_all_t+30": [400.0, 400.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "generator_1_t+15_pred": [110.0, 90.0],
            "generator_1_t+30_pred": [220.0, 180.0],
            "generator_all_t+15_pred": [330.0, 270.0],
            "generator_all_t+30_pred": [440.0, 360.0],
        }
    )
    report = evaluate_wide_targets(actual, predicted, [15, 30])
    overall = report.loc[report["scope"] == "overall"].iloc[0]
    assert overall["mape"] == pytest.approx(0.1)
    assert overall["valid_count"] == 8
    assert len(report) == 7


def test_metric_rejects_shape_mismatch_and_empty_valid_set() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        regression_metrics([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="no finite non-zero"):
        regression_metrics([0.0, np.nan], [1.0, 2.0])
