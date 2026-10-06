import numpy as np
import pytest

from src.round2_v3.october_shift_utils import column_shift_metrics


def test_stable_column_shows_no_shift() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(100.0, 10.0, 20000)
    metrics = column_shift_metrics(train, train[:5000], train[:2000])
    assert abs(metrics["median_shift_in_reference_iqr"]) < 0.15
    assert metrics["out_of_support_rate"] <= 0.011
    assert metrics["hard_out_of_range_rate"] == 0.0


def test_upward_shift_is_localized_above_support() -> None:
    rng = np.random.default_rng(11)
    train = rng.normal(100.0, 10.0, 20000)
    test = rng.normal(130.0, 10.0, 5000)
    metrics = column_shift_metrics(train, train[:5000], test)
    assert metrics["median_shift_in_reference_iqr"] > 1.0
    assert metrics["out_of_support_above_rate"] > 0.5
    assert metrics["out_of_support_below_rate"] == 0.0
    assert metrics["hard_above_rate"] > 0.0


def test_nan_handling_and_missing_rate() -> None:
    train = np.linspace(0.0, 100.0, 1000)
    reference = train[:250]
    test = np.concatenate([[np.nan] * 100, np.linspace(10.0, 90.0, 900)])
    metrics = column_shift_metrics(train, reference, test)
    assert np.isclose(metrics["test_missing_rate"], 100.0 / 1000.0)
    assert metrics["out_of_support_rate"] == 0.0
    assert np.isclose(metrics["test_median"], 50.0)


def test_degenerate_reference_iqr_returns_nan_shift() -> None:
    train = np.linspace(0.0, 100.0, 1000)
    reference = np.full(500, 50.0)
    test = np.linspace(0.0, 100.0, 100)
    metrics = column_shift_metrics(train, reference, test)
    assert np.isnan(metrics["median_shift_in_reference_iqr"])
    assert metrics["reference_iqr"] == 0.0


def test_all_nan_test_column_is_reported_missing() -> None:
    train = np.linspace(0.0, 100.0, 1000)
    metrics = column_shift_metrics(train, train[:250], np.full(50, np.nan))
    assert metrics["test_missing_rate"] == 1.0
    assert np.isnan(metrics["test_median"])
    assert metrics["train_n"] == 1000.0
