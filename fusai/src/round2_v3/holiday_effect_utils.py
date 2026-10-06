"""Utilities for the two-block holiday fuel-residual transfer diagnostic."""

from __future__ import annotations

import numpy as np


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError("weighted median has no valid positive-weight rows")
    order = np.argsort(values[valid], kind="mergesort")
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    cutoff = 0.5 * sorted_weights.sum()
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def mape_optimal_scale(actual: np.ndarray, base_prediction: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(base_prediction, dtype=float)
    valid = (
        np.isfinite(actual)
        & np.isfinite(prediction)
        & (np.abs(actual) > 1e-9)
        & (prediction > 1e-9)
    )
    ratio = actual[valid] / prediction[valid]
    weights = prediction[valid] / np.abs(actual[valid])
    return weighted_median(ratio, weights)


def mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (np.abs(actual) > 1e-9)
    if not valid.any():
        raise ValueError("MAPE has no valid rows")
    return float(np.mean(np.abs(prediction[valid] - actual[valid]) / np.abs(actual[valid])))

