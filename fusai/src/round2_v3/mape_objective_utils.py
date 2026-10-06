"""Utilities for aligning weighted absolute-error training with MAPE."""

from __future__ import annotations

import numpy as np


def normalized_mape_weights(
    target: np.ndarray,
    *,
    denominator_floor: float,
) -> np.ndarray:
    """Return mean-one weights proportional to 1 / max(abs(y), floor)."""
    values = np.asarray(target, dtype=float)
    floor = float(denominator_floor)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("target must be a finite one-dimensional array")
    if floor <= 0:
        raise ValueError("denominator_floor must be positive")
    weights = 1.0 / np.maximum(np.abs(values), floor)
    return weights / weights.mean()

