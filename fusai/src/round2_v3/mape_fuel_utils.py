"""MAPE-aligned linear fuel proxy helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.round2_v3.mape_objective_utils import normalized_mape_weights


def fit_mape_median_fuel_proxy(
    frame: pd.DataFrame,
    *,
    target: str,
    fuel_columns: Iterable[str],
    denominator_floor: float,
) -> Pipeline:
    """Fit inverse-target weighted median regression on finite training rows."""
    columns = list(fuel_columns)
    matrix = frame[columns].to_numpy(dtype=float)
    values = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(matrix).all(axis=1)
    if valid.sum() <= len(columns) + 1:
        raise ValueError("not enough finite rows for MAPE median fuel proxy")
    weights = normalized_mape_weights(
        values[valid], denominator_floor=denominator_floor
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "regressor",
                QuantileRegressor(
                    quantile=0.5,
                    alpha=0.0,
                    fit_intercept=True,
                    solver="highs",
                ),
            ),
        ]
    )
    model.fit(matrix[valid], values[valid], regressor__sample_weight=weights)
    return model


def predict_mape_median_fuel_proxy(
    model: Pipeline,
    frame: pd.DataFrame,
    *,
    fuel_columns: Iterable[str],
) -> np.ndarray:
    matrix = frame[list(fuel_columns)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("fuel prediction matrix contains non-finite values")
    return np.clip(model.predict(matrix), 0.0, None)

