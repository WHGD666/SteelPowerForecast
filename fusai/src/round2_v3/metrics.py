"""Frozen local metrics for round2 v3 validation.

The competition's hidden score mapping is not reproduced here.  These
functions implement the documented local evidence: pooled MAPE, 1-MAPE and
standard error diagnostics, with zero actuals explicitly excluded and counted.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.round2_v3.contracts import TARGETS


def regression_metrics(actual: object, predicted: object) -> dict[str, float | int]:
    """Compute metrics on finite, non-zero actual/prediction pairs."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted shapes differ")

    finite_actual = np.isfinite(actual_values)
    finite_prediction = np.isfinite(predicted_values)
    zero_actual = finite_actual & (actual_values == 0)
    valid = finite_actual & finite_prediction & ~zero_actual
    if not valid.any():
        raise ValueError("no finite non-zero actual/prediction pairs")

    errors = predicted_values[valid] - actual_values[valid]
    absolute_percentage_errors = np.abs(errors / actual_values[valid])
    mape = float(absolute_percentage_errors.mean())
    return {
        "mape": mape,
        "accuracy_1_minus_mape": 1.0 - mape,
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "valid_count": int(valid.sum()),
        "zero_actual_count": int(zero_actual.sum()),
        "missing_actual_count": int((~finite_actual).sum()),
        "missing_prediction_count": int((~finite_prediction).sum()),
    }


def evaluate_wide_targets(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    horizons_minutes: Iterable[int],
    targets: Iterable[str] = TARGETS,
) -> pd.DataFrame:
    """Return pooled, per-target and per-target/horizon metric rows.

    Actual columns use ``generator_1_t+15`` and prediction columns use
    ``generator_1_t+15_pred``.  Aggregation is pooled over cells, not an
    unweighted average of already-aggregated MAPE values.
    """
    horizons = tuple(int(value) for value in horizons_minutes)
    target_names = tuple(targets)
    cells: list[dict[str, object]] = []
    for target in target_names:
        for horizon in horizons:
            actual_column = f"{target}_t+{horizon}"
            prediction_column = f"{target}_t+{horizon}_pred"
            if actual_column not in actual or prediction_column not in predicted:
                raise KeyError(f"missing metric columns: {actual_column}/{prediction_column}")
            for actual_value, prediction_value in zip(
                actual[actual_column].to_numpy(),
                predicted[prediction_column].to_numpy(),
                strict=True,
            ):
                cells.append(
                    {
                        "target": target,
                        "horizon_minutes": horizon,
                        "actual": actual_value,
                        "predicted": prediction_value,
                    }
                )
    long = pd.DataFrame.from_records(cells)
    rows: list[dict[str, object]] = []

    def append_metrics(scope: str, target: str, horizon: int | str, part: pd.DataFrame) -> None:
        rows.append(
            {
                "scope": scope,
                "target": target,
                "horizon_minutes": horizon,
                **regression_metrics(part["actual"], part["predicted"]),
            }
        )

    append_metrics("overall", "all", "all", long)
    for target, part in long.groupby("target", sort=False):
        append_metrics("target", str(target), "all", part)
    for (target, horizon), part in long.groupby(
        ["target", "horizon_minutes"], sort=False
    ):
        append_metrics("target_horizon", str(target), int(horizon), part)
    return pd.DataFrame.from_records(rows)
