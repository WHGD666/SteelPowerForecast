"""Pure contract functions for labels, origins and submission schemas.

This module contains no model code.  Its purpose is to make the official time
semantics executable and independently testable before any experiment starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TARGETS = ("generator_1", "generator_all")
SHORT_HORIZONS_MINUTES = tuple(range(15, 121, 15))
LONG_HORIZONS_MINUTES = tuple(range(15, 1441, 15))


@dataclass(frozen=True)
class IntervalBounds:
    """Inclusive minute-level bounds for one forecast target interval."""

    start: pd.Timestamp
    end: pd.Timestamp


def interval_bounds(origin: pd.Timestamp | str, horizon_minutes: int) -> IntervalBounds:
    """Return the official 15-minute mean interval for one origin/horizon.

    For example, origin 10:00 and horizon 15 maps to 10:00..10:14.
    """
    horizon = int(horizon_minutes)
    if horizon not in LONG_HORIZONS_MINUTES:
        raise ValueError("horizon_minutes must be one of 15, 30, ..., 1440")
    origin_ts = pd.Timestamp(origin)
    start = origin_ts + pd.Timedelta(minutes=horizon - 15)
    end = origin_ts + pd.Timedelta(minutes=horizon - 1)
    return IntervalBounds(start=start, end=end)


def origin_grid(timestamps: Iterable[pd.Timestamp | str]) -> pd.DatetimeIndex:
    """Return unique quarter-hour timestamps in increasing order."""
    values = pd.DatetimeIndex(pd.to_datetime(list(timestamps), errors="raise"))
    values = values[(values.minute % 15 == 0) & (values.second == 0)]
    return pd.DatetimeIndex(values.unique()).sort_values()


def interval_mean_series(
    minute_frame: pd.DataFrame,
    target: str,
) -> pd.Series:
    """Aggregate a complete minute target to left-labelled 15-minute means.

    An interval is returned as missing unless all 15 one-minute observations
    exist and are non-null.  Missing source minutes are never silently filled.
    """
    if "datetime" not in minute_frame or target not in minute_frame:
        raise KeyError(f"required columns missing: datetime/{target}")
    frame = minute_frame[["datetime", target]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    if frame["datetime"].duplicated().any():
        raise ValueError("duplicate datetime values are not allowed")
    frame = frame.sort_values("datetime").set_index("datetime")
    if frame.empty:
        return pd.Series(dtype=float, name=target)
    full_index = pd.date_range(frame.index.min(), frame.index.max(), freq="1min")
    values = pd.to_numeric(frame[target], errors="coerce").reindex(full_index)
    means = values.resample("15min", closed="left", label="left").mean()
    counts = values.resample("15min", closed="left", label="left").count()
    means = means.where(counts == 15)
    means.name = target
    return means


def build_interval_target_matrix(
    minute_frame: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
    horizons_minutes: Iterable[int] = LONG_HORIZONS_MINUTES,
    targets: Iterable[str] = TARGETS,
) -> pd.DataFrame:
    """Build the wide official target matrix for supplied prediction origins."""
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origins), errors="raise"))
    if origin_index.duplicated().any() or not origin_index.is_monotonic_increasing:
        raise ValueError("origins must be unique and increasing")
    horizons = tuple(int(value) for value in horizons_minutes)
    result = pd.DataFrame({"datetime": origin_index})
    for target in targets:
        aggregated = interval_mean_series(minute_frame, target)
        for horizon in horizons:
            if horizon not in LONG_HORIZONS_MINUTES:
                raise ValueError(f"invalid horizon: {horizon}")
            interval_starts = origin_index + pd.to_timedelta(horizon - 15, unit="min")
            result[f"{target}_t+{horizon}"] = aggregated.reindex(interval_starts).to_numpy()
    return result


def expected_result_columns(
    horizons_minutes: Iterable[int],
    targets: Iterable[str] = TARGETS,
) -> list[str]:
    """Return the exact official prediction column order."""
    horizons = [int(value) for value in horizons_minutes]
    return ["datetime"] + [
        f"{target}_t+{horizon}_pred" for target in targets for horizon in horizons
    ]


def validate_prediction_frame(
    frame: pd.DataFrame,
    expected_origins: Iterable[pd.Timestamp | str],
    horizons_minutes: Iterable[int],
) -> dict[str, int | float]:
    """Validate schema, timestamps, finite values and basic physical bounds."""
    expected_columns = expected_result_columns(horizons_minutes)
    if list(frame.columns) != expected_columns:
        raise AssertionError("prediction columns or order differ from the contract")
    origins = pd.DatetimeIndex(pd.to_datetime(list(expected_origins), errors="raise"))
    observed = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="raise"))
    if not observed.equals(origins):
        raise AssertionError("prediction origins differ from expected origins/order")
    if observed.duplicated().any() or not observed.is_monotonic_increasing:
        raise AssertionError("prediction origins must be unique and increasing")
    values = frame.drop(columns="datetime").apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise AssertionError("prediction values contain missing or non-finite values")

    horizons = [int(value) for value in horizons_minutes]
    g1 = values[[f"generator_1_t+{h}_pred" for h in horizons]].to_numpy(float)
    gall = values[[f"generator_all_t+{h}_pred" for h in horizons]].to_numpy(float)
    violations = {
        "generator_1_below_or_equal_zero": int((g1 <= 0).sum()),
        "generator_all_below_or_equal_zero": int((gall <= 0).sum()),
        "generator_1_above_200": int((g1 > 200).sum()),
        "generator_all_above_440": int((gall > 440).sum()),
        "generator_1_above_generator_all": int((g1 > gall).sum()),
    }
    if any(violations.values()):
        raise AssertionError(f"physical contract violations: {violations}")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "minimum_prediction": float(array.min()),
        "maximum_prediction": float(array.max()),
        **violations,
    }
