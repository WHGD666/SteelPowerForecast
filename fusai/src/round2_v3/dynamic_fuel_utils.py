"""Causal feature and cutoff helpers for horizon-specific dynamic fuel models."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_dynamic_fuel_features(
    origin_frame: pd.DataFrame,
    fuel_columns: Iterable[str],
    lag_steps: Iterable[int],
    rolling_mean_steps: Iterable[int],
    *,
    frequency_minutes: int = 15,
    include_current: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build origin-only fuel history features with explicit source offsets.

    The input must already be a complete, ordered origin grid. Every generated
    value uses the current origin or earlier origins only.
    """
    if frequency_minutes <= 0:
        raise ValueError("frequency_minutes must be positive")
    local = origin_frame.copy()
    if "datetime" not in local:
        raise KeyError("origin frame missing datetime")
    local["datetime"] = pd.to_datetime(local["datetime"], errors="raise")
    if local["datetime"].duplicated().any():
        raise ValueError("origin frame contains duplicate datetime values")
    local = local.sort_values("datetime").set_index("datetime")
    expected = pd.date_range(
        local.index.min(), local.index.max(), freq=f"{frequency_minutes}min"
    )
    if not local.index.equals(expected):
        raise ValueError("origin frame must be a complete fixed-frequency grid")

    fuels = list(fuel_columns)
    missing = set(fuels) - set(local.columns)
    if missing:
        raise KeyError(f"fuel columns absent from origin frame: {sorted(missing)}")
    lags = [int(value) for value in lag_steps]
    rolls = [int(value) for value in rolling_mean_steps]
    if any(value <= 0 for value in [*lags, *rolls]):
        raise ValueError("lag and rolling steps must be positive")
    if len(lags) != len(set(lags)) or len(rolls) != len(set(rolls)):
        raise ValueError("lag and rolling steps must be unique")

    values: dict[str, pd.Series] = {}
    registry: list[dict[str, object]] = []

    def add(
        name: str,
        series: pd.Series,
        base_signal: str,
        transform: str,
        source_min: int,
        source_max: int,
    ) -> None:
        if name in values:
            raise ValueError(f"duplicate feature: {name}")
        values[name] = series
        registry.append(
            {
                "feature_name": name,
                "base_signal": base_signal,
                "transform": transform,
                "source_offset_min_minutes": source_min,
                "source_offset_max_minutes": source_max,
                "causal": source_max <= 0,
            }
        )

    for column in fuels:
        series = pd.to_numeric(local[column], errors="coerce")
        if include_current:
            add(column, series, column, "current", 0, 0)
        for steps in lags:
            minutes = steps * frequency_minutes
            add(
                f"{column}__lag_{minutes}m",
                series.shift(steps),
                column,
                f"lag_{minutes}m",
                -minutes,
                -minutes,
            )
        for steps in rolls:
            minutes = steps * frequency_minutes
            add(
                f"{column}__roll_{minutes}m_mean",
                series.rolling(steps, min_periods=steps).mean(),
                column,
                f"rolling_{minutes}m_mean",
                -(steps - 1) * frequency_minutes,
                0,
            )

    feature_frame = pd.DataFrame(values, index=local.index).reset_index()
    registry_frame = pd.DataFrame.from_records(registry)
    if registry_frame.empty or not registry_frame["causal"].all():
        raise AssertionError("dynamic fuel registry contains non-causal features")
    return feature_frame, registry_frame


def latest_mature_origin(
    train_end: pd.Timestamp | str,
    horizon_minutes: int,
    *,
    frequency_minutes: int = 15,
) -> pd.Timestamp:
    """Return the latest aligned origin whose whole target interval is observed."""
    horizon = int(horizon_minutes)
    if horizon <= 0 or horizon % frequency_minutes != 0:
        raise ValueError("horizon must be a positive multiple of origin frequency")
    deadline = pd.Timestamp(train_end) - pd.Timedelta(minutes=horizon - 1)
    return deadline.floor(f"{frequency_minutes}min")


def recent_origin_window(
    origin_times: Iterable[pd.Timestamp | str],
    latest_origin: pd.Timestamp | str,
    *,
    window_days: int,
    frequency_minutes: int = 15,
) -> pd.DatetimeIndex:
    """Select exactly the latest N days of eligible fixed-frequency origins."""
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    index = pd.DatetimeIndex(pd.to_datetime(list(origin_times), errors="raise"))
    if index.duplicated().any() or not index.is_monotonic_increasing:
        raise ValueError("origin times must be unique and ordered")
    eligible = index[index <= pd.Timestamp(latest_origin)]
    expected_count = window_days * 24 * 60 // frequency_minutes
    if len(eligible) < expected_count:
        raise ValueError(
            f"insufficient origins for {window_days}-day window: "
            f"{len(eligible)} < {expected_count}"
        )
    selected = eligible[-expected_count:]
    differences = np.diff(selected.view("int64"))
    expected_delta = frequency_minutes * 60 * 1_000_000_000
    if len(differences) and not np.all(differences == expected_delta):
        raise ValueError("selected recent origin window is not continuous")
    return selected

