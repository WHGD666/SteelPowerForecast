"""Utilities for causal low-fuel episode-onset hazard diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def onset_within_horizon(
    origins: pd.Series | pd.DatetimeIndex,
    episode_starts: pd.Series | pd.DatetimeIndex,
    horizon_minutes: int,
) -> np.ndarray:
    """Label whether an episode starts strictly after origin and by origin+h."""
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError("horizon_minutes must be positive")
    origin_index = pd.DatetimeIndex(pd.to_datetime(origins, errors="raise"))
    starts = np.sort(
        pd.DatetimeIndex(pd.to_datetime(episode_starts, errors="raise")).asi8
    )
    origin_ns = origin_index.asi8
    end_ns = (origin_index + pd.to_timedelta(horizon, unit="min")).asi8
    next_positions = np.searchsorted(starts, origin_ns, side="right")
    labels = np.zeros(len(origin_index), dtype=np.int8)
    valid = next_positions < len(starts)
    labels[valid] = (starts[next_positions[valid]] <= end_ns[valid]).astype(np.int8)
    return labels


def calendar_features(origins: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    timestamp = pd.DatetimeIndex(pd.to_datetime(origins, errors="raise"))
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    hour_angle = 2 * np.pi * minute_of_day / 1440
    weekday_angle = 2 * np.pi * timestamp.dayofweek / 7
    return pd.DataFrame(
        {
            "calendar_hour_sin": np.sin(hour_angle),
            "calendar_hour_cos": np.cos(hour_angle),
            "calendar_dow_sin": np.sin(weekday_angle),
            "calendar_dow_cos": np.cos(weekday_angle),
            "calendar_is_weekend": (timestamp.dayofweek >= 5).astype(float),
        }
    )


def quantile_alert_threshold(
    probabilities: np.ndarray,
    alert_budget_fraction: float,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    budget = float(alert_budget_fraction)
    if not 0 < budget < 1:
        raise ValueError("alert_budget_fraction must be between zero and one")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    return float(np.quantile(values, 1.0 - budget))


def episode_detection_rows(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Summarize pre-onset alerts for the held-out episodes in one fold."""
    required = {"datetime", "alert", "probability", "fold_id", "model"}
    if not required <= set(predictions.columns):
        raise KeyError(f"predictions missing {sorted(required - set(predictions.columns))}")
    horizon = pd.Timedelta(minutes=int(horizon_minutes))
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        start = pd.Timestamp(episode.episode_start)
        window = predictions.loc[
            (predictions["datetime"] >= start - horizon)
            & (predictions["datetime"] < start)
        ].copy()
        alerted = window.loc[window["alert"].astype(bool)]
        rows.append(
            {
                "fold_id": predictions["fold_id"].iloc[0],
                "model": predictions["model"].iloc[0],
                "horizon_minutes": int(horizon_minutes),
                "episode_id": episode.episode_id,
                "episode_start": start,
                "window_rows": int(len(window)),
                "alert_rows": int(len(alerted)),
                "alert_fraction": float(alerted.shape[0] / len(window))
                if len(window)
                else np.nan,
                "detected": bool(len(alerted)),
                "earliest_lead_hours": float(
                    ((start - alerted["datetime"]).dt.total_seconds() / 3600).max()
                )
                if len(alerted)
                else np.nan,
                "maximum_probability": float(window["probability"].max())
                if len(window)
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)
