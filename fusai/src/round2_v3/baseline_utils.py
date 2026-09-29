"""Leakage-safe data assembly and scoring helpers for the v3 baseline."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.round2_v3.metrics import regression_metrics


TARGET_CALENDAR_COLUMNS = (
    "horizon_minutes",
    "horizon_step",
    "target_hour_sin",
    "target_hour_cos",
    "target_minute_sin",
    "target_minute_cos",
    "target_dayofweek_sin",
    "target_dayofweek_cos",
    "target_is_weekend",
    "target_month",
    "target_day_of_month",
)


def select_feature_columns(registry: pd.DataFrame, selection: dict) -> list[str]:
    """Select a compact, declared subset from the frozen feature registry."""
    required = {"feature_name", "base_signal", "transform", "causal"}
    if not required <= set(registry.columns):
        raise KeyError(f"feature registry missing: {sorted(required - set(registry.columns))}")
    causal = registry["causal"].astype(str).str.lower().isin({"true", "1"})
    if not causal.all():
        raise ValueError("feature registry contains non-causal entries")
    base_signals = set(selection["base_signals"])
    transforms = set(selection["transforms"])
    chosen = registry[
        registry["base_signal"].isin(base_signals)
        & registry["transform"].isin(transforms)
    ]
    if selection.get("include_origin_calendar", False):
        chosen = pd.concat([chosen, registry[registry["transform"] == "calendar"]])
    if selection.get("include_all_current_state_indicators", False):
        chosen = pd.concat(
            [chosen, registry[registry["transform"] == "current_state_indicator"]]
        )
    chosen = chosen.drop_duplicates("feature_name")
    missing_signals = base_signals - set(chosen["base_signal"])
    if missing_signals:
        raise ValueError(f"configured base signals selected no features: {sorted(missing_signals)}")
    return chosen["feature_name"].tolist()


def filter_origins_by_stride(
    origins: Iterable[pd.Timestamp | str], stride_minutes: int
) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(list(origins), errors="raise"))
    stride = int(stride_minutes)
    if stride <= 0 or stride % 15 != 0:
        raise ValueError("training origin stride must be a positive multiple of 15 minutes")
    epoch_minutes = values.view("int64") // (60 * 1_000_000_000)
    return values[epoch_minutes % stride == 0]


def build_supervised_long(
    origin_features: pd.DataFrame,
    interval_targets: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
    horizons_minutes: Iterable[int],
    target: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Expand origin rows over horizons and attach official interval labels."""
    x, meta = build_prediction_long(
        origin_features,
        origins,
        horizons_minutes,
        feature_columns,
    )
    labels = interval_targets[["datetime", target]].copy()
    labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
    if labels["datetime"].duplicated().any():
        raise ValueError("interval target timestamps are not unique")
    label_series = pd.to_numeric(labels.set_index("datetime")[target], errors="coerce")
    actual = label_series.reindex(pd.DatetimeIndex(meta["interval_start"])).to_numpy(dtype=float)
    meta["target"] = target
    valid = np.isfinite(actual)
    return x.loc[valid].reset_index(drop=True), actual[valid], meta.loc[valid].reset_index(drop=True)


def build_prediction_long(
    origin_features: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
    horizons_minutes: Iterable[int],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand origin features over horizons without requiring future labels."""
    features = origin_features.copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
    if features["datetime"].duplicated().any():
        raise ValueError("origin feature timestamps are not unique")
    features = features.set_index("datetime")
    missing_features = set(feature_columns) - set(features.columns)
    if missing_features:
        raise KeyError(f"selected features absent: {sorted(missing_features)}")

    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origins), errors="raise"))
    if origin_index.duplicated().any():
        raise ValueError("requested origins are not unique")
    absent_origins = origin_index.difference(features.index)
    if len(absent_origins):
        raise KeyError(f"origin features missing {len(absent_origins)} requested timestamps")
    horizons = np.asarray([int(value) for value in horizons_minutes], dtype=np.int32)
    if len(horizons) == 0 or np.any(horizons <= 0) or np.any(horizons % 15 != 0):
        raise ValueError("horizons must be positive multiples of 15 minutes")

    origin_matrix = features.loc[origin_index, feature_columns].to_numpy(dtype=np.float32)
    repeated_matrix = np.repeat(origin_matrix, len(horizons), axis=0)
    repeated_origins = pd.DatetimeIndex(np.repeat(origin_index.to_numpy(), len(horizons)))
    repeated_horizons = np.tile(horizons, len(origin_index))
    interval_starts = repeated_origins + pd.to_timedelta(
        repeated_horizons - 15, unit="min"
    )
    x = pd.DataFrame(repeated_matrix, columns=feature_columns)
    hour_angle = 2 * np.pi * (interval_starts.hour + interval_starts.minute / 60) / 24
    minute_angle = 2 * np.pi * interval_starts.minute / 60
    day_angle = 2 * np.pi * interval_starts.dayofweek / 7
    x["horizon_minutes"] = repeated_horizons.astype(np.float32)
    x["horizon_step"] = (repeated_horizons / 15).astype(np.float32)
    x["target_hour_sin"] = np.sin(hour_angle).astype(np.float32)
    x["target_hour_cos"] = np.cos(hour_angle).astype(np.float32)
    x["target_minute_sin"] = np.sin(minute_angle).astype(np.float32)
    x["target_minute_cos"] = np.cos(minute_angle).astype(np.float32)
    x["target_dayofweek_sin"] = np.sin(day_angle).astype(np.float32)
    x["target_dayofweek_cos"] = np.cos(day_angle).astype(np.float32)
    x["target_is_weekend"] = (interval_starts.dayofweek >= 5).astype(np.float32)
    x["target_month"] = interval_starts.month.astype(np.float32)
    x["target_day_of_month"] = interval_starts.day.astype(np.float32)
    meta = pd.DataFrame(
        {
            "datetime": repeated_origins,
            "interval_start": interval_starts,
            "horizon_minutes": repeated_horizons,
        }
    )
    return x, meta


def fit_calendar_profile(
    interval_targets: pd.DataFrame, target: str, train_end: pd.Timestamp | str
) -> dict[str, object]:
    """Fit a target-time median profile using only complete past intervals."""
    frame = interval_targets[["datetime", target]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    latest_start = pd.Timestamp(train_end) - pd.Timedelta(minutes=14)
    frame = frame.loc[(frame["datetime"] <= latest_start) & frame[target].notna()].copy()
    if frame.empty:
        raise ValueError("no target intervals available for calendar profile")
    frame["dayofweek"] = frame["datetime"].dt.dayofweek
    frame["slot"] = frame["datetime"].dt.hour * 4 + frame["datetime"].dt.minute // 15
    return {
        "dow_slot": frame.groupby(["dayofweek", "slot"])[target].median().to_dict(),
        "slot": frame.groupby("slot")[target].median().to_dict(),
        "overall": float(frame[target].median()),
    }


def predict_calendar_profile(
    profile: dict[str, object], interval_starts: Iterable[pd.Timestamp | str]
) -> np.ndarray:
    times = pd.DatetimeIndex(pd.to_datetime(list(interval_starts), errors="raise"))
    dow_slot = profile["dow_slot"]
    slot_profile = profile["slot"]
    overall = float(profile["overall"])
    predictions = []
    for value in times:
        slot = value.hour * 4 + value.minute // 15
        predictions.append(dow_slot.get((value.dayofweek, slot), slot_profile.get(slot, overall)))
    return np.asarray(predictions, dtype=float)


def summarize_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled OOF metrics and required diagnostic views."""
    required = {
        "model",
        "period",
        "fold_id",
        "target",
        "horizon_minutes",
        "actual",
        "prediction",
    }
    if not required <= set(predictions.columns):
        raise KeyError(f"OOF predictions missing: {sorted(required - set(predictions.columns))}")
    rows: list[dict[str, object]] = []

    views = [
        ("overall", ["model", "period"]),
        ("target", ["model", "period", "target"]),
        ("fold_target", ["model", "period", "fold_id", "target"]),
        ("target_horizon", ["model", "period", "target", "horizon_minutes"]),
    ]
    for scope, group_columns in views:
        for keys, part in predictions.groupby(group_columns, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(group_columns, keys, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "model": identity.get("model", "all"),
                    "period": identity.get("period", "all"),
                    "fold_id": identity.get("fold_id", "all"),
                    "target": identity.get("target", "all"),
                    "horizon_minutes": identity.get("horizon_minutes", "all"),
                    **regression_metrics(part["actual"], part["prediction"]),
                }
            )
    return pd.DataFrame.from_records(rows)
