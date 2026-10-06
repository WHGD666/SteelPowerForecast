"""Causal helpers for the isolated v28-style g1 anomaly-weight experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FUELS_G1 = [
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
]
GALL_FUEL_COLUMNS = [
    "generator_use_blast_furnace_gas",
    "generator_use_converter_gas",
]
BAL_PROD = [
    "blast_furnace_1",
    "blast_furnace_2",
    "blast_furnace_3",
    "blast_furnace_4",
    "blast_furnace_5",
]
BAL_AH = [
    "air_heater_1",
    "air_heater_2",
    "air_heater_3",
    "air_heater_4",
    "air_heater_5",
]
BAL_BU = [
    "blast_furnace_user1",
    "blast_furnace_user2",
    "blast_furnace_user3",
    "blast_furnace_user4",
]

LEGACY_EXCLUDED_COLUMNS = ["blast_furnace_3", "air_heater_3"]
LEGACY_BAL_PROD = [
    "blast_furnace_1",
    "blast_furnace_2",
    "blast_furnace_4",
    "blast_furnace_5",
]
LEGACY_BAL_AH = ["air_heater_1", "air_heater_2", "air_heater_4"]


def merge_training_tables(paths: Iterable[Path]) -> pd.DataFrame:
    """Merge raw training tables without modifying the source files."""
    parts = [pd.read_csv(path, parse_dates=["datetime"]) for path in paths]
    if not parts:
        raise ValueError("at least one training table is required")
    merged = parts[0]
    for other in parts[1:]:
        merged = merged.merge(other, on="datetime", how="outer", validate="one_to_one")
    merged = merged.sort_values("datetime").reset_index(drop=True)
    if merged["datetime"].duplicated().any():
        raise ValueError("merged training timeline contains duplicate timestamps")
    merged = merged.drop(
        columns=["converter_user3", "blast_furnace_gas_holder_1"],
        errors="ignore",
    )
    if "blast_furnace_3" not in merged:
        raise KeyError("restored v28 column blast_furnace_3 is absent")
    merged["blast_furnace_3"] = merged["blast_furnace_3"].fillna(0.0)
    return merged


def to_v28_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the deployed v28 15-minute grid and causal ffill policy."""
    result = frame.loc[frame["datetime"].dt.minute.mod(15).eq(0)].copy()
    result = result.sort_values("datetime").reset_index(drop=True)
    covariates = [
        column
        for column in result.columns
        if column not in {"datetime", "generator_1", "generator_all"}
    ]
    result[covariates] = result[covariates].ffill(limit=4)
    if result["datetime"].duplicated().any():
        raise ValueError("v28 grid contains duplicate timestamps")
    differences = result["datetime"].diff().dropna()
    if not differences.eq(pd.Timedelta(minutes=15)).all():
        bad = differences.loc[~differences.eq(pd.Timedelta(minutes=15))].head()
        raise ValueError(f"v28 grid is not contiguous at: {bad.to_dict()}")
    return result


def build_v28_features(
    frame: pd.DataFrame,
    targets: Iterable[str] = ("generator_1", "generator_all"),
) -> tuple[pd.DataFrame, list[str]]:
    """Reproduce the feature frame used by the official v28 long-g1 component."""
    target_set = set(targets)
    covariates = [
        column
        for column in frame.columns
        if column != "datetime" and column not in target_set
    ]
    features = frame[covariates].copy()
    timestamp = frame["datetime"]
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    features["feat_hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["feat_hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    weekday_angle = 2 * np.pi * timestamp.dt.dayofweek / 7
    features["feat_dow_sin"] = np.sin(weekday_angle)
    features["feat_dow_cos"] = np.cos(weekday_angle)

    blast_furnace_balance = frame[BAL_PROD].sum(axis=1) - (
        frame[BAL_AH].sum(axis=1)
        + frame[BAL_BU].sum(axis=1)
        + frame["into_gas_mixed_blast_furnace"]
    )
    coke_balance = frame["coke_oven_1"] - frame["into_gas_mixed_coke"]
    converter_balance = frame["converter_1"] - (
        frame["converter_user2"] + frame["into_gas_mixed_converter"]
    )
    for name, values in (
        ("bf", blast_furnace_balance),
        ("coke", coke_balance),
        ("conv", converter_balance),
    ):
        features[f"feat_bal_{name}"] = values
        for window in (4, 16, 96):
            features[f"feat_bal_{name}_rm{window}"] = values.rolling(
                window, min_periods=1
            ).mean()
    features["feat_bal_total"] = (
        blast_furnace_balance + coke_balance + converter_balance
    )
    holder_2 = frame["blast_furnace_gas_holder_2"]
    for lag in (1, 4, 16):
        features[f"feat_holder_d{lag}"] = holder_2.diff(lag)
    features["feat_holder_d1_rm4"] = holder_2.diff(1).rolling(
        4, min_periods=1
    ).mean()
    for column in FUELS_G1:
        for lag in (1, 4):
            features[f"feat_{column}_d{lag}"] = frame[column].diff(lag)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, list(features.columns)


def build_v16_legacy_features(
    frame: pd.DataFrame,
    targets: Iterable[str] = ("generator_1", "generator_all"),
) -> tuple[pd.DataFrame, list[str]]:
    """Reproduce the old-column feature engine upstream of v16_final.

    v16 was built from prepared_v2, which excluded blast_furnace_3 and
    air_heater_3 and used the older incomplete blast-furnace balance.  The
    function intentionally keeps that historical omission for a fair control.
    """
    target_set = set(targets)
    covariates = [
        column
        for column in frame.columns
        if column != "datetime"
        and column not in target_set
        and column not in LEGACY_EXCLUDED_COLUMNS
    ]
    features = frame[covariates].copy()
    timestamp = frame["datetime"]
    minute_of_day = timestamp.dt.hour * 60 + timestamp.dt.minute
    features["feat_hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["feat_hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    weekday_angle = 2 * np.pi * timestamp.dt.dayofweek / 7
    features["feat_day_of_week_sin"] = np.sin(weekday_angle)
    features["feat_day_of_week_cos"] = np.cos(weekday_angle)

    blast_furnace_balance = frame[LEGACY_BAL_PROD].sum(axis=1) - (
        frame[LEGACY_BAL_AH].sum(axis=1)
        + frame[BAL_BU].sum(axis=1)
        + frame["into_gas_mixed_blast_furnace"]
    )
    coke_balance = frame["coke_oven_1"] - frame["into_gas_mixed_coke"]
    converter_balance = frame["converter_1"] - (
        frame["converter_user2"] + frame["into_gas_mixed_converter"]
    )
    total_balance = blast_furnace_balance + coke_balance + converter_balance
    for name, values in (
        ("feat_blast_furnace_balance", blast_furnace_balance),
        ("feat_coke_balance", coke_balance),
        ("feat_converter_balance", converter_balance),
        ("feat_total_gas_balance", total_balance),
    ):
        for window in (4, 16, 96):
            features[f"{name}_roll_mean_{window}"] = values.rolling(
                window, min_periods=1
            ).mean()
        features[name] = values

    holder_2 = frame["blast_furnace_gas_holder_2"]
    for lag in (1, 4, 16):
        features[f"feat_holder_delta_{lag}"] = holder_2.diff(lag)
    features["feat_holder_delta_roll_mean_4"] = holder_2.diff(1).rolling(
        4, min_periods=1
    ).mean()
    for column in FUELS_G1:
        for lag in (1, 4):
            features[f"feat_{column}_diff_{lag}"] = frame[column].diff(lag)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, list(features.columns)


def fit_linear_fuel_proxy(
    frame: pd.DataFrame,
    target: str,
    fuel_columns: list[str],
) -> np.ndarray:
    """Fit the legacy unregularized fuel relation on a declared training frame."""
    matrix = np.column_stack(
        [frame[column].to_numpy(dtype=float) for column in fuel_columns]
        + [np.ones(len(frame), dtype=float)]
    )
    values = frame[target].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(matrix).all(axis=1)
    if valid.sum() <= matrix.shape[1]:
        raise ValueError("not enough finite rows for fuel proxy")
    coefficients, *_ = np.linalg.lstsq(matrix[valid], values[valid], rcond=None)
    return coefficients


def predict_linear_fuel(
    frame: pd.DataFrame,
    fuel_columns: list[str],
    coefficients: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack(
        [frame[column].to_numpy(dtype=float) for column in fuel_columns]
        + [np.ones(len(frame), dtype=float)]
    )
    prediction = matrix @ np.asarray(coefficients, dtype=float)
    return np.clip(prediction, 0.0, None)


def causal_low_fuel_ratio(
    frame: pd.DataFrame,
    coefficients: np.ndarray,
    *,
    smooth_steps: int = 16,
    baseline_steps: int = 672,
    baseline_min_periods: int = 96,
) -> pd.Series:
    """Compute the legacy fuel ratio using backward-looking rolling windows only."""
    fuel = predict_linear_fuel(frame, GALL_FUEL_COLUMNS, coefficients)
    smooth = pd.Series(fuel, index=frame.index).rolling(
        int(smooth_steps), min_periods=1
    ).mean()
    baseline = smooth.rolling(
        int(baseline_steps), min_periods=int(baseline_min_periods)
    ).mean()
    ratio = smooth / baseline
    return ratio.replace([np.inf, -np.inf], np.nan)


def anomaly_sample_weights(
    ratio: pd.Series | np.ndarray,
    *,
    threshold: float,
    multiplier: float,
) -> np.ndarray:
    """Return 1 outside low-fuel rows and multiplier strictly below threshold."""
    values = np.asarray(ratio, dtype=float)
    if not 0 < float(threshold):
        raise ValueError("threshold must be positive")
    if float(multiplier) < 1:
        raise ValueError("multiplier must be at least one")
    return np.where(np.isfinite(values) & (values < threshold), multiplier, 1.0)


def v28_g1_fuel_weight(horizon_minutes: int) -> float:
    """Preserve deployed code semantics: `h_step <= 32` means 15/30 minutes."""
    horizon = int(horizon_minutes)
    if horizon <= 0 or horizon % 15:
        raise ValueError("horizon must be a positive multiple of 15 minutes")
    return 0.8 if horizon <= 32 else 0.6
