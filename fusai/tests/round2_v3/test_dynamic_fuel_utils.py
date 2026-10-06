from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.dynamic_fuel_utils import (
    build_dynamic_fuel_features,
    latest_mature_origin,
    recent_origin_window,
)


def _frame(periods: int = 200) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="15min")
    return pd.DataFrame(
        {
            "datetime": index,
            "fuel_a": np.arange(periods, dtype=float),
            "fuel_b": np.arange(periods, dtype=float) * 10,
        }
    )


def test_dynamic_features_have_expected_offsets_and_values() -> None:
    features, registry = build_dynamic_fuel_features(
        _frame(), ["fuel_a", "fuel_b"], [1, 4], [4], frequency_minutes=15
    )
    row = features.iloc[8]
    assert row["fuel_a"] == 8
    assert row["fuel_a__lag_15m"] == 7
    assert row["fuel_a__lag_60m"] == 4
    assert row["fuel_a__roll_60m_mean"] == pytest.approx(6.5)
    assert registry["causal"].all()
    assert registry["source_offset_max_minutes"].max() == 0
    assert registry["source_offset_min_minutes"].min() == -60


def test_future_perturbation_does_not_change_past_features() -> None:
    source = _frame()
    first, _ = build_dynamic_fuel_features(
        source, ["fuel_a", "fuel_b"], [1, 4], [4], frequency_minutes=15
    )
    changed = source.copy()
    cutoff = pd.Timestamp("2025-01-02 00:00:00")
    changed.loc[changed["datetime"] > cutoff, ["fuel_a", "fuel_b"]] = 1e12
    second, _ = build_dynamic_fuel_features(
        changed, ["fuel_a", "fuel_b"], [1, 4], [4], frequency_minutes=15
    )
    columns = [column for column in first if column != "datetime"]
    pd.testing.assert_frame_equal(
        first.loc[first["datetime"] <= cutoff, columns].reset_index(drop=True),
        second.loc[second["datetime"] <= cutoff, columns].reset_index(drop=True),
    )


def test_latest_mature_origin_matches_interval_contract() -> None:
    train_end = pd.Timestamp("2025-06-30 23:59:00")
    assert latest_mature_origin(train_end, 15) == pd.Timestamp("2025-06-30 23:45:00")
    assert latest_mature_origin(train_end, 1440) == pd.Timestamp("2025-06-30 00:00:00")


def test_recent_window_has_exact_count_and_end() -> None:
    frame = _frame(periods=300)
    latest = frame["datetime"].iloc[-1]
    selected = recent_origin_window(
        frame["datetime"], latest, window_days=2, frequency_minutes=15
    )
    assert len(selected) == 192
    assert selected[-1] == latest
    assert selected[0] == latest - pd.Timedelta(minutes=15 * 191)

