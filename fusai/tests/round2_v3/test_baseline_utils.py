from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3 import run_baseline
from src.round2_v3.baseline_utils import (
    TARGET_CALENDAR_COLUMNS,
    build_supervised_long,
    build_prediction_long,
    filter_origins_by_stride,
    fit_calendar_profile,
    predict_calendar_profile,
    select_feature_columns,
    summarize_oof,
)


def test_runner_imports_the_frozen_target_calendar_contract() -> None:
    assert run_baseline.TARGET_CALENDAR_COLUMNS == TARGET_CALENDAR_COLUMNS
    assert len(TARGET_CALENDAR_COLUMNS) == 11


def test_select_feature_columns_uses_declared_registry_fields() -> None:
    registry = pd.DataFrame(
        {
            "feature_name": ["a", "a__lag_15m", "hour_sin", "a__missing"],
            "base_signal": ["a", "a", "datetime", "a__missing"],
            "transform": ["current", "lag_15m", "calendar", "current_state_indicator"],
            "causal": [True, True, True, True],
        }
    )
    selected = select_feature_columns(
        registry,
        {
            "base_signals": ["a"],
            "transforms": ["current", "lag_15m"],
            "include_origin_calendar": True,
            "include_all_current_state_indicators": True,
        },
    )
    assert selected == ["a", "a__lag_15m", "hour_sin", "a__missing"]


def test_long_dataset_maps_horizons_to_correct_interval_starts() -> None:
    origins = pd.date_range("2025-01-01 00:00:00", periods=3, freq="15min")
    features = pd.DataFrame({"datetime": origins, "feature": [1.0, 2.0, 3.0]})
    intervals = pd.date_range("2025-01-01 00:00:00", periods=8, freq="15min")
    labels = pd.DataFrame(
        {"datetime": intervals, "generator_1": np.arange(8, dtype=float) + 100}
    )
    x, y, meta = build_supervised_long(
        features, labels, origins[:2], [15, 30], "generator_1", ["feature"]
    )
    assert meta["interval_start"].tolist() == [
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 00:15:00"),
        pd.Timestamp("2025-01-01 00:15:00"),
        pd.Timestamp("2025-01-01 00:30:00"),
    ]
    assert y.tolist() == [100.0, 101.0, 101.0, 102.0]
    assert x["feature"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_prediction_long_does_not_require_future_labels() -> None:
    origins = pd.date_range("2025-10-01", periods=2, freq="15min")
    features = pd.DataFrame({"datetime": origins, "feature": [1.0, 2.0]})
    x, meta = build_prediction_long(features, origins, [15, 30], ["feature"])
    assert len(x) == 4
    assert meta["interval_start"].tolist() == [
        pd.Timestamp("2025-10-01 00:00:00"),
        pd.Timestamp("2025-10-01 00:15:00"),
        pd.Timestamp("2025-10-01 00:15:00"),
        pd.Timestamp("2025-10-01 00:30:00"),
    ]


def test_stride_and_calendar_profile_are_past_only() -> None:
    origins = pd.date_range("2025-01-01", periods=8, freq="15min")
    assert filter_origins_by_stride(origins, 60).minute.tolist() == [0, 0]
    labels = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=96 * 8, freq="15min"),
            "generator_1": np.arange(96 * 8, dtype=float) + 1,
        }
    )
    cutoff = pd.Timestamp("2025-01-04 23:59:00")
    profile_a = fit_calendar_profile(labels, "generator_1", cutoff)
    changed = labels.copy()
    changed.loc[changed["datetime"] > cutoff, "generator_1"] = 999999.0
    profile_b = fit_calendar_profile(changed, "generator_1", cutoff)
    assert predict_calendar_profile(profile_a, origins).tolist() == pytest.approx(
        predict_calendar_profile(profile_b, origins).tolist()
    )


def test_oof_summary_pools_cells() -> None:
    predictions = pd.DataFrame(
        {
            "model": ["m"] * 4,
            "period": ["short"] * 4,
            "fold_id": ["f1", "f1", "f2", "f2"],
            "target": ["generator_1", "generator_all"] * 2,
            "horizon_minutes": [15, 15, 15, 15],
            "actual": [100.0, 200.0, 100.0, 200.0],
            "prediction": [90.0, 180.0, 110.0, 220.0],
        }
    )
    report = summarize_oof(predictions)
    overall = report.loc[report["scope"] == "overall"].iloc[0]
    assert overall["mape"] == pytest.approx(0.1)
    assert overall["valid_count"] == 4
