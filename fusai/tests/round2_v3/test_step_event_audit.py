from __future__ import annotations

import numpy as np
import pandas as pd

from src.round2_v3.audit_step_events import (
    add_outcome_classes,
    available_in_formal_test,
    build_causal_transforms,
    event_day_catalog,
    legacy_fuel_ratio,
    merge_event_days,
)


def test_event_days_and_consecutive_episode_grouping() -> None:
    index = pd.date_range("2025-01-01", periods=5 * 96, freq="15min")
    ratio = pd.Series(1.0, index=index)
    ratio.loc["2025-01-01 12:00"] = 0.7
    ratio.loc["2025-01-02 08:00"] = 0.6
    ratio.loc["2025-01-04 09:00"] = 0.75
    days = event_day_catalog(ratio, 0.8, "train")
    episodes = merge_event_days(days, "train")
    assert len(days) == 3
    assert len(episodes) == 2
    assert episodes.iloc[0]["event_day_count"] == 2
    assert episodes.iloc[1]["event_day_count"] == 1


def test_causal_transforms_ignore_future_changes() -> None:
    index = pd.date_range("2025-01-01", periods=900, freq="15min")
    base = pd.DataFrame({"signal": np.arange(len(index), dtype=float)}, index=index)
    changed = base.copy()
    cutoff = index[700]
    changed.loc[changed.index > cutoff, "signal"] = 999999.0
    features_a = build_causal_transforms(base, [1, 2, 4], [1, 4], 168)
    features_b = build_causal_transforms(changed, [1, 2, 4], [1, 4], 168)
    pd.testing.assert_frame_equal(
        features_a.loc[features_a.index <= cutoff],
        features_b.loc[features_b.index <= cutoff],
    )


def test_legacy_ratio_uses_trailing_windows_only() -> None:
    index = pd.date_range("2025-01-01", periods=20, freq="15min")
    grid = pd.DataFrame(
        {
            "datetime": index,
            "fuel_a": np.full(20, 10.0),
            "fuel_b": np.full(20, 5.0),
        }
    )
    coefficients = np.array([1.0, 1.0, 0.0])
    ratio_a = legacy_fuel_ratio(grid, ["fuel_a", "fuel_b"], coefficients, 4, 8, 4)
    changed = grid.copy()
    changed.loc[changed.index > 10, "fuel_a"] = 1000.0
    ratio_b = legacy_fuel_ratio(changed, ["fuel_a", "fuel_b"], coefficients, 4, 8, 4)
    pd.testing.assert_series_equal(ratio_a.iloc[:11], ratio_b.iloc[:11])


def test_outcome_classes_distinguish_drop_and_rebound() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": ["a", "b"],
            "future_2h_g1_mean_change": [-10.0, 2.0],
            "future_6h_g1_mean_change": [-7.0, -6.0],
            "future_12h_g1_mean_change": [-3.0, -8.0],
            "future_24h_g1_mean_change": [4.0, -9.0],
        }
    )
    result = add_outcome_classes(frame, -5.0)
    assert result["sustained_drop_6h"].tolist() == [True, False]
    assert result["sustained_drop_12h"].tolist() == [False, False]
    assert result["rebound_by_24h"].tolist() == [True, False]


def test_true_target_history_is_diagnostic_only() -> None:
    assert not available_in_formal_test("g1_previous_complete_interval__delta_4h")
    assert not available_in_formal_test("gall_previous_complete_interval__level")
    assert available_in_formal_test("holder_2__delta_4h")
