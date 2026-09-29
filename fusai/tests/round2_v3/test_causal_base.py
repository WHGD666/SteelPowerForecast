from __future__ import annotations

import pandas as pd

from src.round2_v3.prepare_causal_base import build_causal_partition


def _frames(future_value: float = 3.0) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    times = pd.date_range("2025-01-01 00:00:00", periods=3, freq="1min")
    frames = {
        "gas": pd.DataFrame(
            {"datetime": times, "gas_a": [1.0, 2.0, future_value], "partial": [None, 5.0, 6.0]}
        ),
        "gas_user": pd.DataFrame({"datetime": times.delete(1), "user_a": [10.0, 30.0]}),
        "gas_holder": pd.DataFrame({"datetime": times, "holder_a": [100.0, 101.0, 102.0]}),
        "load_process": pd.DataFrame({"datetime": times, "use_a": [7.0, 8.0, 9.0]}),
    }
    allowed = {
        "gas": ["gas_a", "partial"],
        "gas_user": ["user_a"],
        "gas_holder": ["holder_a"],
        "load_process": ["use_a"],
    }
    return frames, allowed


def test_alignment_inserts_missing_timestamp_without_backfill() -> None:
    frames, allowed = _frames()
    result = build_causal_partition(
        frames,
        allowed,
        ["partial"],
        "2025-01-01 00:00:00",
        "2025-01-01 00:02:00",
    )
    assert pd.isna(result.loc[1, "user_a"])
    assert result.loc[1, "user_a__missing"] == 1
    assert result.loc[1, "timestamp_inserted__gas_user"] == 1
    assert result["partial__active"].tolist() == [0, 1, 1]


def test_future_raw_change_cannot_change_earlier_causal_rows() -> None:
    frames_a, allowed = _frames(future_value=3.0)
    frames_b, _ = _frames(future_value=999999.0)
    base_a = build_causal_partition(
        frames_a, allowed, ["partial"], "2025-01-01", "2025-01-01 00:02:00"
    )
    base_b = build_causal_partition(
        frames_b, allowed, ["partial"], "2025-01-01", "2025-01-01 00:02:00"
    )
    pd.testing.assert_frame_equal(base_a.iloc[:2], base_b.iloc[:2])


def test_causal_base_contains_no_target_columns() -> None:
    frames, allowed = _frames()
    result = build_causal_partition(
        frames, allowed, ["partial"], "2025-01-01", "2025-01-01 00:02:00"
    )
    assert "generator_1" not in result
    assert "generator_all" not in result
