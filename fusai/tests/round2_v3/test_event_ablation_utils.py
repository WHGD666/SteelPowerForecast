from __future__ import annotations

import pandas as pd

from src.round2_v3.event_ablation_utils import attach_event_context, horizon_bucket


def test_event_context_distinguishes_pre_active_recovery_and_target() -> None:
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-02 00:00",
                    "2025-01-02 12:00",
                    "2025-01-03 12:00",
                ]
            ),
            "interval_start": pd.to_datetime(
                [
                    "2025-01-01 01:00",
                    "2025-01-02 12:00",
                    "2025-01-03 00:00",
                    "2025-01-04 00:00",
                ]
            ),
            "horizon_minutes": [60, 720, 720, 720],
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["event_1"],
            "episode_start": pd.to_datetime(["2025-01-02 06:00"]),
            "episode_end": pd.to_datetime(["2025-01-03 06:00"]),
        }
    )
    result = attach_event_context(frame, episodes, context_hours=24)
    assert result["origin_regime"].tolist() == [
        "ordinary",
        "pre_event",
        "active_event",
        "recovery",
    ]
    assert result["target_in_event"].tolist() == [False, True, True, False]


def test_horizon_bucket_boundaries() -> None:
    values = pd.Series([15, 120, 135, 360, 375, 720, 735, 1440])
    assert horizon_bucket(values).tolist() == [
        "h015_120",
        "h015_120",
        "h135_360",
        "h135_360",
        "h375_720",
        "h375_720",
        "h735_1440",
        "h735_1440",
    ]
