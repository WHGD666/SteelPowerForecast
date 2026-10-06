import pandas as pd

from src.round2_v3.run_short_g1_dynamic_fuel import summarize_predictions


def test_episode_summary_excludes_blank_episode_ids() -> None:
    rows = []
    for variant in ("v16_control", "candidate"):
        rows.extend(
            [
                {
                    "variant": variant,
                    "fold_id": "fold",
                    "horizon_minutes": 15,
                    "actual": 100.0,
                    "prediction": 101.0,
                    "origin_regime": "ordinary",
                    "target_episode_id": "",
                },
                {
                    "variant": variant,
                    "fold_id": "fold",
                    "horizon_minutes": 15,
                    "actual": 100.0,
                    "prediction": 101.0,
                    "origin_regime": "pre_event",
                    "target_episode_id": "episode_1",
                },
            ]
        )
    metrics = summarize_predictions(pd.DataFrame(rows))
    episodes = metrics.loc[metrics["scope"] == "episode", "target_episode_id"]
    assert set(episodes) == {"episode_1"}
