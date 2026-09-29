from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.round2_v3.build_origin_features import build_origin_features


ROOT = Path(__file__).resolve().parents[2]


def test_first_test_origin_uses_training_history() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "round2_v3" / "features_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = copy.deepcopy(config)
    config["signal_columns"] = ["signal"]
    config["derived_observed_aggregates"] = {}
    config["history"]["lag_minutes"] = [15]
    config["history"]["rolling_windows_minutes"] = [15]
    config["history"]["missing_fraction_windows_minutes"] = [15]
    timestamps = pd.date_range("2025-09-30 23:30:00", periods=46, freq="1min")
    base = pd.DataFrame(
        {
            "datetime": timestamps,
            "signal": np.arange(46, dtype=float),
            "signal__missing": 0,
            "timestamp_inserted__gas": 0,
        }
    )
    features, _ = build_origin_features(base, config)
    first_test = features.loc[
        features["datetime"] == pd.Timestamp("2025-10-01 00:00:00")
    ].iloc[0]
    assert first_test["signal__lag_15m"] == 15.0
    assert first_test["signal__roll_15m_mean"] == np.mean(np.arange(16, 31))
