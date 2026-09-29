from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.round2_v3.build_origin_features import build_origin_features


ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
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
    return config


def _base(future_value: float = 45.0) -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=46, freq="1min")
    signal = np.arange(46, dtype=float)
    signal[-1] = future_value
    return pd.DataFrame(
        {
            "datetime": times,
            "signal": signal,
            "signal__missing": np.zeros(46, dtype=np.int8),
            "timestamp_inserted__gas": np.zeros(46, dtype=np.int8),
        }
    )


def test_feature_values_use_origin_and_past_only() -> None:
    features, registry = build_origin_features(_base(), _config())
    row = features.loc[features["datetime"] == pd.Timestamp("2025-01-01 00:30:00")].iloc[0]
    assert row["signal"] == 30.0
    assert row["signal__lag_15m"] == 15.0
    assert row["signal__roll_15m_mean"] == np.mean(np.arange(16, 31))
    assert registry["causal"].all()
    assert registry["source_offset_max_minutes"].max() == 0


def test_changing_future_value_does_not_change_past_origin_features() -> None:
    features_a, _ = build_origin_features(_base(future_value=45.0), _config())
    features_b, _ = build_origin_features(_base(future_value=999999.0), _config())
    past_a = features_a.loc[features_a["datetime"] <= pd.Timestamp("2025-01-01 00:30:00")]
    past_b = features_b.loc[features_b["datetime"] <= pd.Timestamp("2025-01-01 00:30:00")]
    pd.testing.assert_frame_equal(past_a, past_b)


def test_origin_grid_is_quarter_hour_and_target_free() -> None:
    features, _ = build_origin_features(_base(), _config())
    assert features["datetime"].dt.minute.tolist() == [0, 15, 30, 45]
    assert "generator_1" not in features
    assert "generator_all" not in features
