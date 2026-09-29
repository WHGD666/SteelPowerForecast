from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.round2_v3.build_labels import build_interval_table
from src.round2_v3.build_splits import CONFIG_PATH, materialize


def test_interval_artifact_uses_complete_left_closed_blocks() -> None:
    timestamps = pd.date_range("2025-01-01", periods=45, freq="1min")
    frame = pd.DataFrame(
        {
            "datetime": timestamps,
            "generator_1": np.arange(45, dtype=float),
            "generator_all": np.arange(45, dtype=float) + 100,
        }
    ).drop(index=20)
    table = build_interval_table(frame)
    assert table.loc[0, "generator_1"] == 7.0
    assert pd.isna(table.loc[1, "generator_1"])
    assert table.loc[2, "generator_all"] == 137.0
    assert table["label_complete"].tolist() == [True, False, True]


def test_frozen_split_config_is_causal_disjoint_and_inside_raw_range() -> None:
    config = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    assignments = materialize(config)
    assert config["status"] == "frozen_for_v3_baseline"
    assert config["random_split"] is False
    assert assignments["validation_origin"].is_unique
    assert (assignments["train_end"] < assignments["validation_origin"]).all()
    assert (
        assignments["latest_eligible_train_origin"]
        + pd.Timedelta(minutes=1439)
        <= assignments["train_end"]
    ).all()
    assert (assignments["long_label_end"] <= pd.Timestamp(config["raw_train_end"])).all()
    assert set(assignments["validation_origin"].dt.minute.unique()) <= {0, 15, 30, 45}


def test_consumed_period_is_not_mislabeled_as_sealed() -> None:
    config = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    assignments = materialize(config)
    consumed = assignments.loc[assignments["fold_id"] == "wf_06_consumed_dev"]
    assert not consumed.empty
    assert set(consumed["role"]) == {"consumed_development_only"}
    assert "sealed" not in set(assignments["role"])
