import importlib.util
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_splits", ROOT / "src" / "build_splits.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_split_counts_and_boundaries() -> None:
    config = yaml.safe_load((ROOT / "configs" / "splits_v1.yaml").read_text(encoding="utf-8"))
    assignments = MODULE.build_assignments(config)
    development = assignments[assignments["role"] == "development"]
    holdout = assignments[assignments["role"] == "holdout"]
    assert development.groupby("fold_id").size().to_dict() == {
        "dev_01": 192,
        "dev_02": 192,
        "dev_03": 192,
    }
    assert len(holdout) == 185
    assert holdout["sealed"].all()
    assert holdout["max_target_time"].max() == pd.Timestamp("2025-05-01 00:00:00")


def test_training_cutoff_precedes_every_validation_origin() -> None:
    config = yaml.safe_load((ROOT / "configs" / "splits_v1.yaml").read_text(encoding="utf-8"))
    assignments = MODULE.build_assignments(config)
    assert (assignments["train_end"] < assignments["origin"]).all()
