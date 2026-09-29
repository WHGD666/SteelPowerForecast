from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def test_all_eight_official_csv_files_exist() -> None:
    expected = {
        "train/Semi_gas.csv",
        "train/Semi_gas_holder.csv",
        "train/Semi_gas_user.csv",
        "train/Semi_load.csv",
        "test/Semi_test_gas.csv",
        "test/Semi_test_gas_holder.csv",
        "test/Semi_test_gas_user.csv",
        "test/Semi_test_load.csv",
    }
    observed = {path.relative_to(RAW).as_posix() for path in RAW.rglob("*.csv")}
    assert expected <= observed


def test_train_test_time_boundary_and_test_origin_count() -> None:
    train = pd.read_csv(RAW / "train" / "Semi_load.csv", usecols=["datetime"])
    test = pd.read_csv(RAW / "test" / "Semi_test_load.csv", usecols=["datetime"])
    train_times = pd.to_datetime(train["datetime"], errors="raise")
    test_times = pd.to_datetime(test["datetime"], errors="raise")
    assert train_times.max() == pd.Timestamp("2025-09-30 23:59:00")
    assert test_times.min() == pd.Timestamp("2025-10-01 00:00:00")
    assert test_times.min() - train_times.max() == pd.Timedelta(minutes=1)
    origins = test_times[(test_times.dt.minute % 15 == 0) & (test_times.dt.second == 0)]
    assert len(origins) == 960
    assert origins.iloc[0] == pd.Timestamp("2025-10-01 00:00:00")
    assert origins.iloc[-1] == pd.Timestamp("2025-10-10 23:45:00")


def test_test_targets_are_completely_hidden() -> None:
    frame = pd.read_csv(
        RAW / "test" / "Semi_test_load.csv",
        usecols=["generator_1", "generator_all"],
    )
    assert frame["generator_1"].notna().sum() == 0
    assert frame["generator_all"].notna().sum() == 0
