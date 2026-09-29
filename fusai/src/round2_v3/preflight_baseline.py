"""Fast, no-training preflight for the formal v3 baseline command."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd
import yaml

from src.round2_v3.baseline_utils import (
    build_supervised_long,
    filter_origins_by_stride,
    select_feature_columns,
)
from src.round2_v3.run_baseline import PROJECT_ROOT, _validate_inputs


CONFIG_PATH = PROJECT_ROOT / "configs" / "round2_v3" / "baseline_v1.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = {key: PROJECT_ROOT / value for key, value in config["inputs"].items()}
    manifests = _validate_inputs(config, paths)
    registry = pd.read_csv(paths["feature_registry"])
    selected = select_feature_columns(registry, config["feature_selection"])
    feature_path = PROJECT_ROOT / manifests["feature"]["partitions"][0]["output"]
    feature_header = pd.read_csv(feature_path, nrows=0).columns.tolist()
    missing = set(selected) - set(feature_header)
    if missing:
        raise AssertionError(f"selected features absent from artifact: {sorted(missing)}")
    if {"generator_1", "generator_all"} & set(feature_header):
        raise AssertionError("target columns leaked into origin features")

    features = pd.read_csv(feature_path, nrows=3, low_memory=False)
    labels = pd.read_csv(PROJECT_ROOT / manifests["label"]["output"])
    splits = pd.read_csv(paths["split_assignments"])
    model_splits = splits.loc[splits["role"].isin(config["fold_roles"])].copy()
    model_splits["validation_origin"] = pd.to_datetime(
        model_splits["validation_origin"], errors="raise"
    )
    label_index = labels.copy()
    label_index["datetime"] = pd.to_datetime(label_index["datetime"], errors="raise")
    label_index = label_index.set_index("datetime")

    validation_rows = 0
    training_rows_upper_bound = 0
    for _, fold in model_splits.groupby("fold_id", sort=False):
        val_origins = pd.DatetimeIndex(fold["validation_origin"])
        latest_train_origin = pd.Timestamp(fold["latest_eligible_train_origin"].iloc[0])
        all_train_origins = pd.date_range(
            manifests["feature"]["partitions"][0]["datetime_min"],
            latest_train_origin,
            freq="15min",
        )
        for period in config["periods"].values():
            horizons = period["horizons_minutes"]
            interval_starts = pd.DatetimeIndex(
                [origin + pd.Timedelta(minutes=horizon - 15) for origin in val_origins for horizon in horizons]
            )
            actual = label_index.reindex(interval_starts)[config["targets"]]
            if actual.isna().any().any():
                raise AssertionError("validation target matrix contains missing labels")
            validation_rows += len(interval_starts) * len(config["targets"])
            training_origins = filter_origins_by_stride(
                all_train_origins, period["training_origin_stride_minutes"]
            )
            training_rows_upper_bound += (
                len(training_origins) * len(horizons) * len(config["targets"])
            )

    # Exercise the exact long-format path on a tiny slice without fitting.
    tiny_origins = pd.DatetimeIndex(pd.to_datetime(features["datetime"].iloc[:2]))
    build_supervised_long(
        features,
        labels,
        tiny_origins,
        [15, 30],
        config["targets"][0],
        selected,
    )
    print(
        "PASS "
        f"folds={model_splits['fold_id'].nunique()} selected_origin_features={len(selected)} "
        f"model_features={len(selected) + 11} validation_cells={validation_rows} "
        f"training_rows_upper_bound={training_rows_upper_bound} lightgbm={lgb.__version__}"
    )


if __name__ == "__main__":
    main()
