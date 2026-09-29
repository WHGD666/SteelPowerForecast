"""No-training preflight for the hierarchical target experiment."""

from __future__ import annotations

import json

import pandas as pd
import yaml

from src.common.io_utils import sha256
from src.round2_v3.baseline_utils import filter_origins_by_stride, select_feature_columns
from src.round2_v3.run_baseline import PROJECT_ROOT, _validate_inputs


CONFIG_PATH = PROJECT_ROOT / "configs" / "round2_v3" / "hierarchical_v1.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    control_config_path = PROJECT_ROOT / config["control_config"]
    control = yaml.safe_load(control_config_path.read_text(encoding="utf-8"))
    paths = {key: PROJECT_ROOT / value for key, value in control["inputs"].items()}
    manifests = _validate_inputs(control, paths)
    control_dir = PROJECT_ROOT / control["output"]["root"] / config["control_run_id"]
    control_manifest = json.loads(
        (control_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if control_manifest["status"] != "completed":
        raise AssertionError("control run is not completed")
    control_oof = PROJECT_ROOT / control_manifest["artifacts"]["oof_predictions"]
    if sha256(control_oof) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("control OOF hash mismatch")

    labels = pd.read_csv(PROJECT_ROOT / manifests["label"]["output"])
    large = labels["generator_all"] - labels["generator_1"]
    if (large.dropna() < 0).any():
        raise AssertionError("large-unit label is negative")
    registry = pd.read_csv(paths["feature_registry"])
    selected = select_feature_columns(registry, control["feature_selection"])
    splits = pd.read_csv(paths["split_assignments"])
    splits = splits.loc[splits["role"].isin(control["fold_roles"])].copy()
    training_rows_upper_bound = 0
    for _, fold in splits.groupby("fold_id", sort=False):
        start = pd.Timestamp(manifests["feature"]["partitions"][0]["datetime_min"])
        latest = pd.Timestamp(fold["latest_eligible_train_origin"].iloc[0])
        origins = pd.date_range(start, latest, freq="15min")
        for period in control["periods"].values():
            sampled = filter_origins_by_stride(
                origins, period["training_origin_stride_minutes"]
            )
            training_rows_upper_bound += len(sampled) * len(period["horizons_minutes"])
    print(
        "PASS "
        f"control_run={config['control_run_id']} folds={splits['fold_id'].nunique()} "
        f"models={splits['fold_id'].nunique() * len(control['periods'])} "
        f"selected_origin_features={len(selected)} "
        f"training_rows_upper_bound={training_rows_upper_bound} "
        f"large_label_min={large.min():.6f}"
    )


if __name__ == "__main__":
    main()
