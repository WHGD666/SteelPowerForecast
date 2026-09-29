from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.common.io_utils import sha256


ROOT = Path(__file__).resolve().parents[2]


def test_prepared_label_manifest_matches_files_and_known_gaps() -> None:
    manifest_path = ROOT / "manifests" / "round2_v3" / "prepared_labels_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = ROOT / manifest["source"]
    output = ROOT / manifest["output"]
    assert sha256(source) == manifest["source_sha256"]
    assert sha256(output) == manifest["output_sha256"]
    assert manifest["rows"] == 14496
    assert manifest["complete_intervals"] == 14494
    assert manifest["incomplete_interval_starts"] == [
        "2025-08-27T18:00:00",
        "2025-09-09T08:15:00",
    ]


def test_split_assignment_hash_matches_frozen_manifest() -> None:
    manifest = json.loads(
        (ROOT / "manifests" / "round2_v3" / "splits_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assignment = ROOT / "manifests" / "round2_v3" / "validation_origins_v3.csv"
    assert sha256(assignment) == manifest["assignment_sha256"]
    assert manifest["fold_count"] == 6
    assert manifest["validation_origin_rows"] == 1536


def test_feature_contract_exactly_accounts_for_raw_non_target_columns() -> None:
    contract = yaml.safe_load(
        (ROOT / "configs" / "round2_v3" / "feature_contract_v3.yaml").read_text(
            encoding="utf-8"
        )
    )
    expected_by_family = {}
    paths = {
        "gas": ROOT / "data" / "raw" / "train" / "Semi_gas.csv",
        "gas_user": ROOT / "data" / "raw" / "train" / "Semi_gas_user.csv",
        "gas_holder": ROOT / "data" / "raw" / "train" / "Semi_gas_holder.csv",
        "load_process": ROOT / "data" / "raw" / "train" / "Semi_load.csv",
    }
    targets = set(contract["target_columns"])
    excluded = {item["column"] for item in contract["excluded_columns"]}
    for family, path in paths.items():
        columns = set(pd.read_csv(path, nrows=0).columns) - {"datetime"} - targets - excluded
        expected_by_family[family] = columns
    observed_by_family = {
        family: set(columns)
        for family, columns in contract["raw_covariates_allowed_at_or_before_origin"].items()
    }
    assert observed_by_family == expected_by_family
    flattened = [column for columns in observed_by_family.values() for column in columns]
    assert len(flattened) == len(set(flattened))
    assert "future_process_observations" in contract["forbidden_for_submission_models"]
