"""Build test-origin features with uninterrupted train-to-test history."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.common.io_utils import read_csv_strict, sha256, write_json
from src.round2_v3.build_origin_features import build_origin_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "round2_v3" / "features_v1.yaml"
BASE_DIR = PROJECT_ROOT / "data" / "prepared" / "round2_v3_causal_base_1"
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "prepared" / "round2_v3_test_features_with_history_1"
)
CENTRAL_MANIFEST = (
    PROJECT_ROOT / "manifests" / "round2_v3" / "test_features_with_history_v1.json"
)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {OUTPUT_DIR}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    train_path = BASE_DIR / "train_causal_base_1min.csv"
    test_path = BASE_DIR / "test_causal_base_1min.csv"
    train, train_encoding = read_csv_strict(train_path)
    test, test_encoding = read_csv_strict(test_path)
    if list(train.columns) != list(test.columns):
        raise AssertionError("train/test causal base schemas differ")
    train["datetime"] = pd.to_datetime(train["datetime"], errors="raise")
    test["datetime"] = pd.to_datetime(test["datetime"], errors="raise")
    if test["datetime"].min() - train["datetime"].max() != pd.Timedelta(minutes=1):
        raise AssertionError("train/test causal bases are not minute-contiguous")
    combined = pd.concat([train, test], ignore_index=True)
    if combined["datetime"].duplicated().any():
        raise AssertionError("combined causal base has duplicate timestamps")

    all_features, registry = build_origin_features(combined, config)
    test_start = test["datetime"].min()
    test_end = test["datetime"].max()
    test_features = all_features.loc[
        all_features["datetime"].between(test_start, test_end)
    ].reset_index(drop=True)
    expected_origins = test.loc[
        (test["datetime"].dt.minute % 15 == 0) & (test["datetime"].dt.second == 0),
        "datetime",
    ].reset_index(drop=True)
    if not test_features["datetime"].equals(expected_origins):
        raise AssertionError("test feature origins differ from official test origin grid")
    if len(test_features) != 960:
        raise AssertionError(f"expected 960 test origins, got {len(test_features)}")
    if not registry["causal"].all() or registry["source_offset_max_minutes"].max() > 0:
        raise AssertionError("inference feature registry is not causal")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output_path = OUTPUT_DIR / "test_origin_features_with_history.csv"
    registry_path = OUTPUT_DIR / "feature_registry.csv"
    test_features.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format=config["output"]["float_format"],
    )
    registry.to_csv(registry_path, index=False, encoding="utf-8")
    manifest = {
        "artifact_version": "round2_v3_test_features_with_history_1",
        "feature_config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "feature_config_sha256": sha256(CONFIG_PATH),
        "train_causal_base": str(train_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "train_causal_base_sha256": sha256(train_path),
        "test_causal_base": str(test_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_causal_base_sha256": sha256(test_path),
        "train_encoding": train_encoding,
        "test_encoding": test_encoding,
        "history_contiguous_across_boundary": True,
        "history_start": train["datetime"].min().isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "rows": int(len(test_features)),
        "columns": int(len(test_features.columns)),
        "feature_count": int(len(registry)),
        "maximum_source_time_offset_minutes": int(
            registry["source_offset_max_minutes"].max()
        ),
        "output": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "output_sha256": sha256(output_path),
        "registry": str(registry_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "registry_sha256": sha256(registry_path),
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    write_json(CENTRAL_MANIFEST, manifest)
    print(
        "PASS "
        f"features={manifest['feature_count']} max_source_offset="
        f"{manifest['maximum_source_time_offset_minutes']} rows={manifest['rows']} "
        f"history_contiguous=true output={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
