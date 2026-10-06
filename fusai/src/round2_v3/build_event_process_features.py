"""Build target-free causal process dynamics for the event-signal ablation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

from src.common.io_utils import read_csv_strict, sha256, write_json
from src.round2_v3.audit_step_events import (
    AIR_HEATERS,
    BF_PRODUCTION,
    BF_USERS,
    CONVERTER_USERS,
    GENERATOR_FUELS,
    MIXED_GAS,
    _quarter_hour_grid,
    available_in_formal_test,
    build_causal_transforms,
    build_observed_signals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "round2_v3" / "event_process_features_v1.yaml"
)
BASE_DIR = PROJECT_ROOT / "data" / "prepared" / "round2_v3_causal_base_1"
BASE_MANIFEST = BASE_DIR / "manifest.json"
CENTRAL_MANIFEST = (
    PROJECT_ROOT / "manifests" / "round2_v3" / "event_process_features_v1.json"
)
CENTRAL_REGISTRY = (
    PROJECT_ROOT / "manifests" / "round2_v3" / "event_process_feature_registry_v1.csv"
)


def required_covariates() -> list[str]:
    return sorted(
        set(
            BF_PRODUCTION
            + AIR_HEATERS
            + BF_USERS
            + CONVERTER_USERS
            + GENERATOR_FUELS
            + MIXED_GAS
            + ["coke_oven_1", "converter_1", "blast_furnace_gas_holder_2"]
        )
    )


def feature_registry(feature_names: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for name in feature_names:
        if name.endswith("__level"):
            base_signal = name.removesuffix("__level")
            transform = "level"
            minimum_offset = 0
        elif match := re.search(r"__(relative_)?delta_(\d+)h$", name):
            hours = int(match.group(2))
            base_signal = name[: match.start()]
            transform = match.group(0).removeprefix("__")
            minimum_offset = -(2 * hours * 60 - 15)
        elif match := re.search(r"__std_(\d+)h$", name):
            hours = int(match.group(1))
            base_signal = name[: match.start()]
            transform = match.group(0).removeprefix("__")
            minimum_offset = -(hours * 60 - 15)
        elif match := re.search(r"__level_vs_(\d+)h$", name):
            hours = int(match.group(1))
            base_signal = name[: match.start()]
            transform = match.group(0).removeprefix("__")
            minimum_offset = -(hours * 60 - 15)
        else:
            raise ValueError(f"unrecognized event feature name: {name}")
        records.append(
            {
                "feature_name": name,
                "base_signal": base_signal,
                "transform": transform,
                "source_offset_min_minutes": minimum_offset,
                "source_offset_max_minutes": 0,
                "causal": True,
                "target_history": False,
                "available_in_formal_test": available_in_formal_test(name),
            }
        )
    registry = pd.DataFrame.from_records(records)
    if registry["feature_name"].duplicated().any():
        raise AssertionError("event process feature names are not unique")
    if not registry["causal"].all() or registry["target_history"].any():
        raise AssertionError("event process registry violates feature contract")
    if not registry["available_in_formal_test"].all():
        raise AssertionError("event process artifact contains unavailable target history")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / config["output"]["directory"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {output_dir}")

    covariates = required_covariates()
    frames: dict[str, pd.DataFrame] = {}
    encodings: dict[str, str] = {}
    source_paths: dict[str, Path] = {}
    for partition in ("train", "test"):
        path = BASE_DIR / f"{partition}_causal_base_1min.csv"
        frame, encoding = read_csv_strict(
            path, usecols=["datetime", *covariates]
        )
        frames[partition] = frame
        encodings[partition] = encoding
        source_paths[partition] = path

    combined = pd.concat([frames["train"], frames["test"]], ignore_index=True)
    grid = _quarter_hour_grid(combined, covariates)
    signals = build_observed_signals(grid, interval_targets=None)
    history = config["history"]
    features = build_causal_transforms(
        signals,
        history["delta_hours"],
        history["volatility_hours"],
        int(history["level_baseline_hours"]),
    )
    unavailable = [name for name in features.columns if not available_in_formal_test(name)]
    if unavailable:
        raise AssertionError(f"target-history features unexpectedly present: {unavailable}")
    registry = feature_registry(list(features.columns))

    base_manifest = yaml.safe_load(BASE_MANIFEST.read_text(encoding="utf-8"))
    partition_bounds = {
        item["partition"]: (
            pd.Timestamp(item["datetime_min"]),
            pd.Timestamp(item["datetime_max"]),
        )
        for item in base_manifest["partitions"]
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    partition_records: list[dict[str, object]] = []
    float_format = config["output"]["float_format"]
    for partition in ("train", "test"):
        lower, upper = partition_bounds[partition]
        part = features.loc[(features.index >= lower) & (features.index <= upper)].copy()
        part.index.name = "datetime"
        part = part.reset_index()
        expected_rows = int(
            next(
                item["rows"] for item in base_manifest["partitions"]
                if item["partition"] == partition
            )
            / 15
        )
        if len(part) != expected_rows:
            raise AssertionError(
                f"{partition} origin rows {len(part)} != expected {expected_rows}"
            )
        output_path = output_dir / config["output"][f"{partition}_file"]
        part.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format=float_format,
        )
        partition_records.append(
            {
                "partition": partition,
                "source": source_paths[partition].relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": sha256(source_paths[partition]),
                "source_encoding": encodings[partition],
                "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
                "output_sha256": sha256(output_path),
                "rows": int(len(part)),
                "columns": int(len(part.columns)),
                "datetime_min": part["datetime"].min().isoformat(),
                "datetime_max": part["datetime"].max().isoformat(),
            }
        )

    registry_path = output_dir / config["output"]["registry_file"]
    registry.to_csv(registry_path, index=False, encoding="utf-8")
    registry.to_csv(CENTRAL_REGISTRY, index=False, encoding="utf-8")
    manifest = {
        "artifact_version": config["feature_version"],
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "base_manifest": BASE_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "base_manifest_sha256": sha256(BASE_MANIFEST),
        "feature_count": int(len(registry)),
        "all_features_causal": True,
        "maximum_source_time_offset_minutes": 0,
        "minimum_source_time_offset_minutes": int(
            registry["source_offset_min_minutes"].min()
        ),
        "target_history_included": False,
        "future_observations_included": False,
        "train_test_history_continuity": True,
        "registry": registry_path.relative_to(PROJECT_ROOT).as_posix(),
        "registry_sha256": sha256(registry_path),
        "partitions": partition_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(CENTRAL_MANIFEST, manifest)
    print(
        "PASS "
        f"features={manifest['feature_count']} "
        f"min_source_offset={manifest['minimum_source_time_offset_minutes']} "
        + " ".join(
            f"{item['partition']}_rows={item['rows']}" for item in partition_records
        )
        + f" output={output_dir}"
    )


if __name__ == "__main__":
    main()
