"""Create a causally aligned minute-level covariate base.

The output deliberately preserves missing values and audited raw magnitudes.
It never backfills, interpolates, uses targets or computes future-dependent
statistics.  Model features are built in a later, separately tested step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.common.io_utils import read_csv_strict, sha256, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_CONTRACT_PATH = (
    PROJECT_ROOT / "configs" / "round2_v3" / "feature_contract_v3.yaml"
)
CLEANING_CONFIG_PATH = PROJECT_ROOT / "configs" / "round2_v3" / "cleaning_v3.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "prepared" / "round2_v3_causal_base_1"
CENTRAL_MANIFEST = PROJECT_ROOT / "manifests" / "round2_v3" / "causal_base_v3.json"

FILES = {
    "train": {
        "gas": "Semi_gas.csv",
        "gas_user": "Semi_gas_user.csv",
        "gas_holder": "Semi_gas_holder.csv",
        "load_process": "Semi_load.csv",
    },
    "test": {
        "gas": "Semi_test_gas.csv",
        "gas_user": "Semi_test_gas_user.csv",
        "gas_holder": "Semi_test_gas_holder.csv",
        "load_process": "Semi_test_load.csv",
    },
}


def align_family(
    frame: pd.DataFrame,
    columns: list[str],
    full_index: pd.DatetimeIndex,
    family: str,
) -> pd.DataFrame:
    """Align one raw family to a complete minute grid without filling values."""
    required = {"datetime", *columns}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{family} missing required columns: {sorted(missing)}")
    local = frame[["datetime", *columns]].copy()
    local["datetime"] = pd.to_datetime(local["datetime"], errors="raise")
    if local["datetime"].duplicated().any():
        raise ValueError(f"{family} has duplicate datetime values")
    observed_timestamps = pd.DatetimeIndex(local["datetime"])
    local = local.sort_values("datetime").set_index("datetime").reindex(full_index)
    for column in columns:
        local[column] = pd.to_numeric(local[column], errors="coerce")
        local[f"{column}__missing"] = local[column].isna().astype("int8")
    local[f"timestamp_inserted__{family}"] = (
        ~full_index.isin(observed_timestamps)
    ).astype("int8")
    return local


def build_causal_partition(
    frames: dict[str, pd.DataFrame],
    allowed_by_family: dict[str, list[str]],
    partial_activation_columns: list[str],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> pd.DataFrame:
    """Merge raw families and add only current/past-derived state flags."""
    full_index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="1min")
    parts = [
        align_family(frames[family], columns, full_index, family)
        for family, columns in allowed_by_family.items()
    ]
    result = pd.concat(parts, axis=1)
    if result.columns.duplicated().any():
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"duplicate merged feature columns: {duplicates}")
    for column in partial_activation_columns:
        if column not in result:
            raise KeyError(f"partial activation column absent: {column}")
        # cummax is causal: rows before the first observation stay inactive.
        result[f"{column}__active"] = result[column].notna().cummax().astype("int8")
    result.index.name = "datetime"
    return result.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite versioned prepared data: {output_dir}"
        )

    feature_contract = yaml.safe_load(FEATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    cleaning_config = yaml.safe_load(CLEANING_CONFIG_PATH.read_text(encoding="utf-8"))
    if cleaning_config["covariate_missing"]["backfill_allowed"]:
        raise ValueError("v3 causal base forbids backfill")
    if cleaning_config["covariate_missing"]["maximum_forward_fill_minutes"] != 0:
        raise ValueError("baseline causal base expects zero forward fill")

    allowed = feature_contract["raw_covariates_allowed_at_or_before_origin"]
    partial = [item["column"] for item in feature_contract["partial_activation_columns"]]
    output_dir.mkdir(parents=True, exist_ok=False)
    partition_manifests = []

    for partition in ("train", "test"):
        raw_dir = PROJECT_ROOT / "data" / "raw" / partition
        frames: dict[str, pd.DataFrame] = {}
        source_records = []
        for family, filename in FILES[partition].items():
            source_path = raw_dir / filename
            frame, encoding = read_csv_strict(source_path)
            frames[family] = frame
            source_records.append(
                {
                    "family": family,
                    "path": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "encoding": encoding,
                    "sha256": sha256(source_path),
                }
            )

        load_times = pd.to_datetime(frames["load_process"]["datetime"], errors="raise")
        base = build_causal_partition(
            frames,
            allowed,
            partial,
            start=load_times.min(),
            end=load_times.max(),
        )
        output_csv = output_dir / f"{partition}_causal_base_1min.csv"
        base.to_csv(
            output_csv,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        inserted_columns = [
            column for column in base if column.startswith("timestamp_inserted__")
        ]
        raw_feature_columns = [column for columns in allowed.values() for column in columns]
        partition_manifests.append(
            {
                "partition": partition,
                "rows": int(len(base)),
                "columns": int(len(base.columns)),
                "datetime_min": base["datetime"].min().isoformat(),
                "datetime_max": base["datetime"].max().isoformat(),
                "output": str(output_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "output_sha256": sha256(output_csv),
                "source_files": source_records,
                "inserted_timestamp_counts": {
                    column: int(base[column].sum()) for column in inserted_columns
                },
                "raw_missing_counts_after_alignment": {
                    column: int(base[column].isna().sum()) for column in raw_feature_columns
                },
            }
        )

    manifest = {
        "artifact_version": "round2_v3_causal_base_1",
        "feature_contract_sha256": sha256(FEATURE_CONTRACT_PATH),
        "cleaning_config_sha256": sha256(CLEANING_CONFIG_PATH),
        "maximum_source_time_offset_minutes": 0,
        "target_columns_included": False,
        "backfill_used": False,
        "forward_fill_used": False,
        "interpolation_used": False,
        "outlier_replacement_used": False,
        "raw_values_preserved": True,
        "partitions": partition_manifests,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(CENTRAL_MANIFEST, manifest)
    print(
        "PASS "
        + " ".join(
            f"{item['partition']}_rows={item['rows']}" for item in partition_manifests
        )
        + f" output={output_dir}"
    )


if __name__ == "__main__":
    main()
