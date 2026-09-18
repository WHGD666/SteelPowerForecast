"""Build causally prepared IronFlow tables without modifying raw sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"cannot read {path}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def load_partition(root: Path, families: list[str], partition: str) -> tuple[pd.DataFrame, dict[str, str]]:
    merged: pd.DataFrame | None = None
    hashes: dict[str, str] = {}
    for family in families:
        name = f"Pre_{family}.csv" if partition == "train" else f"Pre_test_{family}.csv"
        path = root / ("data" if partition == "train" else "test") / name
        frame = read_csv(path)
        if "datetime" not in frame.columns:
            raise AssertionError(f"{path}: missing datetime")
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise", format="mixed")
        if frame["datetime"].duplicated().any():
            raise AssertionError(f"{path}: duplicate timestamp")
        if frame.duplicated().any():
            raise AssertionError(f"{path}: duplicate full row")
        if not frame["datetime"].is_monotonic_increasing:
            raise AssertionError(f"{path}: datetime not monotonic")
        hashes[path.relative_to(root).as_posix()] = file_hash(path)
        if merged is None:
            merged = frame
        else:
            overlap = (set(merged.columns) & set(frame.columns)) - {"datetime"}
            if overlap:
                raise AssertionError(f"{path}: conflicting columns {sorted(overlap)}")
            merged = merged.merge(frame, on="datetime", how="outer", validate="one_to_one", sort=True)
    assert merged is not None
    return merged.sort_values("datetime").reset_index(drop=True), hashes


def complete_grid(frame: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    observed = set(frame["datetime"])
    grid = pd.date_range(frame["datetime"].min(), frame["datetime"].max(), freq=frequency)
    inserted = [timestamp for timestamp in grid if timestamp not in observed]
    prepared = frame.set_index("datetime").reindex(grid)
    prepared.index.name = "datetime"
    return prepared.reset_index(), inserted


def prepare_partition(
    frame: pd.DataFrame,
    missing_flag_columns: list[str],
    excluded: list[str],
    targets: list[str],
    frequency: str,
    fill_limit: int,
    boundary_timestamp: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete, inserted = complete_grid(frame, frequency)
    complete = complete.drop(columns=[column for column in excluded if column in complete.columns])
    before = complete.copy()
    complete["feat_inserted_timestamp"] = complete["datetime"].isin(inserted).astype("int8")
    complete["feat_boundary_context_overlap"] = (complete["datetime"] == boundary_timestamp).astype("int8")

    for column in missing_flag_columns:
        if column in complete.columns:
            complete[f"feat_missing_{column}"] = complete[column].isna().astype("int8")

    base_columns = [column for column in complete.columns if column != "datetime" and not column.startswith("feat_")]
    covariates = [column for column in base_columns if column not in targets]
    for column in covariates:
        complete[column] = complete[column].ffill(limit=fill_limit)

    for target in targets:
        complete[f"feat_current_{target}"] = complete[target].ffill(limit=fill_limit)

    filled_counts = {
        column: int((before[column].isna() & complete[column].notna()).sum())
        for column in covariates
        if int((before[column].isna() & complete[column].notna()).sum()) > 0
    }
    for target in targets:
        current = f"feat_current_{target}"
        count = int((before[target].isna() & complete[current].notna()).sum())
        if count:
            filled_counts[current] = count

    residual_covariate_missing = {
        column: int(complete[column].isna().sum())
        for column in covariates
        if complete[column].isna().any()
    }
    feature_columns = sorted(column for column in complete.columns if column.startswith("feat_"))
    base_order = [column for column in before.columns if column not in excluded]
    complete = complete[base_order + feature_columns]
    details = {
        "rows_before_grid": int(len(frame)),
        "rows_after_grid": int(len(complete)),
        "inserted_timestamps": [timestamp.isoformat() for timestamp in inserted],
        "excluded_all_empty_columns": [column for column in excluded if column in frame.columns],
        "filled_counts": filled_counts,
        "residual_covariate_missing": residual_covariate_missing,
        "target_non_null_rows": {target: int(complete[target].notna().sum()) for target in targets},
        "feature_column_count": len(feature_columns),
    }
    return complete, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare IronFlow data using causal cleaning v1")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/cleaning_v1.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = root / config["source_policy"]["output_directory"]
    out.mkdir(parents=True, exist_ok=True)

    families = list(config["source_families"])
    targets = list(config["targets"])
    excluded = list(config["exclude_all_empty_columns"])
    frequency = str(config["time"]["frequency"])
    fill_limit = int(config["missing_policy"]["covariates"]["limit_steps"])
    boundary = pd.Timestamp(config["time"]["boundary_context_timestamp"])

    train_raw, train_hashes = load_partition(root, families, "train")
    test_raw, test_hashes = load_partition(root, families, "test")
    expected_columns = list(train_raw.columns)
    if expected_columns != list(test_raw.columns):
        raise AssertionError("train/test merged base column order differs")

    missing_flag_columns = [
        column
        for column in expected_columns
        if column != "datetime" and column not in excluded and (train_raw[column].isna().any() or test_raw[column].isna().any())
    ]
    # Grid insertion can make every retained value column missing.  These flags
    # must therefore exist consistently across train and test schemas.
    retained_value_columns = [column for column in expected_columns if column != "datetime" and column not in excluded]
    missing_flag_columns = sorted(set(missing_flag_columns) | set(retained_value_columns))

    train, train_details = prepare_partition(train_raw, missing_flag_columns, excluded, targets, frequency, fill_limit, boundary)
    test, test_details = prepare_partition(test_raw, missing_flag_columns, excluded, targets, frequency, fill_limit, boundary)
    if list(train.columns) != list(test.columns):
        raise AssertionError("prepared train/test schemas differ")
    if train["datetime"].duplicated().any() or test["datetime"].duplicated().any():
        raise AssertionError("prepared output has duplicate timestamps")
    if train_details["residual_covariate_missing"] or test_details["residual_covariate_missing"]:
        raise AssertionError("causal fill left missing covariates; review manifest before widening fill")

    train_path = out / config["output"]["train_file"]
    test_path = out / config["output"]["test_file"]
    input_path = out / config["output"]["input_file"]
    train.to_csv(train_path, index=False, encoding=config["output"]["encoding"], date_format="%Y-%m-%d %H:%M:%S")
    test.to_csv(test_path, index=False, encoding=config["output"]["encoding"], date_format="%Y-%m-%d %H:%M:%S")
    input_frame = test.drop(columns=targets)
    diagnostic_exclusions = [
        column
        for column in config["output"].get("input_exclude_diagnostic_columns", [])
        if column in input_frame.columns
    ]
    input_frame = input_frame.drop(columns=diagnostic_exclusions)
    input_constant_columns = [
        column
        for column in input_frame.columns
        if column != "datetime" and input_frame[column].nunique(dropna=False) <= 1
    ]
    input_frame = input_frame.drop(columns=input_constant_columns)
    if input_frame.isna().any().any():
        missing = {column: int(value) for column, value in input_frame.isna().sum().items() if value}
        raise AssertionError(f"input.csv contains missing values: {missing}")
    input_frame.to_csv(input_path, index=False, encoding=config["output"]["encoding"], date_format="%Y-%m-%d %H:%M:%S")
    manifest = {
        "cleaning_version": config["cleaning_version"],
        "config": config_path.relative_to(root).as_posix(),
        "source_hashes": {**train_hashes, **test_hashes},
        "policy": {
            "frequency": frequency,
            "causal_forward_fill_limit_steps": fill_limit,
            "target_raw_values_filled": False,
            "outlier_values_mutated": False,
            "backfill_used": False,
        },
        "train": train_details,
        "test": test_details,
        "columns": list(train.columns),
        "input_feature_columns": [column for column in input_frame.columns if column != "datetime"],
        "input_excluded_constant_columns": input_constant_columns,
        "input_excluded_diagnostic_columns": diagnostic_exclusions,
        "outputs": {
            train_path.relative_to(root).as_posix(): file_hash(train_path),
            test_path.relative_to(root).as_posix(): file_hash(test_path),
            input_path.relative_to(root).as_posix(): file_hash(input_path),
        },
    }
    manifest_path = out / config["output"]["manifest_file"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_safe), encoding="utf-8")
    print(json.dumps({"status": "PASS", "out": out.as_posix(), "train_rows": len(train), "test_rows": len(test), "prepared_columns": len(train.columns), "input_columns": len(input_frame.columns), "manifest": manifest_path.as_posix()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
