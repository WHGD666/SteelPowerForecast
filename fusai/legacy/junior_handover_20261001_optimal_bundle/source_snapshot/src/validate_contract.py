"""Validate the frozen IronFlow information and submission contract.

This validator is intentionally independent of model code.  It checks raw
schema/timestamps and, when supplied, input.csv and s_result.csv.  It never
changes source data or fills missing values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_config import load_contract_config  # noqa: E402


TARGETS = ["generator_1", "generator_all"]
ORIGINAL_FAMILIES = ["gas", "gas_holder", "gas_user", "load"]


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"cannot read {path}")


def parse_datetime(frame: pd.DataFrame) -> pd.Series:
    if "datetime" not in frame.columns:
        raise AssertionError("missing datetime column")
    return pd.to_datetime(frame["datetime"], errors="coerce", format="mixed")


def check_time(
    frame: pd.DataFrame,
    label: str,
    expected_count: int | None = None,
    step_seconds: int = 900,
) -> list[str]:
    errors: list[str] = []
    times = parse_datetime(frame)
    if times.isna().any():
        errors.append(f"{label}: invalid datetime count={int(times.isna().sum())}")
    valid = times.dropna()
    if valid.duplicated().any():
        errors.append(f"{label}: duplicate datetime count={int(valid.duplicated(keep=False).sum())}")
    if not valid.is_monotonic_increasing:
        errors.append(f"{label}: datetime is not monotonic increasing")
    if len(valid) > 1:
        delta = valid.diff().dropna().dt.total_seconds()
        if (delta != step_seconds).any():
            errors.append(
                f"{label}: non-{step_seconds // 60}-minute intervals={int((delta != step_seconds).sum())}"
            )
    if expected_count is not None and len(frame) != expected_count:
        errors.append(f"{label}: expected rows={expected_count}, observed={len(frame)}")
    return errors


def expected_short_columns(offsets: list[int]) -> list[str]:
    columns = ["datetime"]
    for target in TARGETS:
        columns.extend(f"{target}_t+{offset}_pred" for offset in offsets)
    return columns


def validate_raw(root: Path, contract: dict[str, object]) -> tuple[list[str], list[str], dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {}
    train_frames: dict[str, pd.DataFrame] = {}
    test_frames: dict[str, pd.DataFrame] = {}
    train_prefix = str(contract["data_prefix"])
    test_prefix = str(contract["test_data_prefix"])
    raw_sampling = int(contract.get("raw_sampling_minutes", contract["sampling_minutes"]))
    step_seconds = raw_sampling * 60
    gap_marker = f"non-{step_seconds // 60}-minute intervals"
    expected_origin_rows = contract.get("expected_origin_rows")
    for family in ORIGINAL_FAMILIES:
        train_path = root / "data" / f"{train_prefix}_{family}.csv"
        test_path = root / "test" / f"{test_prefix}_{family}.csv"
        train = read_csv(train_path)
        test = read_csv(test_path)
        train_frames[family] = train
        test_frames[family] = test
        train_issues = check_time(train, f"data/{train_prefix}_{family}.csv", step_seconds=step_seconds)
        for issue in train_issues:
            if gap_marker in issue:
                warnings.append(issue + " (known source gap; must be handled fold-locally)")
            else:
                errors.append(issue)
        errors.extend(
            check_time(
                test,
                f"test/{test_prefix}_{family}.csv",
                expected_origin_rows,
                step_seconds=step_seconds,
            )
        )
        if list(train.columns) != list(test.columns):
            errors.append(f"{family}: train/test column order differs")
        details[family] = {"train_rows": len(train), "test_rows": len(test), "columns": list(train.columns)}

    test_origins = parse_datetime(test_frames["gas"])
    origin_sets = {family: set(parse_datetime(frame).dropna()) for family, frame in test_frames.items()}
    common_origins = set.intersection(*origin_sets.values())
    if expected_origin_rows is not None and len(common_origins) != int(expected_origin_rows):
        errors.append("test: four family origin sets are not identical")
    elif len(common_origins) != len(test_origins):
        errors.append("test: four family origin sets are not identical")
    details["test_origin_count"] = len(test_origins)
    details["test_origin_min"] = test_origins.min().isoformat()
    details["test_origin_max"] = test_origins.max().isoformat()
    details["contract_stage"] = contract.get("stage")
    details["contract_config"] = contract.get("source_path")

    target_presence = {
        target: int(test_frames["load"][target].notna().sum())
        for target in TARGETS
        if target in test_frames["load"].columns
    }
    if set(target_presence) != set(TARGETS):
        warnings.append(f"test/{test_prefix}_load.csv does not expose both current-origin target columns")
    else:
        warnings.append("current-origin target observations are present; validator treats only row t as visible")
    details["current_origin_target_non_null"] = target_presence
    return errors, warnings, details


def validate_submission(
    root: Path,
    input_path: Path | None,
    result_path: Path | None,
    origins: pd.Series,
    contract: dict[str, object],
) -> tuple[list[str], list[str], dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {}
    train_prefix = str(contract["data_prefix"])
    step_seconds = int(contract["sampling_minutes"]) * 60
    offsets = [int(value) for value in contract["result_offsets_minutes"]]
    if input_path is not None:
        frame = read_csv(input_path)
        errors.extend(check_time(frame, "input.csv", len(origins), step_seconds=step_seconds))
        if list(parse_datetime(frame)) != list(origins):
            errors.append("input.csv: datetime origins do not exactly match test origins/order")
        missing = {column: int(value) for column, value in frame.isna().sum().items() if value}
        if missing:
            errors.append(f"input.csv: missing values are not allowed in submitted features: {missing}")
        constant = [column for column in frame.columns if column != "datetime" and frame[column].nunique(dropna=False) <= 1]
        if constant:
            errors.append(f"input.csv: constant non-time columns are not allowed: {constant}")
        extras = [column for column in frame.columns if column not in {"datetime"}]
        raw_columns = set().union(
            *[
                set(pd.read_csv(root / "data" / f"{train_prefix}_{family}.csv", nrows=0).columns)
                for family in ORIGINAL_FAMILIES
            ]
        )
        bad_extra = [column for column in extras if column not in raw_columns and not str(column).startswith("feat_")]
        if bad_extra:
            errors.append(f"input.csv: non-original columns without feat_ prefix: {bad_extra}")
        details["input_columns"] = list(frame.columns)
    if result_path is not None:
        frame = read_csv(result_path)
        expected = expected_short_columns(offsets)
        if list(frame.columns) != expected:
            errors.append(f"s_result.csv: expected exact columns {expected}, observed {list(frame.columns)}")
        errors.extend(check_time(frame, "s_result.csv", len(origins), step_seconds=step_seconds))
        if list(parse_datetime(frame)) != list(origins):
            errors.append("s_result.csv: datetime origins do not exactly match test origins/order")
        for column in expected[1:]:
            if column in frame and not pd.api.types.is_numeric_dtype(frame[column]):
                errors.append(f"s_result.csv: prediction column is not numeric: {column}")
        details["result_columns"] = list(frame.columns)
    return errors, warnings, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate IronFlow frozen rolling contract")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument(
        "--contract-config",
        type=Path,
        default=None,
        help="semi contract config (e.g. configs/protocol_v2.yaml); omit to keep the frozen v1 contract",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = args.contract_config
    if contract_path is not None and not contract_path.is_absolute():
        contract_path = root / contract_path
    contract = load_contract_config(contract_path)
    errors, warnings, details = validate_raw(root, contract)
    test_prefix = str(contract["test_data_prefix"])
    raw_origins = parse_datetime(read_csv(root / "test" / f"{test_prefix}_gas.csv"))
    step_minutes = int(contract["sampling_minutes"])
    if step_minutes > 1:
        origins = raw_origins[raw_origins.dt.minute % step_minutes == 0].reset_index(drop=True)
    else:
        origins = raw_origins.reset_index(drop=True)
    submission_errors, submission_warnings, submission_details = validate_submission(
        root, args.input, args.result, origins, contract
    )
    errors.extend(submission_errors)
    warnings.extend(submission_warnings)
    details.update(submission_details)
    status = "PASS" if not errors else "FAIL"
    result = {"status": status, "errors": errors, "warnings": warnings, "details": details}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
