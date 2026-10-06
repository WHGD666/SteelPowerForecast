"""Build and verify the first IronFlow preliminary persistence submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True, encoding="utf-8", errors="replace"
        ).strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}


def build_submission_input(prepared_input: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Restore cleaned current targets under official original names.

    The causal current-target features are removed because they would be exact
    duplicates of the restored raw-name columns in this persistence candidate.
    """
    if "datetime" not in prepared_input:
        raise AssertionError("prepared input is missing datetime")
    current_columns = [f"feat_current_{target}" for target in targets]
    missing = [column for column in current_columns if column not in prepared_input]
    if missing:
        raise AssertionError(f"prepared input is missing current target features: {missing}")

    raw_features = [
        column
        for column in prepared_input.columns
        if column != "datetime" and not column.startswith("feat_") and column not in targets
    ]
    engineered_features = [
        column
        for column in prepared_input.columns
        if column.startswith("feat_") and column not in current_columns
    ]
    output = prepared_input[["datetime", *raw_features]].copy()
    for target, current_column in zip(targets, current_columns):
        output[target] = prepared_input[current_column]
    for column in engineered_features:
        output[column] = prepared_input[column]
    return output


def build_persistence_predictions(
    submission_input: pd.DataFrame,
    targets: list[str],
    horizons_minutes: list[int],
) -> pd.DataFrame:
    output = submission_input[["datetime"]].copy()
    for target in targets:
        if target not in submission_input:
            raise AssertionError(f"submission input is missing target history column: {target}")
        current = pd.to_numeric(submission_input[target], errors="raise")
        for horizon in horizons_minutes:
            output[f"{target}_t+{horizon}_pred"] = current
    return output


def validate_submission(
    submission_input: pd.DataFrame,
    predictions: pd.DataFrame,
    expected_datetimes: pd.Series,
    targets: list[str],
    horizons_minutes: list[int],
) -> dict[str, Any]:
    expected_prediction_columns = ["datetime"] + [
        f"{target}_t+{horizon}_pred"
        for target in targets
        for horizon in horizons_minutes
    ]
    if list(predictions.columns) != expected_prediction_columns:
        raise AssertionError("s_result.csv column names or order differ from the frozen contract")
    if len(submission_input) != len(expected_datetimes) or len(predictions) != len(expected_datetimes):
        raise AssertionError("submission row count does not cover every test origin")
    if submission_input["datetime"].duplicated().any() or predictions["datetime"].duplicated().any():
        raise AssertionError("submission contains duplicate prediction origins")
    if not submission_input["datetime"].equals(expected_datetimes.reset_index(drop=True)):
        raise AssertionError("input.csv datetime order differs from prepared test origins")
    if not predictions["datetime"].equals(expected_datetimes.reset_index(drop=True)):
        raise AssertionError("s_result.csv datetime order differs from prepared test origins")
    if submission_input.isna().any().any() or predictions.isna().any().any():
        raise AssertionError("submission contains missing values")
    if not submission_input["datetime"].is_monotonic_increasing:
        raise AssertionError("submission origins are not increasing")
    feature_columns = [column for column in submission_input if column != "datetime"]
    unexpected_names = [
        column
        for column in feature_columns
        if column.startswith("feat") and not column.startswith("feat_")
    ]
    if unexpected_names:
        raise AssertionError(f"engineered feature names lack feat_ prefix: {unexpected_names}")
    prediction_values = predictions.drop(columns="datetime").to_numpy(dtype=float)
    if not np.isfinite(prediction_values).all():
        raise AssertionError("prediction output contains non-finite values")
    constant_columns = [
        column
        for column in feature_columns
        if submission_input[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        raise AssertionError(f"submission input contains invalid constant columns: {constant_columns}")
    return {
        "rows": len(predictions),
        "input_columns": len(submission_input.columns),
        "prediction_columns": len(predictions.columns),
        "duplicate_origins": 0,
        "missing_cells": 0,
        "constant_input_columns": [],
        "minimum_prediction": float(prediction_values.min()),
        "maximum_prediction": float(prediction_values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/submission_persistence_v1.yaml")
    )
    parser.add_argument("--team-name", help="Override the registered competition team name")
    parser.add_argument("--submission-id", help="Optional immutable output directory name")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    team_name = args.team_name or str(config["team_name"])
    forbidden_filename_characters = set('\\/:*?"<>|')
    if (
        not team_name.strip()
        or team_name != team_name.strip()
        or any(character in forbidden_filename_characters or ord(character) < 32 for character in team_name)
    ):
        raise ValueError("team name is empty or contains characters unsafe for a ZIP filename")
    targets = list(config["targets"])
    horizons = [int(value) for value in config["horizons_minutes"]]
    source_run_id = str(config["source_development_run_id"])
    source_summary_path = root / "outputs" / "baseline_runs" / source_run_id / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    ranking = source_summary["ranking_by_development_mape"]
    if not ranking or ranking[0]["model"] != config["selected_model"]:
        raise AssertionError("configured submission model is not the best recorded development model")
    if source_summary.get("sealed_holdout_evaluated") is not False:
        raise AssertionError("probe submission source must not consume the sealed holdout")

    prepared_dir = root / config["prepared_directory"]
    prepared_input_path = prepared_dir / config["prepared_input_file"]
    prepared_test_path = prepared_dir / config["prepared_test_file"]
    preparation_manifest_path = prepared_dir / config["preparation_manifest_file"]
    preparation_manifest = json.loads(preparation_manifest_path.read_text(encoding="utf-8"))
    for path in [prepared_input_path, prepared_test_path]:
        relative = path.relative_to(root).as_posix()
        expected_hash = preparation_manifest["outputs"][relative]
        if file_hash(path) != expected_hash:
            raise AssertionError(f"prepared artifact hash mismatch: {relative}")

    prepared_input = pd.read_csv(prepared_input_path, parse_dates=["datetime"], low_memory=False)
    prepared_test = pd.read_csv(prepared_test_path, parse_dates=["datetime"], low_memory=False)
    submission_input = build_submission_input(prepared_input, targets)
    predictions = build_persistence_predictions(submission_input, targets, horizons)
    validation = validate_submission(
        submission_input,
        predictions,
        prepared_test["datetime"],
        targets,
        horizons,
    )

    submission_id = args.submission_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + config["submission_version"]
    )
    output_dir = root / config["output"]["root"] / submission_id
    if output_dir.exists():
        raise FileExistsError(f"immutable submission directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    input_path = output_dir / config["output"]["input_file"]
    result_path = output_dir / config["output"]["short_result_file"]
    archive_path = output_dir / f"{team_name}_{config['output']['archive_suffix']}"
    decimals = int(config["prediction_policy"]["decimal_places"])
    submission_input.to_csv(
        input_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format=f"%.{decimals}f",
        lineterminator="\n",
    )
    predictions.to_csv(
        result_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format=f"%.{decimals}f",
        lineterminator="\n",
    )
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(input_path, arcname=input_path.name)
        archive.write(result_path, arcname=result_path.name)
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != ["input.csv", "s_result.csv"]:
            raise AssertionError(f"unexpected ZIP contents: {archive.namelist()}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise AssertionError(f"corrupt ZIP member: {bad_member}")

    manifest = {
        "submission_id": submission_id,
        "status": "verified",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_stage": config["competition_stage"],
        "team_name": team_name,
        "selected_model": config["selected_model"],
        "source_development_run_id": source_run_id,
        "source_development_mape": ranking[0]["mape"],
        "source_is_internal_validation_not_official_score": True,
        "sealed_holdout_evaluated": False,
        "git": git_state(root),
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_hash(config_path),
        },
        "sources": {
            prepared_input_path.relative_to(root).as_posix(): file_hash(prepared_input_path),
            prepared_test_path.relative_to(root).as_posix(): file_hash(prepared_test_path),
            preparation_manifest_path.relative_to(root).as_posix(): file_hash(
                preparation_manifest_path
            ),
            source_summary_path.relative_to(root).as_posix(): file_hash(source_summary_path),
        },
        "validation": validation,
        "package_members": ["input.csv", "s_result.csv"],
        "outputs": {
            input_path.name: file_hash(input_path),
            result_path.name: file_hash(result_path),
            archive_path.name: file_hash(archive_path),
        },
    }
    manifest_path = output_dir / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))
    print(f"PASS archive={archive_path}")
    print("NOTE official score is available only after competition-platform submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
