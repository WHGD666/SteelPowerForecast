"""Independent final integrity verification for a generated submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.common.io_utils import sha256
from src.round2_v3.contracts import (
    LONG_HORIZONS_MINUTES,
    SHORT_HORIZONS_MINUTES,
    validate_prediction_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIABLE_SUBMISSION_STATUSES = {"ready_for_upload", "submitted"}


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_submission_status(status: str) -> None:
    if status not in VERIFIABLE_SUBMISSION_STATUSES:
        expected = ", ".join(sorted(VERIFIABLE_SUBMISSION_STATUSES))
        raise AssertionError(
            f"submission status {status!r} is not verifiable; expected one of: {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id")
    args = parser.parse_args()
    submission_dir = PROJECT_ROOT / "submissions" / "round2_v3" / args.submission_id
    manifest = json.loads(
        (submission_dir / "submission_manifest.json").read_text(encoding="utf-8")
    )
    validate_submission_status(manifest["status"])
    artifacts = manifest["artifacts"]
    short_path = PROJECT_ROOT / artifacts["s_result"]
    long_path = PROJECT_ROOT / artifacts["l_result"]
    archive_path = PROJECT_ROOT / artifacts["archive"]
    if sha256(short_path) != artifacts["s_result_sha256"]:
        raise AssertionError("s_result hash mismatch")
    if sha256(long_path) != artifacts["l_result_sha256"]:
        raise AssertionError("l_result hash mismatch")
    if sha256(archive_path) != artifacts["archive_sha256"]:
        raise AssertionError("archive hash mismatch")
    for record in manifest["training"]:
        model_path = PROJECT_ROOT / record["model"]
        if sha256(model_path) != record["model_sha256"]:
            raise AssertionError(f"model hash mismatch: {model_path}")

    origins = pd.date_range("2025-10-01 00:00:00", "2025-10-10 23:45:00", freq="15min")
    short = pd.read_csv(short_path)
    long = pd.read_csv(long_path)
    short_report = validate_prediction_frame(short, origins, SHORT_HORIZONS_MINUTES)
    long_report = validate_prediction_frame(long, origins, LONG_HORIZONS_MINUTES)
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != ["s_result.csv", "l_result.csv"]:
            raise AssertionError(f"unexpected ZIP members: {archive.namelist()}")
        if archive.testzip() is not None:
            raise AssertionError("ZIP CRC test failed")
        if bytes_sha256(archive.read("s_result.csv")) != artifacts["s_result_sha256"]:
            raise AssertionError("ZIP s_result bytes differ from manifest")
        if bytes_sha256(archive.read("l_result.csv")) != artifacts["l_result_sha256"]:
            raise AssertionError("ZIP l_result bytes differ from manifest")
    print(
        "PASS "
        f"submission_id={args.submission_id} archive_sha256={artifacts['archive_sha256']} "
        f"short={short_report['rows']}x{short_report['columns']} "
        f"long={long_report['rows']}x{long_report['columns']} "
        "models=4 zip_members=2"
    )


if __name__ == "__main__":
    main()
