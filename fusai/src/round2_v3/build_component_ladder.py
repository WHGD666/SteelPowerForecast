"""Build controlled v29b/v34 component-blend candidates without training.

The platform returns only an aggregate score.  This builder therefore freezes
all v29b ``generator_all`` predictions and changes only ``generator_1``.  Short
and long weights are independent so each platform comparison changes one
declared block at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.contracts import (
    LONG_HORIZONS_MINUTES,
    SHORT_HORIZONS_MINUTES,
    validate_prediction_frame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "component_ladder_v1.yaml"
)
RUNNER_SOURCE = Path(__file__).resolve()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _validate_weight(weight: float, allowed: list[float]) -> float:
    value = float(weight)
    if not any(np.isclose(value, candidate, rtol=0.0, atol=1e-12) for candidate in allowed):
        raise ValueError(f"weight {value} is not in frozen allowed_weights={allowed}")
    return value


def _find_member(archive: zipfile.ZipFile, prefix: str, filename: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if name.startswith(prefix) and name.endswith("/" + filename)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {filename!r} member with prefix {prefix!r}, got {matches}"
        )
    return matches[0]


def load_component(
    archive: zipfile.ZipFile,
    component: dict[str, Any],
) -> dict[str, tuple[pd.DataFrame, bytes]]:
    loaded: dict[str, tuple[pd.DataFrame, bytes]] = {}
    for filename, hash_key in (
        ("s_result.csv", "s_result_sha256"),
        ("l_result.csv", "l_result_sha256"),
    ):
        member = _find_member(archive, str(component["member_prefix"]), filename)
        payload = archive.read(member)
        observed_hash = bytes_sha256(payload)
        if observed_hash != component[hash_key]:
            raise AssertionError(
                f"source hash mismatch for {member}: {observed_hash} != {component[hash_key]}"
            )
        frame = pd.read_csv(io.BytesIO(payload), low_memory=False)
        loaded[filename] = (frame, payload)
    return loaded


def blend_generator_1(
    base: pd.DataFrame,
    alternate: pd.DataFrame,
    weight: float,
    decimals: int = 6,
) -> pd.DataFrame:
    if list(base.columns) != list(alternate.columns):
        raise AssertionError("base and alternate prediction columns differ")
    if not base["datetime"].equals(alternate["datetime"]):
        raise AssertionError("base and alternate prediction origins differ")
    result = base.copy()
    if np.isclose(weight, 0.0, rtol=0.0, atol=1e-12):
        return result
    g1_columns = [column for column in base if column.startswith("generator_1_")]
    base_values = base[g1_columns].to_numpy(float)
    alternate_values = alternate[g1_columns].to_numpy(float)
    blended = (1.0 - weight) * base_values + weight * alternate_values
    result[g1_columns] = np.round(blended, decimals)
    return result


def frame_bytes(frame: pd.DataFrame, float_format: str) -> bytes:
    return frame.to_csv(
        index=False,
        encoding="utf-8",
        float_format=float_format,
        lineterminator="\n",
    ).encode("utf-8")


def write_deterministic_zip(
    path: Path,
    members: list[tuple[str, bytes]],
    timestamp: tuple[int, int, int, int, int, int],
) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def block_change_report(
    candidate: pd.DataFrame,
    base: pd.DataFrame,
    alternate: pd.DataFrame,
    weight: float,
) -> dict[str, Any]:
    g1_columns = [column for column in base if column.startswith("generator_1_")]
    gall_columns = [column for column in base if column.startswith("generator_all_")]
    candidate_g1 = candidate[g1_columns].to_numpy(float)
    base_g1 = base[g1_columns].to_numpy(float)
    alternate_g1 = alternate[g1_columns].to_numpy(float)
    candidate_gall = candidate[gall_columns].to_numpy(float)
    base_gall = base[gall_columns].to_numpy(float)
    if not np.array_equal(candidate_gall, base_gall):
        raise AssertionError("frozen generator_all block changed")
    if np.isclose(weight, 0.0, rtol=0.0, atol=1e-12) and not np.array_equal(
        candidate_g1, base_g1
    ):
        raise AssertionError("zero-weight generator_1 differs from base")
    if np.isclose(weight, 1.0, rtol=0.0, atol=1e-12) and not np.array_equal(
        candidate_g1, alternate_g1
    ):
        raise AssertionError("unit-weight generator_1 differs from alternate")
    delta = np.abs(candidate_g1 - base_g1)
    return {
        "generator_all_exactly_frozen": True,
        "generator_1_cells": int(delta.size),
        "generator_1_cells_changed_vs_base": int(np.count_nonzero(delta)),
        "generator_1_mean_abs_change_vs_base": float(delta.mean()),
        "generator_1_max_abs_change_vs_base": float(delta.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--short-g1-v34-weight", type=float, required=True)
    parser.add_argument("--long-g1-v34-weight", type=float, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blend_contract = config["blend_contract"]
    allowed = [float(value) for value in blend_contract["allowed_weights"]]
    short_weight = _validate_weight(args.short_g1_v34_weight, allowed)
    long_weight = _validate_weight(args.long_g1_v34_weight, allowed)
    decimals = int(blend_contract["round_decimals"])

    bundle_path = PROJECT_ROOT / config["source_bundle"]["path"]
    if sha256(bundle_path) != config["source_bundle"]["sha256"]:
        raise AssertionError("source bundle SHA-256 mismatch")

    components = config["source_bundle"]["components"]
    base_name = str(blend_contract["base_component"])
    alternate_name = str(blend_contract["alternate_component"])
    with zipfile.ZipFile(bundle_path, "r") as archive:
        if archive.testzip() is not None:
            raise AssertionError("source bundle CRC check failed")
        base = load_component(archive, components[base_name])
        alternate = load_component(archive, components[alternate_name])

    origins = pd.date_range("2025-10-01 00:00:00", "2025-10-10 23:45:00", freq="15min")
    candidates: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}
    for filename, horizons, weight in (
        ("s_result.csv", SHORT_HORIZONS_MINUTES, short_weight),
        ("l_result.csv", LONG_HORIZONS_MINUTES, long_weight),
    ):
        base_frame = base[filename][0]
        alternate_frame = alternate[filename][0]
        candidate = blend_generator_1(base_frame, alternate_frame, weight, decimals)
        contract_report = validate_prediction_frame(candidate, origins, horizons)
        reports[filename] = {
            "contract": contract_report,
            "change": block_change_report(
                candidate, base_frame, alternate_frame, weight
            ),
        }
        candidates[filename] = candidate

    config_hash = sha256(config_path)[:10]
    weight_tag = f"sg1w{round(short_weight * 100):03d}_lg1w{round(long_weight * 100):03d}"
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ") + f"_component_ladder_{weight_tag}_{config_hash}"
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate run: {output_dir}")
    payload_dir = output_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=False)

    output_bytes: dict[str, bytes] = {}
    for filename, weight in (
        ("s_result.csv", short_weight),
        ("l_result.csv", long_weight),
    ):
        if np.isclose(weight, 0.0, rtol=0.0, atol=1e-12):
            payload = base[filename][1]
        else:
            payload = frame_bytes(candidates[filename], config["output"]["float_format"])
        output_bytes[filename] = payload
        (payload_dir / filename).write_bytes(payload)

    archive_path = output_dir / config["output"]["archive_filename"]
    timestamp = tuple(int(value) for value in config["output"]["zip_timestamp"])
    if len(timestamp) != 6:
        raise ValueError("zip_timestamp must contain six integers")
    write_deterministic_zip(
        archive_path,
        [("s_result.csv", output_bytes["s_result.csv"]), ("l_result.csv", output_bytes["l_result.csv"])],
        timestamp,  # type: ignore[arg-type]
    )
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.namelist() != ["s_result.csv", "l_result.csv"]:
            raise AssertionError("candidate ZIP member order differs from contract")
        if archive.testzip() is not None:
            raise AssertionError("candidate ZIP CRC check failed")
        for filename in ("s_result.csv", "l_result.csv"):
            if archive.read(filename) != output_bytes[filename]:
                raise AssertionError(f"candidate ZIP member differs: {filename}")

    manifest = {
        "run_id": run_id,
        "status": "ready_for_platform_experiment",
        "task": "round2_short_and_long_forecast",
        "experiment_role": config["experiment_role"],
        "protocol_version": config["protocol_version"],
        "experiment_version": config["experiment_version"],
        "control_submission_id": config["control_submission_id"],
        "hypothesis": config["hypothesis"],
        "primary_change": config["primary_change"],
        "created_at": created_at.isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "command": {
            "working_directory": str(PROJECT_ROOT),
            "argv": [sys.executable, *sys.argv],
        },
        "git": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyyaml": yaml.__version__,
        },
        "holdout_evaluated": False,
        "platform_result": None,
        "weights": {
            "short_generator_1_v34": short_weight,
            "long_generator_1_v34": long_weight,
            "v29b_complement": {
                "short": 1.0 - short_weight,
                "long": 1.0 - long_weight,
            },
        },
        "frozen_blocks": {
            "short_generator_all": "v29b exact",
            "long_generator_all": "v29b exact",
        },
        "fingerprints": {
            "config": sha256(config_path),
            "runner": sha256(RUNNER_SOURCE),
            "source_bundle": sha256(bundle_path),
            "v29b_s_result": components[base_name]["s_result_sha256"],
            "v29b_l_result": components[base_name]["l_result_sha256"],
            "v34_s_result": components[alternate_name]["s_result_sha256"],
            "v34_l_result": components[alternate_name]["l_result_sha256"],
        },
        "validation": reports,
        "artifacts": {
            "s_result": str((payload_dir / "s_result.csv").relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "s_result_sha256": bytes_sha256(output_bytes["s_result.csv"]),
            "l_result": str((payload_dir / "l_result.csv").relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "l_result_sha256": bytes_sha256(output_bytes["l_result.csv"]),
            "archive": str(archive_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "archive_sha256": sha256(archive_path),
            "archive_members": ["s_result.csv", "l_result.csv"],
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(
        "PASS "
        f"run_id={run_id} short_weight={short_weight} long_weight={long_weight} "
        f"archive={archive_path} archive_sha256={manifest['artifacts']['archive_sha256']} "
        f"duration_seconds={manifest['duration_seconds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
