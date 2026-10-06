"""Build the preregistered National-Day-only long-g1 fuel correction probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.anomaly_weight_utils import (
    FUELS_G1,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
    v28_g1_fuel_weight,
)
from src.round2_v3.build_component_ladder import (
    frame_bytes,
    load_component,
    write_deterministic_zip,
)
from src.round2_v3.build_submission_baseline import (
    _append_submission_registry,
    apply_physical_projection,
)
from src.round2_v3.contracts import (
    LONG_HORIZONS_MINUTES,
    SHORT_HORIZONS_MINUTES,
    validate_prediction_frame,
)
from src.round2_v3.run_baseline import _git_value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/submission_holiday_long_g1_probe_v1.yaml"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("anomaly_weight_utils.py"),
    RUNNER_SOURCE.with_name("build_component_ladder.py"),
    RUNNER_SOURCE.with_name("build_submission_baseline.py"),
    RUNNER_SOURCE.with_name("contracts.py"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    return PROJECT_ROOT / value


def holiday_target_mask(
    origins: pd.DatetimeIndex,
    horizons: list[int],
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
) -> np.ndarray:
    """Mark cells by the start of each predicted 15-minute target interval."""
    origin_matrix = origins.to_numpy(dtype="datetime64[ns]")[:, None]
    offsets = np.asarray(horizons, dtype="timedelta64[m]")[None, :] - np.timedelta64(15, "m")
    target_starts = origin_matrix + offsets
    return (target_starts >= interval_start.to_datetime64()) & (
        target_starts <= interval_end.to_datetime64()
    )


def apply_holiday_fuel_correction(
    base: pd.DataFrame,
    origins: pd.DatetimeIndex,
    fuel_origin: np.ndarray,
    horizons: list[int],
    *,
    scale: float,
    interval_start: pd.Timestamp,
    interval_end: pd.Timestamp,
    projection: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0 < scale < 1:
        raise ValueError("holiday correction scale must be strictly between zero and one")
    if len(base) != len(origins) or len(fuel_origin) != len(origins):
        raise ValueError("base, origins, and fuel predictions must have identical row counts")
    g1_columns = [f"generator_1_t+{h}_pred" for h in horizons]
    gall_columns = [f"generator_all_t+{h}_pred" for h in horizons]
    g1_before = base[g1_columns].to_numpy(dtype=float)
    gall_before = base[gall_columns].to_numpy(dtype=float)
    mask = holiday_target_mask(origins, horizons, interval_start, interval_end)
    weights = np.asarray([v28_g1_fuel_weight(h) for h in horizons], dtype=float)
    delta = (scale - 1.0) * np.asarray(fuel_origin, dtype=float)[:, None] * weights[None, :]
    raw = np.where(mask, g1_before + delta, g1_before)
    candidate = base.copy()
    candidate[g1_columns] = raw
    projected, violations = apply_physical_projection(candidate, horizons, projection)
    g1_after = projected[g1_columns].to_numpy(dtype=float)
    gall_after = projected[gall_columns].to_numpy(dtype=float)
    if not np.array_equal(gall_before, gall_after):
        raise AssertionError("frozen v29b long generator_all changed")
    if not np.array_equal(g1_before[~mask], g1_after[~mask]):
        raise AssertionError("non-holiday v29b long generator_1 changed")
    absolute_delta = np.abs(g1_after - g1_before)
    near = np.asarray(horizons) <= 30
    holiday_delta = absolute_delta[mask]
    near_delta = absolute_delta[:, near][mask[:, near]]
    far_delta = absolute_delta[:, ~near][mask[:, ~near]]

    def optional_mean(values: np.ndarray) -> float | None:
        return float(values.mean()) if values.size else None

    return projected, {
        "formula": "v29b_g1 + fuel_weight_h*(0.95-1)*september_ols_fuel_origin",
        "target_interval_start": interval_start.isoformat(),
        "target_interval_end": interval_end.isoformat(),
        "fixed_scale": float(scale),
        "pre_projection_violations": violations,
        "short_bytes_exactly_frozen": True,
        "long_generator_all_exactly_frozen": True,
        "nonholiday_long_generator_1_exactly_frozen": True,
        "total_g1_cells": int(mask.size),
        "holiday_g1_cells": int(mask.sum()),
        "nonholiday_g1_cells": int((~mask).sum()),
        "changed_g1_cells": int(np.count_nonzero(g1_after - g1_before)),
        "holiday_mean_abs_change_mw": float(holiday_delta.mean()),
        "holiday_max_abs_change_mw": float(holiday_delta.max()),
        "near_holiday_cells": int(mask[:, near].sum()),
        "far_holiday_cells": int(mask[:, ~near].sum()),
        "near_mean_abs_change_mw": optional_mean(near_delta),
        "far_mean_abs_change_mw": optional_mean(far_delta),
        "generator_1_min": float(g1_after.min()),
        "generator_1_max": float(g1_after.max()),
    }


def _validate_sources(config: dict[str, Any]) -> dict[str, Any]:
    diagnostic_config_path = _resolve(config["source_diagnostic_config"])
    diagnostic_manifest_path = _resolve(config["source_diagnostic_manifest"])
    diagnostic_manifest = _read_json(diagnostic_manifest_path)
    if diagnostic_manifest["run_id"] != config["source_diagnostic_run_id"]:
        raise AssertionError("holiday diagnostic run id mismatch")
    if diagnostic_manifest["status"] != "completed" or not diagnostic_manifest["diagnostic_gate_passed"]:
        raise AssertionError("holiday diagnostic is incomplete or failed its preregistered gate")
    config_key = diagnostic_config_path.relative_to(PROJECT_ROOT).as_posix()
    if diagnostic_manifest["fingerprints"].get(config_key) != sha256(diagnostic_config_path):
        raise AssertionError("holiday diagnostic config no longer matches its run")
    for name, relative in diagnostic_manifest["artifacts"].items():
        path = _resolve(relative)
        if sha256(path) != diagnostic_manifest["artifact_sha256"][name]:
            raise AssertionError(f"holiday diagnostic artifact hash mismatch: {name}")

    correction = config["holiday_correction"]
    if not np.isclose(float(correction["fixed_scale"]), 0.95, atol=0.0, rtol=0.0):
        raise AssertionError("candidate is frozen to the preregistered scale 0.95")
    if [v28_g1_fuel_weight(15), v28_g1_fuel_weight(30), v28_g1_fuel_weight(45)] != [
        float(correction["near_weight"]),
        float(correction["near_weight"]),
        float(correction["far_weight"]),
    ]:
        raise AssertionError("configured fuel weights differ from deployed v28 semantics")

    fingerprint_path = _resolve(config["sources"]["raw_fingerprints"])
    expected = {row["file"]: row["sha256"] for row in _read_json(fingerprint_path)["files"]}
    train_paths = [_resolve(value) for value in config["sources"]["raw_train_tables"]]
    test_paths = [_resolve(value) for value in config["sources"]["raw_test_tables"]]
    for path in [*train_paths, *test_paths]:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256(path) != expected.get(relative):
            raise AssertionError(f"raw table hash mismatch: {relative}")
    bundle = _resolve(config["sources"]["v29b_bundle"]["path"])
    if sha256(bundle) != config["sources"]["v29b_bundle"]["sha256"]:
        raise AssertionError("v29b source bundle hash mismatch")
    artifact_paths = [_resolve(value) for value in diagnostic_manifest["artifacts"].values()]
    return {
        "diagnostic_config_path": diagnostic_config_path,
        "diagnostic_manifest_path": diagnostic_manifest_path,
        "diagnostic_manifest": diagnostic_manifest,
        "diagnostic_artifact_paths": artifact_paths,
        "fingerprint_path": fingerprint_path,
        "train_paths": train_paths,
        "test_paths": test_paths,
        "bundle": bundle,
    }


def _prepare_fuel(config: dict[str, Any], sources: dict[str, Any]) -> dict[str, Any]:
    train_raw = merge_training_tables(sources["train_paths"])
    test_raw = merge_training_tables(sources["test_paths"])
    grid = to_v28_grid(
        pd.concat([train_raw, test_raw], ignore_index=True, sort=False)
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    train_end = pd.Timestamp(train_raw["datetime"].max()).floor("15min")
    test_origins = pd.DatetimeIndex(grid.loc[grid["datetime"] > train_end, "datetime"])
    expected = pd.date_range("2025-10-01 00:00:00", "2025-10-10 23:45:00", freq="15min")
    if not test_origins.equals(expected):
        raise AssertionError("test origin timeline differs from submission contract")
    fit_start = pd.Timestamp(config["holiday_correction"]["fuel_fit_start"])
    fuel_train = grid.loc[(grid["datetime"] >= fit_start) & (grid["datetime"] <= train_end)]
    coefficients = fit_linear_fuel_proxy(fuel_train, "generator_1", FUELS_G1)
    test_grid = grid.set_index("datetime").loc[test_origins].reset_index()
    fuel_origin = predict_linear_fuel(test_grid, FUELS_G1, coefficients)
    if not np.isfinite(fuel_origin).all():
        raise AssertionError("test fuel proxy contains non-finite values")
    return {
        "train_end": train_end,
        "test_origins": test_origins,
        "fit_start": fit_start,
        "fit_rows": int(len(fuel_train)),
        "coefficients": coefficients,
        "fuel_origin": fuel_origin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources = _validate_sources(config)
    fuel = _prepare_fuel(config, sources)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"diagnostic_gate=true fuel_fit_rows={fuel['fit_rows']} "
            f"test_origins={len(fuel['test_origins'])} fixed_scale={config['holiday_correction']['fixed_scale']}",
            flush=True,
        )
        return

    identity_paths = [
        config_path,
        sources["diagnostic_config_path"],
        sources["diagnostic_manifest_path"],
        *sources["diagnostic_artifact_paths"],
        sources["fingerprint_path"],
        sources["bundle"],
        *sources["train_paths"],
        *sources["test_paths"],
        RUNNER_SOURCE,
        *HELPER_SOURCES,
    ]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    submission_id = (
        created_at.astimezone().strftime("%Y%m%dT%H%M%S")
        + "_holiday_long_g1_probe_v1_"
        + digest.hexdigest()[:10]
    )
    output_dir = PROJECT_ROOT / config["output"]["root"] / submission_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite submission: {output_dir}")
    payload_dir = output_dir / "payload"
    payload_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "submission_id": submission_id,
        "status": "building",
        "source_run_id": config["source_diagnostic_run_id"],
        "control_submission_id": config["control_submission_id"],
        "experiment_role": config["experiment_role"],
        "submission_version": config["submission_version"],
        "hypothesis": config["hypothesis"],
        "primary_change": config["primary_change"],
        "known_risks": config["known_risks"],
        "created_at": created_at.isoformat(),
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
        "holdout_evaluated": False,
        "platform_result": None,
        "training": [],
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths
        },
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "submission_manifest.json", manifest)
    try:
        v29b = config["sources"]["v29b_bundle"]
        with zipfile.ZipFile(sources["bundle"], "r") as archive:
            if archive.testzip() is not None:
                raise AssertionError("v29b source bundle CRC failed")
            loaded = load_component(archive, v29b)
        v29b_short, short_bytes = loaded["s_result.csv"]
        v29b_long, _ = loaded["l_result.csv"]
        origins = fuel["test_origins"]
        horizons = list(LONG_HORIZONS_MINUTES)
        validate_prediction_frame(v29b_short, origins, SHORT_HORIZONS_MINUTES)
        validate_prediction_frame(v29b_long, origins, horizons)
        correction = config["holiday_correction"]
        candidate_long, change_report = apply_holiday_fuel_correction(
            v29b_long,
            origins,
            fuel["fuel_origin"],
            horizons,
            scale=float(correction["fixed_scale"]),
            interval_start=pd.Timestamp(correction["target_interval_start"]),
            interval_end=pd.Timestamp(correction["target_interval_end"]),
            projection=config["physical_projection"],
        )
        short_validation = validate_prediction_frame(v29b_short, origins, SHORT_HORIZONS_MINUTES)
        long_validation = validate_prediction_frame(candidate_long, origins, horizons)
        short_path = payload_dir / "s_result.csv"
        long_path = payload_dir / "l_result.csv"
        short_path.write_bytes(short_bytes)
        long_bytes = frame_bytes(candidate_long, config["output"]["float_format"])
        long_path.write_bytes(long_bytes)
        archive_path = output_dir / config["output"]["archive_filename"]
        timestamp = tuple(int(value) for value in config["output"]["zip_timestamp"])
        write_deterministic_zip(
            archive_path,
            [("s_result.csv", short_bytes), ("l_result.csv", long_bytes)],
            timestamp,  # type: ignore[arg-type]
        )
        with zipfile.ZipFile(archive_path, "r") as archive:
            if archive.namelist() != ["s_result.csv", "l_result.csv"]:
                raise AssertionError("candidate ZIP members differ from contract")
            if archive.testzip() is not None:
                raise AssertionError("candidate ZIP CRC failed")
            if archive.read("s_result.csv") != short_bytes or archive.read("l_result.csv") != long_bytes:
                raise AssertionError("candidate ZIP payload differs from frozen files")
        report_path = output_dir / "change_report.json"
        write_json(report_path, change_report)
        manifest.update(
            {
                "status": "ready_for_upload",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "fuel_proxy": {
                    "fit_start": fuel["fit_start"].isoformat(),
                    "fit_end": fuel["train_end"].isoformat(),
                    "fit_rows": fuel["fit_rows"],
                    "columns": FUELS_G1,
                    "coefficients": np.asarray(fuel["coefficients"]).tolist(),
                },
                "frozen_blocks": config["frozen_blocks"],
                "source_diagnostic": {
                    "run_id": sources["diagnostic_manifest"]["run_id"],
                    "diagnostic_gate_passed": True,
                },
                "validation": {
                    "short": short_validation,
                    "long": long_validation,
                    "change_vs_v29b": change_report,
                    "short_bytes_exactly_v29b": True,
                },
                "artifacts": {
                    "change_report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                    "change_report_sha256": sha256(report_path),
                    "s_result": short_path.relative_to(PROJECT_ROOT).as_posix(),
                    "s_result_sha256": sha256(short_path),
                    "l_result": long_path.relative_to(PROJECT_ROOT).as_posix(),
                    "l_result_sha256": sha256(long_path),
                    "archive": archive_path.relative_to(PROJECT_ROOT).as_posix(),
                    "archive_sha256": sha256(archive_path),
                    "archive_members": ["s_result.csv", "l_result.csv"],
                },
            }
        )
        write_json(output_dir / "submission_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_submission_registry(
                {
                    "submission_id": submission_id,
                    "source_run_id": config["source_diagnostic_run_id"],
                    "control_submission_id": config["control_submission_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "archive_filename": config["output"]["archive_filename"],
                    "archive_sha256": manifest["artifacts"]["archive_sha256"],
                    "s_result_sha256": manifest["artifacts"]["s_result_sha256"],
                    "l_result_sha256": manifest["artifacts"]["l_result_sha256"],
                    "anomaly_status": "not_submitted",
                    "is_official_best": False,
                    "reproducibility_status": "ready_for_upload_isolated_holiday_probe",
                    "notes": "Only holiday-target long g1 changed; v29b short bytes, long gall, and nonholiday long g1 frozen.",
                }
            )
        print(
            f"PASS submission_id={submission_id} archive={archive_path} "
            f"sha256={manifest['artifacts']['archive_sha256']} "
            f"changed_cells={change_report['changed_g1_cells']} "
            f"duration_seconds={manifest['duration_seconds']}",
            flush=True,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "submission_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
