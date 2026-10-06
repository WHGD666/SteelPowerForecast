"""Audit cross-holiday transfer of a generator_1 fuel-proxy scale."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.anomaly_weight_utils import (
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
)
from src.round2_v3.holiday_effect_utils import mape, mape_optimal_scale
from src.round2_v3.run_anomaly_weight_g1 import (
    _append_event,
    _append_registry,
    _git_value,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/diagnostics/holiday_effect_audit_v1.yaml"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("holiday_effect_utils.py")
GRID_SOURCE = RUNNER_SOURCE.with_name("anomaly_weight_utils.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _date_mask(frame: pd.DataFrame, dates: list[str]) -> np.ndarray:
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    normalized = frame["datetime"].dt.normalize()
    return normalized.between(start, end).to_numpy(dtype=bool)


def _run_identity(config_path: Path, inputs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *inputs]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_holiday_audit_" + digest.hexdigest()[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    raw_manifest_path = PROJECT_ROOT / inputs["raw_fingerprints"]
    raw_manifest = _read_json(raw_manifest_path)
    expected = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
    raw_paths = [PROJECT_ROOT / item for item in inputs["raw_train_tables"]]
    for path in raw_paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if expected.get(relative) != sha256(path):
            raise AssertionError(f"raw fingerprint mismatch: {relative}")
    calendar = config["calendar_source"]
    if calendar["labor_observed"] != ["2025-05-03", "2025-05-05"]:
        raise AssertionError("labor observed dates differ from frozen protocol")
    if calendar["dragon_boat"] != ["2025-05-31", "2025-06-02"]:
        raise AssertionError("dragon boat dates differ from frozen protocol")
    diagnostic = config["diagnostic"]
    if diagnostic["future_process_observations_allowed"] or diagnostic["test_targets_used"] or diagnostic["platform_feedback_used_for_fit"]:
        raise AssertionError("diagnostic information boundary violated")
    if args.preflight_only:
        print(
            "PASS preflight blocks=2 directions=2 target=generator_1 "
            "test_targets=false create_submission=false",
            flush=True,
        )
        return

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    identity_inputs = [*raw_paths, raw_manifest_path, RUNNER_SOURCE, UTILS_SOURCE, GRID_SOURCE]
    run_id = _run_identity(config_path, identity_inputs)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    event_log = output_dir / "events.jsonl"
    git_commit = _git_value("rev-parse", "HEAD")
    git_dirty = bool(_git_value("status", "--porcelain"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": config["task"],
        "experiment_role": config["experiment_role"],
        "diagnostic_version": config["diagnostic_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "holdout_evaluated": False,
        "submission_eligible": False,
        "calendar_source": calendar,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in [config_path, *identity_inputs]
        },
        "failure": None,
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False
    try:
        grid = to_v28_grid(merge_training_tables(raw_paths))
        masks = {
            "labor_observed": _date_mask(grid, calendar["labor_observed"]),
            "dragon_boat": _date_mask(grid, calendar["dragon_boat"]),
        }
        holiday_mask = masks["labor_observed"] | masks["dragon_boat"]
        regular = grid.loc[~holiday_mask].copy()
        fuels = list(diagnostic["fuel_columns"])
        coefficients = fit_linear_fuel_proxy(regular, "generator_1", fuels)
        grid = grid.copy()
        grid["base_prediction"] = predict_linear_fuel(grid, fuels, coefficients)

        block_stats: dict[str, dict[str, float]] = {}
        for name, mask in masks.items():
            part = grid.loc[mask].copy()
            actual = part["generator_1"].to_numpy(dtype=float)
            base = part["base_prediction"].to_numpy(dtype=float)
            scale = mape_optimal_scale(actual, base)
            block_stats[name] = {
                "rows": int(len(part)),
                "actual_mean": float(np.mean(actual)),
                "base_mean": float(np.mean(base)),
                "base_mape": mape(actual, base),
                "own_optimal_scale": scale,
            }

        rows: list[dict[str, object]] = []
        directions = [
            ("labor_observed", "dragon_boat"),
            ("dragon_boat", "labor_observed"),
        ]
        for calibration_name, validation_name in directions:
            scale = block_stats[calibration_name]["own_optimal_scale"]
            part = grid.loc[masks[validation_name]].copy()
            actual = part["generator_1"].to_numpy(dtype=float)
            base = part["base_prediction"].to_numpy(dtype=float)
            base_mape = mape(actual, base)
            corrected_mape = mape(actual, scale * base)
            rows.append(
                {
                    "calibration_block": calibration_name,
                    "validation_block": validation_name,
                    "validation_rows": len(part),
                    "learned_scale": scale,
                    "validation_base_mape": base_mape,
                    "validation_corrected_mape": corrected_mape,
                    "validation_accuracy_gain_pct": (base_mape - corrected_mape) * 100.0,
                }
            )
        transfer = pd.DataFrame(rows)
        scales = [block_stats[name]["own_optimal_scale"] for name in masks]
        gains = transfer["validation_accuracy_gain_pct"].to_numpy(dtype=float)
        gate = config["gate"]
        pass_directions = bool((gains > 0).all())
        pass_mean = float(gains.mean()) >= float(gate["minimum_mean_transfer_gain_pct"])
        pass_side = bool((np.asarray(scales) > 1).all() or (np.asarray(scales) < 1).all())
        pass_scale_distance = abs(scales[0] - scales[1]) <= float(gate["maximum_absolute_scale_difference"])
        pass_rows = min(item["rows"] for item in block_stats.values()) >= int(gate["minimum_rows_per_block"])
        gate_passed = bool(pass_directions and pass_mean and pass_side and pass_scale_distance and pass_rows)
        gate_frame = pd.DataFrame(
            [
                {
                    "mean_transfer_gain_pct": float(gains.mean()),
                    "minimum_transfer_gain_pct": float(gains.min()),
                    "labor_optimal_scale": scales[0],
                    "dragon_boat_optimal_scale": scales[1],
                    "absolute_scale_difference": abs(scales[0] - scales[1]),
                    "pass_both_directions": pass_directions,
                    "pass_mean_gain": pass_mean,
                    "pass_same_side": pass_side,
                    "pass_scale_distance": pass_scale_distance,
                    "pass_minimum_rows": pass_rows,
                    "diagnostic_gate_passed": gate_passed,
                }
            ]
        )
        block_frame = pd.DataFrame(
            [{"block": name, **values} for name, values in block_stats.items()]
        )
        paths = {
            "block_summary": output_dir / "block_summary.csv",
            "cross_block_transfer": output_dir / "cross_block_transfer.csv",
            "gate_results": output_dir / "gate_results.csv",
        }
        block_frame.to_csv(paths["block_summary"], index=False, float_format="%.10f")
        transfer.to_csv(paths["cross_block_transfer"], index=False, float_format="%.10f")
        gate_frame.to_csv(paths["gate_results"], index=False, float_format="%.10f")
        ended = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended.isoformat(),
                "duration_seconds": round(time.perf_counter() - clock, 3),
                "diagnostic_gate_passed": gate_passed,
                "artifacts": {
                    name: path.relative_to(PROJECT_ROOT).as_posix()
                    for name, path in paths.items()
                },
                "artifact_sha256": {name: sha256(path) for name, path in paths.items()},
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "two_block_holiday_fuel_scale_audit",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"diagnostic_gate_passed={gate_passed}; mean_transfer_gain_pct={gains.mean():.4f}",
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", diagnostic_gate_passed=gate_passed)
        print("\n" + block_frame.to_string(index=False), flush=True)
        print("\n" + transfer.to_string(index=False), flush=True)
        print("\n" + gate_frame.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} diagnostic_gate_passed={gate_passed}",
            flush=True,
        )
    except Exception as exc:
        ended = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "failed",
                "ended_at": ended.isoformat(),
                "duration_seconds": round(time.perf_counter() - clock, 3),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", failure_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "two_block_holiday_fuel_scale_audit",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()

