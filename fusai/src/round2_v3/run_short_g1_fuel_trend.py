"""Evaluate a damped one-hour fuel-trend correction for v16 short-g1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.anomaly_weight_utils import (
    FUELS_G1,
    fit_linear_fuel_proxy,
    merge_training_tables,
    to_v28_grid,
)
from src.round2_v3.event_ablation_utils import attach_event_context
from src.round2_v3.run_short_g1_dynamic_fuel import (
    evaluate_gate,
    summarize_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/short_g1_fuel_trend_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("anomaly_weight_utils.py"),
    RUNNER_SOURCE.with_name("event_ablation_utils.py"),
    RUNNER_SOURCE.with_name("run_short_g1_dynamic_fuel.py"),
    RUNNER_SOURCE.with_name("metrics.py"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {item["run_id"] for item in reader}
    if not fields or str(row["run_id"]) in existing:
        raise RuntimeError("invalid or duplicate experiment registry entry")
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow({field: row.get(field, "") for field in fields})


def _run_id(config_path: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *paths]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_short_fuel_trend_{digest.hexdigest()[:10]}"


def damped_fuel_trend_correction(
    current_fuels: np.ndarray,
    lagged_fuels: np.ndarray,
    coefficients: np.ndarray,
    horizon_minutes: int,
    *,
    fuel_weight: float,
    full_extrapolation_minutes: int,
    maximum_factor: float,
    cap_mw: float,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current_fuels, dtype=float)
    lagged = np.asarray(lagged_fuels, dtype=float)
    coef = np.asarray(coefficients, dtype=float)
    if current.shape != lagged.shape or current.ndim != 2:
        raise ValueError("current and lagged fuel matrices must be shape aligned")
    if coef.shape != (current.shape[1] + 1,):
        raise ValueError("fuel coefficients must contain one slope per fuel plus intercept")
    if not np.isfinite(current).all() or not np.isfinite(lagged).all() or not np.isfinite(coef).all():
        raise ValueError("fuel inputs and coefficients must be finite")
    if horizon_minutes <= 0 or full_extrapolation_minutes <= 0 or cap_mw <= 0:
        raise ValueError("horizon, extrapolation period, and cap must be positive")
    factor = min(float(horizon_minutes) / float(full_extrapolation_minutes), float(maximum_factor))
    raw = float(fuel_weight) * factor * ((current - lagged) @ coef[:-1])
    return np.clip(raw, -float(cap_mw), float(cap_mw)), raw


def _load_control(path: Path, variant: str, folds: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime", "interval_start"])
    frame = frame.loc[(frame["variant"] == variant) & frame["fold_id"].isin(folds)].copy()
    frame = frame.rename(columns={"actual_interval_mean": "actual"})
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    if frame.empty or frame.duplicated(key).any():
        raise AssertionError("control OOF is empty or has duplicate keys")
    result = frame[key + ["actual", "prediction"]].copy()
    result["variant"] = "v16_control"
    return result


def _validate(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    fingerprints = _read_json(paths["raw_fingerprints"])
    expected = {item["file"]: item["sha256"] for item in fingerprints["files"]}
    raw_paths = [PROJECT_ROOT / value for value in config["inputs"]["raw_train_tables"]]
    for path in raw_paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256(path) != expected.get(relative):
            raise AssertionError(f"raw table hash mismatch: {relative}")
    split_manifest = _read_json(paths["split_manifest"])
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    control_manifest = _read_json(paths["control_manifest"])
    if control_manifest["run_id"] != config["control_run_id"] or control_manifest["status"] != "completed":
        raise AssertionError("control run identity/status mismatch")
    if sha256(paths["control_oof"]) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("control OOF hash mismatch")
    scope = config["scope"]
    trend = config["fuel_trend"]
    if scope["period"] != "short" or scope["target"] != "generator_1":
        raise AssertionError("experiment must remain short generator_1 only")
    if scope["generator_all_changed"] or scope["long_outputs_changed"]:
        raise AssertionError("generator_all and long outputs must remain frozen")
    if scope["future_process_observations_allowed"] or scope["true_target_history_allowed"]:
        raise AssertionError("forbidden information was enabled")
    if trend["broad_parameter_search"]:
        raise AssertionError("parameter search is forbidden")
    if list(trend["columns"]) != FUELS_G1:
        raise AssertionError("fuel columns differ from the v16 g1 fuel proxy")
    frozen = (int(trend["lag_minutes"]), int(trend["full_extrapolation_minutes"]), float(trend["maximum_extrapolation_factor"]), float(trend["v16_fuel_weight"]), float(trend["symmetric_correction_cap_mw"]))
    if frozen != (60, 60, 1.0, 0.8, 5.0):
        raise AssertionError("fuel trend parameters differ from preregistration")
    folds = list(scope["fold_ids"])
    horizons = [int(value) for value in scope["horizons_minutes"]]
    if horizons != list(range(15, 121, 15)) or len(folds) != int(config["budget"]["fold_count"]):
        raise AssertionError("fold/horizon budget mismatch")
    return {"raw_paths": raw_paths, "split_manifest": split_manifest, "control_manifest": control_manifest, "folds": folds, "horizons": horizons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    path_keys = ["raw_fingerprints", "split_manifest", "split_assignments", "event_episodes", "control_manifest", "control_oof"]
    paths = {key: PROJECT_ROOT / config["inputs"][key] for key in path_keys}
    validated = _validate(config, paths)
    grid = to_v28_grid(merge_training_tables(validated["raw_paths"]))
    grid_indexed = grid.set_index("datetime")
    split = pd.read_csv(paths["split_assignments"], parse_dates=["train_end", "validation_origin"])
    split = split.loc[(split["role"] == "model_selection") & split["fold_id"].isin(validated["folds"])].copy()
    if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
        raise AssertionError("fold order/content differs from preregistration")
    control = _load_control(paths["control_oof"], config["scope"]["control_variant"], validated["folds"])
    trend = config["fuel_trend"]
    all_candidates: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for fold_id, fold in split.groupby("fold_id", sort=False):
        train_end = pd.Timestamp(fold["train_end"].iloc[0])
        origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
        fuel_training = grid.loc[(grid["datetime"] >= pd.Timestamp(trend["fuel_proxy_fit_start"])) & (grid["datetime"] <= train_end)]
        coefficients = fit_linear_fuel_proxy(fuel_training, "generator_1", FUELS_G1)
        current = grid_indexed.loc[origins, FUELS_G1].to_numpy(float)
        lagged = grid_indexed.loc[origins - pd.Timedelta(minutes=int(trend["lag_minutes"])), FUELS_G1].to_numpy(float)
        for horizon in validated["horizons"]:
            correction, raw_correction = damped_fuel_trend_correction(current, lagged, coefficients, horizon, fuel_weight=float(trend["v16_fuel_weight"]), full_extrapolation_minutes=int(trend["full_extrapolation_minutes"]), maximum_factor=float(trend["maximum_extrapolation_factor"]), cap_mw=float(trend["symmetric_correction_cap_mw"]))
            part = control.loc[(control["fold_id"] == fold_id) & (control["horizon_minutes"] == horizon)].sort_values("datetime").reset_index(drop=True)
            if not pd.DatetimeIndex(part["datetime"]).equals(origins):
                raise AssertionError("control origins differ from split origins")
            part["prediction"] = part["prediction"].to_numpy(float) + correction
            part["variant"] = config["scope"]["promotion_candidate"]
            part["raw_correction_mw"] = raw_correction
            part["applied_correction_mw"] = correction
            all_candidates.append(part)
        fit_rows.append({"fold_id": fold_id, "train_end": train_end, "fit_start": pd.Timestamp(trend["fuel_proxy_fit_start"]), "fit_rows": len(fuel_training), "coefficients": json.dumps(np.asarray(coefficients).tolist())})
    candidate = pd.concat(all_candidates, ignore_index=True)
    if len(candidate) != len(control):
        raise AssertionError("candidate/control coverage differs")
    episodes = pd.read_csv(paths["event_episodes"])
    control_eval = attach_event_context(control.copy(), episodes, context_hours=24)
    candidate_eval = attach_event_context(candidate.copy(), episodes, context_hours=24)
    evaluation = pd.concat([control_eval, candidate_eval], ignore_index=True)
    if args.preflight_only:
        print(f"PASS preflight folds={len(validated['folds'])} horizons={len(validated['horizons'])} fuel_fits={len(fit_rows)} rows={len(candidate)} max_source_offset=0", flush=True)
        return

    identity_paths = [*paths.values(), *validated["raw_paths"], RUNNER_SOURCE, *HELPER_SOURCES]
    run_id = _run_id(config_path, identity_paths)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": run_id, "status": "running", "task": config["task"], "experiment_role": config["experiment_role"], "protocol_version": "round2_v3", "experiment_version": config["experiment_version"], "control_run_id": config["control_run_id"], "started_at": started_at.isoformat(), "holdout_evaluated": False, "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]}, "git": {"branch": _git_value("branch", "--show-current"), "commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))}, "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__}, "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, *identity_paths]}, "failure": {"type": None, "message": None, "traceback": None}, "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    registry_written = False
    try:
        metrics = summarize_predictions(evaluation)
        gate_config = {"variants": {"promotion_candidate": config["scope"]["promotion_candidate"]}, "scope": {"fold_ids": validated["folds"], "horizons_minutes": validated["horizons"]}, "selection_gate": config["selection_gate"]}
        gates = evaluate_gate(metrics, gate_config)
        comparison = metrics.loc[metrics["scope"] == "overall", ["variant", "mape", "accuracy_1_minus_mape"]].copy()
        base_accuracy = float(comparison.loc[comparison["variant"] == "v16_control", "accuracy_1_minus_mape"].iloc[0])
        comparison["delta_accuracy_pct"] = (comparison["accuracy_1_minus_mape"] - base_accuracy) * 100
        print("\n" + comparison.sort_values("mape").to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        oof_path, metrics_path, gates_path, fits_path = [output_dir / name for name in ("oof_predictions.csv", "metrics.csv", "gate_results.csv", "fit_summary.csv")]
        evaluation.to_csv(oof_path, index=False, encoding="utf-8", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        pd.DataFrame(fit_rows).to_csv(fits_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
        passed = bool(gates.iloc[0]["component_gate_passed"])
        primary_mape = float(comparison.loc[comparison["variant"] == config["scope"]["promotion_candidate"], "mape"].iloc[0])
        corrections = candidate["applied_correction_mw"].to_numpy(float)
        raw_corrections = candidate["raw_correction_mw"].to_numpy(float)
        manifest.update({"status": "completed", "ended_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(time.perf_counter() - started, 3), "primary_variant": config["scope"]["promotion_candidate"], "primary_short_g1_mape": primary_mape, "component_gate_passed": passed, "standalone_submission_eligible": passed, "submission_eligible": False, "correction_summary": {"cells": len(corrections), "mean_abs_mw": float(np.abs(corrections).mean()), "max_abs_mw": float(np.abs(corrections).max()), "raw_cells_exceeding_cap": int((np.abs(raw_corrections) > float(trend["symmetric_correction_cap_mw"])).sum())}, "artifacts": {"oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(), "oof_predictions_sha256": sha256(oof_path), "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(), "metrics_sha256": sha256(metrics_path), "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(), "gate_results_sha256": sha256(gates_path), "fit_summary": fits_path.relative_to(PROJECT_ROOT).as_posix(), "fit_summary_sha256": sha256(fits_path)}})
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "split_sha256": validated["split_manifest"]["assignment_sha256"], "control_run_id": config["control_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "damped_one_hour_fuel_trend", "short_g1_mape": primary_mape, "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"component_gate_passed={passed}; correction_mean_abs_mw={manifest['correction_summary']['mean_abs_mw']:.4f}"})
            registry_written = True
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={passed}", flush=True)
    except Exception as exc:
        manifest["status"] = "failed"; manifest["ended_at"] = datetime.now(timezone.utc).isoformat(); manifest["duration_seconds"] = round(time.perf_counter() - started, 3); manifest["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}; write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "damped_one_hour_fuel_trend", "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
