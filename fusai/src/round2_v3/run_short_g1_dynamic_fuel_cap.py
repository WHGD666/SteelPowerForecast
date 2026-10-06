"""Evaluate the preregistered symmetric 5 MW dynamic short-g1 safety cap."""

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
from src.round2_v3.run_short_g1_dynamic_fuel import evaluate_gate, summarize_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/short_g1_dynamic_fuel_clip5_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()


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
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_short_dynamic_cap_{digest.hexdigest()[:10]}"


def apply_symmetric_cap(control: np.ndarray, candidate: np.ndarray, cap_mw: float) -> np.ndarray:
    if not np.isfinite(cap_mw) or cap_mw <= 0:
        raise ValueError("cap_mw must be finite and positive")
    control = np.asarray(control, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if control.shape != candidate.shape or not np.isfinite(control).all() or not np.isfinite(candidate).all():
        raise ValueError("control/candidate arrays must be finite and shape aligned")
    return control + np.clip(candidate - control, -cap_mw, cap_mw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = PROJECT_ROOT / config["inputs"]["source_manifest"]
    oof_path = PROJECT_ROOT / config["inputs"]["source_oof"]
    source_manifest = _read_json(manifest_path)
    if source_manifest["run_id"] != config["control_run_id"] or source_manifest["status"] != "completed":
        raise AssertionError("source run identity/status mismatch")
    if sha256(oof_path) != source_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("source OOF hash mismatch")
    scope = config["scope"]
    if scope["broad_cap_search"] or not np.isclose(float(scope["correction_cap_mw"]), 5.0):
        raise AssertionError("only the preregistered 5 MW cap is allowed")
    source = pd.read_csv(oof_path, parse_dates=["datetime", "interval_start"])
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    control = source.loc[source["variant"] == scope["control_variant"]].copy()
    alternate = source.loc[source["variant"] == scope["source_candidate_variant"], key + ["prediction"]].copy()
    alternate = alternate.rename(columns={"prediction": "source_candidate_prediction"})
    if control.duplicated(key).any() or alternate.duplicated(key).any():
        raise AssertionError("source predictions contain duplicate keys")
    merged = control.merge(alternate, on=key, how="inner", validate="one_to_one")
    if len(merged) != len(control) or len(merged) != len(alternate):
        raise AssertionError("source control/candidate coverage differs")
    if sorted(merged["fold_id"].unique()) != sorted(scope["fold_ids"]):
        raise AssertionError("source folds differ from preregistration")
    if sorted(merged["horizon_minutes"].unique()) != scope["horizons_minutes"]:
        raise AssertionError("source horizons differ from preregistration")
    capped_prediction = apply_symmetric_cap(
        merged["prediction"].to_numpy(float),
        merged["source_candidate_prediction"].to_numpy(float),
        float(scope["correction_cap_mw"]),
    )
    candidate = merged.drop(columns=["source_candidate_prediction"]).copy()
    candidate["prediction"] = capped_prediction
    candidate["variant"] = scope["promotion_candidate"]
    evaluation = pd.concat([control, candidate], ignore_index=True)
    changed = np.abs(capped_prediction - merged["prediction"].to_numpy(float))
    source_changed = np.abs(merged["source_candidate_prediction"].to_numpy(float) - merged["prediction"].to_numpy(float))
    change_report = {
        "cells": int(len(changed)),
        "cap_mw": float(scope["correction_cap_mw"]),
        "source_cells_exceeding_cap": int((source_changed > float(scope["correction_cap_mw"])).sum()),
        "source_mean_abs_change_mw": float(source_changed.mean()),
        "capped_mean_abs_change_mw": float(changed.mean()),
        "capped_max_abs_change_mw": float(changed.max()),
    }
    if args.preflight_only:
        print(f"PASS preflight rows={len(merged)} cap_mw={scope['correction_cap_mw']} source_exceeding_cap={change_report['source_cells_exceeding_cap']}", flush=True)
        return

    identity_paths = [manifest_path, oof_path, RUNNER_SOURCE, RUNNER_SOURCE.with_name("run_short_g1_dynamic_fuel.py")]
    run_id = _run_id(config_path, identity_paths)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": run_id, "status": "running", "task": config["task"], "experiment_role": config["experiment_role"], "protocol_version": "round2_v3", "experiment_version": config["experiment_version"], "control_run_id": config["control_run_id"], "started_at": started_at.isoformat(), "holdout_evaluated": False, "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"branch": _git_value("branch", "--show-current"), "commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, *identity_paths]}, "failure": {"type": None, "message": None, "traceback": None}, "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    registry_written = False
    try:
        metrics = summarize_predictions(evaluation)
        gate_config = {
            "variants": {"promotion_candidate": scope["promotion_candidate"]},
            "scope": {"fold_ids": scope["fold_ids"], "horizons_minutes": scope["horizons_minutes"]},
            "selection_gate": config["selection_gate"],
        }
        gates = evaluate_gate(metrics, gate_config)
        comparison = metrics.loc[metrics["scope"] == "overall", ["variant", "mape", "accuracy_1_minus_mape"]].copy()
        base_accuracy = float(comparison.loc[comparison["variant"] == "v16_control", "accuracy_1_minus_mape"].iloc[0])
        comparison["delta_accuracy_pct"] = (comparison["accuracy_1_minus_mape"] - base_accuracy) * 100
        print("\n" + comparison.sort_values("mape").to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        evaluation_path, metrics_path, gates_path, report_path = [output_dir / name for name in ("oof_predictions.csv", "metrics.csv", "gate_results.csv", "change_report.json")]
        evaluation.to_csv(evaluation_path, index=False, encoding="utf-8", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        write_json(report_path, change_report)
        passed = bool(gates.iloc[0]["component_gate_passed"])
        primary_mape = float(comparison.loc[comparison["variant"] == scope["promotion_candidate"], "mape"].iloc[0])
        manifest.update({"status": "completed", "ended_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(time.perf_counter() - started, 3), "primary_variant": scope["promotion_candidate"], "primary_short_g1_mape": primary_mape, "component_gate_passed": passed, "standalone_submission_eligible": passed, "submission_eligible": False, "change_report": change_report, "artifacts": {"oof_predictions": evaluation_path.relative_to(PROJECT_ROOT).as_posix(), "oof_predictions_sha256": sha256(evaluation_path), "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(), "metrics_sha256": sha256(metrics_path), "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(), "gate_results_sha256": sha256(gates_path), "change_report": report_path.relative_to(PROJECT_ROOT).as_posix(), "change_report_sha256": sha256(report_path)}})
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "dynamic_fuel_70_30_symmetric_5mw_cap", "short_g1_mape": primary_mape, "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"component_gate_passed={passed}; source_cells_exceeding_cap={change_report['source_cells_exceeding_cap']}"})
            registry_written = True
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={passed}", flush=True)
    except Exception as exc:
        manifest["status"] = "failed"; manifest["ended_at"] = datetime.now(timezone.utc).isoformat(); manifest["duration_seconds"] = round(time.perf_counter() - started, 3); manifest["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}; write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "dynamic_fuel_70_30_symmetric_5mw_cap", "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
