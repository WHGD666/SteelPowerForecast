"""Audit fold-local holder-2 boundary pressure against frozen long-g1 residuals."""

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
from src.round2_v3.holder_boundary_utils import (
    REQUIRED_COLUMNS,
    build_boundary_features,
    fit_boundary_contract,
    spearman_rank_correlation,
    summarize_pressure_relationships,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/diagnostics/holder2_boundary_audit_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCE = RUNNER_SOURCE.with_name("holder_boundary_utils.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [item for item in manifest["partitions"] if item["partition"] == name]
    if len(rows) != 1:
        raise AssertionError(f"feature manifest has {len(rows)} {name} partitions")
    return rows[0]


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {item["run_id"] for item in reader}
    if not fields or str(row["run_id"]) in existing:
        raise RuntimeError("invalid or duplicate experiment registry entry")
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(
            {field: row.get(field, "") for field in fields}
        )


def _append_event(path: Path, event: str, **values: object) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _validate(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    feature_manifest_path = PROJECT_ROOT / config["inputs"]["event_feature_manifest"]
    feature_registry_path = PROJECT_ROOT / config["inputs"]["event_feature_registry"]
    control_manifest_path = PROJECT_ROOT / config["inputs"]["control_manifest"]
    control_oof_path = PROJECT_ROOT / config["inputs"]["control_oof"]
    feature_manifest = _read_json(feature_manifest_path)
    control_manifest = _read_json(control_manifest_path)
    feature_partition = _partition(feature_manifest, "train")
    feature_path = PROJECT_ROOT / feature_partition["output"]
    if sha256(feature_path) != feature_partition["output_sha256"]:
        raise AssertionError("event feature artifact hash mismatch")
    if sha256(feature_registry_path) != feature_manifest["registry_sha256"]:
        raise AssertionError("event feature registry hash mismatch")
    if feature_manifest["target_history_included"]:
        raise AssertionError("event features unexpectedly include target history")
    if int(feature_manifest["maximum_source_time_offset_minutes"]) > 0:
        raise AssertionError("event features contain future process observations")
    if control_manifest["status"] != "completed":
        raise AssertionError("control scientific run is not completed")
    expected_oof_hash = control_manifest["artifacts"]["oof_predictions_sha256"]
    if sha256(control_oof_path) != expected_oof_hash:
        raise AssertionError("control OOF hash mismatch")
    if config["validation"]["true_target_history_allowed"]:
        raise AssertionError("true target history must remain forbidden")
    if config["validation"]["future_process_observations_allowed"]:
        raise AssertionError("future process observations must remain forbidden")

    required = list(config["boundary_contract"]["required_columns"])
    if tuple(required) != tuple(REQUIRED_COLUMNS):
        raise AssertionError("boundary required-column contract drifted")
    registry = pd.read_csv(feature_registry_path)
    chosen = registry.loc[registry["feature_name"].isin(required)]
    if set(chosen["feature_name"]) != set(required):
        raise KeyError("boundary feature registry is incomplete")
    if not chosen["causal"].astype(bool).all() or chosen["target_history"].astype(bool).any():
        raise AssertionError("boundary inputs violate causal feature contract")
    expected_weights = {
        "centered_position": 0.45,
        "multiscale_velocity": 0.25,
        "signed_boundary_dwell": 0.20,
        "gas_balance_state": 0.10,
    }
    if config["boundary_contract"]["score_weights"] != expected_weights:
        raise AssertionError("fixed pressure-score weights changed")
    if not np.isclose(sum(expected_weights.values()), 1.0):
        raise AssertionError("pressure-score weights must sum to one")

    blocks: list[dict[str, Any]] = []
    seen: set[pd.Timestamp] = set()
    for block in config["validation"]["blocks"]:
        start = pd.Timestamp(block["validation_start"])
        end = pd.Timestamp(block["validation_end"])
        origins = pd.date_range(start, end, freq="15min")
        if len(origins) != 960:
            raise AssertionError(f"{block['fold_id']} is not a complete ten-day block")
        if any(origin in seen for origin in origins):
            raise AssertionError("validation blocks overlap")
        seen.update(origins)
        blocks.append({**block, "start": start, "end": end, "origins": origins})
    return {
        "config_path": config_path,
        "feature_manifest_path": feature_manifest_path,
        "feature_registry_path": feature_registry_path,
        "control_manifest_path": control_manifest_path,
        "control_oof_path": control_oof_path,
        "feature_path": feature_path,
        "feature_manifest": feature_manifest,
        "control_manifest": control_manifest,
        "required_columns": required,
        "blocks": blocks,
    }


def _origin_outcomes(control: pd.DataFrame, near_bucket: str, far_bucket: str) -> pd.DataFrame:
    frame = control.copy()
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    valid = np.isfinite(frame["actual"]) & np.isfinite(frame["prediction"]) & (frame["actual"] != 0)
    frame = frame.loc[valid].copy()
    frame["signed_percentage_residual"] = (
        (frame["actual"] - frame["prediction"]) / frame["actual"].abs()
    )
    keys = ["fold_id", "datetime"]
    residual = (
        frame.groupby(keys + ["horizon_bucket"], sort=False)["signed_percentage_residual"]
        .mean()
        .unstack("horizon_bucket")
    )
    actual = (
        frame.groupby(keys + ["horizon_bucket"], sort=False)["actual"]
        .mean()
        .unstack("horizon_bucket")
    )
    for bucket in (near_bucket, far_bucket):
        if bucket not in residual or bucket not in actual:
            raise AssertionError(f"control OOF lacks horizon bucket {bucket}")
    result = pd.DataFrame(index=residual.index)
    result["near_signed_percentage_residual"] = residual[near_bucket]
    result["far_signed_percentage_residual"] = residual[far_bucket]
    result["near_actual_mean"] = actual[near_bucket]
    result["far_actual_mean"] = actual[far_bucket]
    result["far_minus_near_actual_ratio"] = (
        (result["far_actual_mean"] - result["near_actual_mean"])
        / result["near_actual_mean"].abs().clip(lower=1e-6)
    )
    return result.reset_index()


def _build_fold_diagnostics(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    block: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    boundary_config = config["boundary_contract"]
    history = features.loc[features["datetime"] < block["start"]].copy()
    if len(history) < 96 * 14:
        raise AssertionError(f"{block['fold_id']} has less than 14 days of boundary history")
    contract = fit_boundary_contract(
        history,
        reference_quantiles=boundary_config["reference_quantiles"],
        zone_quantiles=boundary_config["zone_quantiles"],
        scale_floor=float(boundary_config["robust_scale_floor"]),
    )
    through_validation = features.loc[features["datetime"] <= block["end"]].copy()
    boundary = build_boundary_features(
        through_validation[list(REQUIRED_COLUMNS)],
        contract,
        maximum_dwell_steps=int(boundary_config["maximum_dwell_steps"]),
    )
    boundary.insert(0, "datetime", through_validation["datetime"].to_numpy())
    history_scores = boundary.loc[boundary["datetime"] < block["start"], "boundary_pressure_score"]
    score_low_q, score_high_q = [float(value) for value in boundary_config["score_extreme_quantiles"]]
    score_low, score_high = np.quantile(history_scores.to_numpy(dtype=float), [score_low_q, score_high_q])
    contract["score_low"] = float(score_low)
    contract["score_high"] = float(score_high)

    validation = boundary.loc[boundary["datetime"].isin(block["origins"])].copy()
    if len(validation) != len(block["origins"]):
        raise AssertionError(f"{block['fold_id']} boundary origins are incomplete")
    validation["fold_id"] = block["fold_id"]
    validation["score_group"] = "middle"
    validation.loc[validation["boundary_pressure_score"] <= score_low, "score_group"] = "low"
    validation.loc[validation["boundary_pressure_score"] >= score_high, "score_group"] = "high"
    fold_outcomes = outcomes.loc[outcomes["fold_id"] == block["fold_id"]]
    joined = validation.merge(fold_outcomes, on=["fold_id", "datetime"], how="left", validate="one_to_one")
    if joined["far_signed_percentage_residual"].isna().all():
        raise AssertionError(f"{block['fold_id']} has no valid far residual outcomes")
    summary = {
        "fold_id": block["fold_id"],
        "history_rows": len(history),
        "validation_origins": len(validation),
        "history_end": history["datetime"].max(),
        "reference_low": contract["reference_low"],
        "zone_low": contract["zone_low"],
        "zone_high": contract["zone_high"],
        "reference_high": contract["reference_high"],
        "score_low": contract["score_low"],
        "score_high": contract["score_high"],
        "validation_low_score_fraction": float((validation["score_group"] == "low").mean()),
        "validation_high_score_fraction": float((validation["score_group"] == "high").mean()),
    }
    return joined, summary, contract


def _feature_correlations(per_origin: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "holder_position",
        "distance_to_low",
        "distance_to_high",
        "high_dwell_fraction",
        "low_dwell_fraction",
        "signed_dwell",
        "velocity_1h",
        "velocity_4h",
        "velocity_12h",
        "velocity_mean",
        "balance_state",
        "boundary_pressure_score",
    ]
    rows: list[dict[str, object]] = []
    for scope, part in [("pooled", per_origin)] + [
        (str(fold), group) for fold, group in per_origin.groupby("fold_id", sort=False)
    ]:
        for feature in feature_columns:
            rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "far_residual_spearman": spearman_rank_correlation(
                        part[feature], part["far_signed_percentage_residual"]
                    ),
                    "target_change_spearman": spearman_rank_correlation(
                        part[feature], part["far_minus_near_actual_ratio"]
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _evaluate_gate(summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    gate = config["diagnostic_gate"]
    pooled = summary.loc[summary["scope"] == "pooled"].iloc[0]
    blocks = summary.loc[summary["scope"] != "pooled"].copy()
    positive_rho_fraction = float((blocks["pressure_residual_spearman"] > 0).mean())
    positive_spread_fraction = float((blocks["high_minus_low_far_residual_pct"] > 0).mean())
    min_extreme_fraction = float(
        blocks[["high_score_fraction", "low_score_fraction"]].to_numpy(dtype=float).min()
    )
    pass_rho = float(pooled["pressure_residual_spearman"]) >= float(
        gate["minimum_pooled_far_residual_spearman"]
    )
    pass_block_direction = positive_rho_fraction >= float(gate["minimum_positive_block_fraction"])
    pass_spread = float(pooled["high_minus_low_far_residual_pct"]) >= float(
        gate["minimum_pooled_high_low_residual_spread_pct"]
    )
    pass_spread_blocks = positive_spread_fraction >= float(gate["minimum_positive_block_fraction"])
    pass_coverage = min_extreme_fraction >= float(gate["minimum_extreme_group_fraction_per_block"])
    passed = bool(pass_rho and pass_block_direction and pass_spread and pass_spread_blocks and pass_coverage)
    return pd.DataFrame(
        [
            {
                "pooled_far_residual_spearman": pooled["pressure_residual_spearman"],
                "positive_rho_block_fraction": positive_rho_fraction,
                "pooled_high_low_far_residual_spread_pct": pooled["high_minus_low_far_residual_pct"],
                "positive_spread_block_fraction": positive_spread_fraction,
                "minimum_extreme_group_fraction": min_extreme_fraction,
                "pass_pooled_rho": pass_rho,
                "pass_block_direction": pass_block_direction,
                "pass_pooled_spread": pass_spread,
                "pass_spread_blocks": pass_spread_blocks,
                "pass_coverage": pass_coverage,
                "diagnostic_gate_passed": passed,
                "model_experiment_allowed": bool(passed and gate["model_experiment_allowed_on_pass"]),
                "platform_submission_eligible": False,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = _validate(config, config_path)
    features = pd.read_csv(
        validated["feature_path"],
        usecols=["datetime", *validated["required_columns"]],
        parse_dates=["datetime"],
        low_memory=False,
    ).sort_values("datetime", kind="stable")
    if features["datetime"].duplicated().any():
        raise AssertionError("event feature timestamps are not unique")
    control = pd.read_csv(validated["control_oof_path"], parse_dates=["datetime"], low_memory=False)
    control = control.loc[control["variant"] == config["validation"]["control_variant"]].copy()
    expected_rows = len(validated["blocks"]) * 960 * 96
    if len(control) != expected_rows:
        raise AssertionError(f"control OOF row count {len(control)} != {expected_rows}")
    if control.duplicated(["fold_id", "datetime", "horizon_minutes"]).any():
        raise AssertionError("control OOF keys are duplicated")
    outcomes = _origin_outcomes(
        control,
        str(config["validation"]["near_bucket"]),
        str(config["validation"]["far_bucket"]),
    )

    per_fold: list[pd.DataFrame] = []
    contract_rows: list[dict[str, object]] = []
    contracts: dict[str, dict[str, float]] = {}
    for block in validated["blocks"]:
        joined, summary, contract = _build_fold_diagnostics(features, outcomes, block, config)
        per_fold.append(joined)
        contract_rows.append(summary)
        contracts[block["fold_id"]] = contract
    per_origin = pd.concat(per_fold, ignore_index=True)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"blocks={len(validated['blocks'])} origins={len(per_origin)} "
            f"features={len(validated['required_columns'])} models=0 "
            f"minimum_history_rows={min(int(row['history_rows']) for row in contract_rows)}",
            flush=True,
        )
        return

    identity_paths = [
        config_path,
        validated["feature_manifest_path"],
        validated["feature_registry_path"],
        validated["control_manifest_path"],
        validated["control_oof_path"],
        validated["feature_path"],
        RUNNER_SOURCE,
        HELPER_SOURCE,
    ]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "_holder2_boundary_" + digest.hexdigest()[:10]
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    event_log = output_dir / "events.jsonl"
    started_clock = time.perf_counter()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": config["task"],
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "diagnostic_version": config["diagnostic_version"],
        "control_artifact_id": config["control_artifact_id"],
        "control_scientific_run_id": config["control_scientific_run_id"],
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "model_count": 0,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths},
        "fold_contracts": contracts,
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False
    try:
        block_ids = [block["fold_id"] for block in validated["blocks"]]
        relationship_summary = summarize_pressure_relationships(per_origin, block_ids=block_ids)
        feature_correlations = _feature_correlations(per_origin)
        gate_results = _evaluate_gate(relationship_summary, config)
        frames = {
            "per_origin_diagnostics.csv": per_origin,
            "fold_boundary_contracts.csv": pd.DataFrame.from_records(contract_rows),
            "relationship_summary.csv": relationship_summary,
            "feature_correlations.csv": feature_correlations,
            "gate_results.csv": gate_results,
        }
        artifacts: dict[str, str] = {}
        for name, frame in frames.items():
            path = output_dir / name
            frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
            stem = name.removesuffix(".csv")
            artifacts[stem] = path.relative_to(PROJECT_ROOT).as_posix()
            artifacts[f"{stem}_sha256"] = sha256(path)
        gate = gate_results.iloc[0]
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "diagnostic_gate_passed": bool(gate["diagnostic_gate_passed"]),
                "model_experiment_allowed": bool(gate["model_experiment_allowed"]),
                "platform_submission_eligible": False,
                "artifacts": artifacts,
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
                    "model_name": "holder2_boundary_mechanism_audit",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"diagnostic_gate_passed={bool(gate['diagnostic_gate_passed'])}; models=0; no submission",
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", diagnostic_gate_passed=bool(gate["diagnostic_gate_passed"]))
        print("\n" + relationship_summary.to_string(index=False), flush=True)
        print("\n" + gate_results.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} "
            f"diagnostic_gate_passed={bool(gate['diagnostic_gate_passed'])} model_experiment_allowed={bool(gate['model_experiment_allowed'])}",
            flush=True,
        )
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "failed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", error_type=type(exc).__name__, message=str(exc))
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
                    "model_name": "holder2_boundary_mechanism_audit",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
