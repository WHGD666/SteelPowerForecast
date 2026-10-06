"""Confirm the preregistered inverse holder-2 boundary signal on fresh blocks.

The D-078 audit rejected the preregistered positive direction but exposed a
consistent inverse relationship on all four development blocks.  This runner
tests that inverse hypothesis the only clean way available: rebuild the frozen
shared-horizon tree control on ten-day blocks that were never used in any prior
analysis, then apply the sign-flipped gate preregistered before any fresh-block
computation.  It trains no correction model and creates no submission.
"""

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

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.analog_trajectory_utils import build_future_trajectory_matrix
from src.round2_v3.baseline_utils import build_prediction_long
from src.round2_v3.holder_boundary_utils import (
    REQUIRED_COLUMNS,
    build_boundary_features,
    check_fresh_block_layout,
    evaluate_inverse_gate,
    fit_boundary_contract,
    origin_residual_outcomes,
    summarize_pressure_relationships,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/diagnostics/holder2_inverse_confirm_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("holder_boundary_utils.py"),
    RUNNER_SOURCE.with_name("analog_trajectory_utils.py"),
    RUNNER_SOURCE.with_name("baseline_utils.py"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


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


def _horizon_bucket(horizon: pd.Series) -> pd.Series:
    return pd.cut(
        horizon,
        bins=[0, 120, 360, 720, 1440],
        labels=["h015_120", "h135_360", "h375_720", "h735_1440"],
        include_lowest=True,
    ).astype(str)


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [item for item in manifest["partitions"] if item["partition"] == name]
    if len(rows) != 1:
        raise AssertionError(f"feature manifest has {len(rows)} {name} partitions")
    return rows[0]


def _block_history_days(features: pd.DataFrame, start: pd.Timestamp) -> float:
    history = features.loc[features["datetime"] < start, "datetime"]
    if history.empty:
        return 0.0
    return float((start - history.min()).total_seconds() / 86400.0)


def _validate(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    feature_manifest_path = PROJECT_ROOT / config["inputs"]["event_feature_manifest"]
    feature_registry_path = PROJECT_ROOT / config["inputs"]["event_feature_registry"]
    label_manifest_path = PROJECT_ROOT / config["inputs"]["label_manifest"]
    feature_manifest = _read_json(feature_manifest_path)
    label_manifest = _read_json(label_manifest_path)
    feature_path = PROJECT_ROOT / _partition(feature_manifest, "train")["output"]
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(feature_path) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("event feature artifact hash mismatch")
    if sha256(feature_registry_path) != feature_manifest["registry_sha256"]:
        raise AssertionError("event feature registry hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if feature_manifest["target_history_included"] or feature_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("event features violate the causal information contract")
    if config["validation"]["true_target_history_allowed"] or config["validation"]["future_process_observations_allowed"]:
        raise AssertionError("target history and future process observations must be forbidden")

    registry = pd.read_csv(feature_registry_path)
    tree_features = list(config["tree_features"])
    boundary_columns = list(config["boundary_contract"]["required_columns"])
    if tuple(boundary_columns) != tuple(REQUIRED_COLUMNS):
        raise AssertionError("boundary required-column contract drifted")
    for column in [*tree_features, *boundary_columns]:
        chosen = registry.loc[registry["feature_name"] == column]
        if len(chosen) != 1:
            raise KeyError(f"feature absent from registry: {column}")
        if not bool(chosen.iloc[0]["causal"]) or bool(chosen.iloc[0]["target_history"]):
            raise AssertionError(f"feature violates causal contract: {column}")
    expected_weights = {
        "centered_position": 0.45,
        "multiscale_velocity": 0.25,
        "signed_boundary_dwell": 0.20,
        "gas_balance_state": 0.10,
    }
    if config["boundary_contract"]["score_weights"] != expected_weights:
        raise AssertionError("fixed pressure-score weights changed")

    blocks = check_fresh_block_layout(
        config["validation"]["blocks"], config["validation"]["dev_blocks_forbidden_overlap"]
    )
    return {
        "config_path": config_path,
        "feature_manifest_path": feature_manifest_path,
        "feature_registry_path": feature_registry_path,
        "label_manifest_path": label_manifest_path,
        "feature_manifest": feature_manifest,
        "label_manifest": label_manifest,
        "feature_path": feature_path,
        "label_path": label_path,
        "tree_features": tree_features,
        "boundary_columns": boundary_columns,
        "blocks": blocks,
    }


def _build_block_control(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    block: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object], int]:
    tree_config = config["tree_control"]
    horizons = [int(value) for value in config["validation"]["horizons_minutes"]]
    target = str(config["validation"]["target"])
    latest_library_origin = block["start"] - pd.Timedelta(minutes=1440)
    feature_index = features.set_index("datetime")
    eligible = feature_index.index[feature_index.index <= latest_library_origin]
    stride = int(tree_config["library_origin_stride_minutes"])
    epoch = eligible.view("int64") // (60 * 1_000_000_000)
    library_origins = eligible[epoch % stride == 0]
    library_trajectory = build_future_trajectory_matrix(labels, library_origins, horizons, target)
    complete = np.isfinite(library_trajectory).all(axis=1)
    library_origins = library_origins[complete]
    library_trajectory = library_trajectory[complete]
    if len(library_origins) < int(config["validation"]["minimum_library_origins"]):
        raise AssertionError(f"{block['fold_id']} has insufficient library origins")

    actual = build_future_trajectory_matrix(labels, block["origins"], horizons, target)
    missing_cells = int((~np.isfinite(actual)).sum())

    x_train, _ = build_prediction_long(features, library_origins, horizons, _tree_feature_names(config))
    y_train = library_trajectory.reshape(-1)
    x_validation, validation_meta = build_prediction_long(
        features, block["origins"], horizons, _tree_feature_names(config)
    )
    model_config = dict(tree_config["lightgbm"])
    weight_mode = model_config.pop("sample_weight")
    weight_floor = float(model_config.pop("sample_weight_floor"))
    sample_weight = None
    if weight_mode == "inverse_absolute_target":
        sample_weight = 1.0 / np.maximum(np.abs(y_train), weight_floor)
    elif weight_mode != "none":
        raise ValueError(f"unsupported sample weight mode: {weight_mode}")
    model = lgb.LGBMRegressor(**model_config)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    prediction = model.predict(x_validation).reshape(actual.shape)

    oof = validation_meta.copy()
    oof["fold_id"] = block["fold_id"]
    oof["actual"] = actual.reshape(-1)
    oof["prediction"] = prediction.reshape(-1)
    oof["horizon_bucket"] = _horizon_bucket(oof["horizon_minutes"])

    finite = np.isfinite(oof["actual"].to_numpy(dtype=float)) & (
        oof["actual"].to_numpy(dtype=float) != 0
    )
    control_mape = float(
        np.mean(
            np.abs(
                (oof["actual"].to_numpy(dtype=float)[finite] - prediction.reshape(-1)[finite])
                / oof["actual"].to_numpy(dtype=float)[finite]
            )
        )
    )
    summary = {
        "fold_id": block["fold_id"],
        "validation_start": block["start"],
        "validation_end": block["end"],
        "latest_library_origin": library_origins.max(),
        "library_origins": len(library_origins),
        "training_rows": len(y_train),
        "validation_origins": len(block["origins"]),
        "missing_validation_label_cells": missing_cells,
        "control_mape": control_mape,
        "tree_feature_count": x_train.shape[1],
    }
    return oof, summary, missing_cells


def _tree_feature_names(config: dict[str, Any]) -> list[str]:
    return list(config["tree_features"])


def _build_block_boundary(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    block: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    boundary_config = config["boundary_contract"]
    history = features.loc[features["datetime"] < block["start"]]
    minimum_rows = 96 * int(config["validation"]["minimum_boundary_history_days"])
    if len(history) < minimum_rows:
        raise AssertionError(f"{block['fold_id']} has less than the required boundary history")
    contract = fit_boundary_contract(
        history,
        reference_quantiles=boundary_config["reference_quantiles"],
        zone_quantiles=boundary_config["zone_quantiles"],
        scale_floor=float(boundary_config["robust_scale_floor"]),
    )
    through_validation = features.loc[features["datetime"] <= block["end"]]
    boundary = build_boundary_features(
        through_validation[list(REQUIRED_COLUMNS)],
        contract,
        maximum_dwell_steps=int(boundary_config["maximum_dwell_steps"]),
    )
    boundary.insert(0, "datetime", through_validation["datetime"].to_numpy())
    history_scores = boundary.loc[boundary["datetime"] < block["start"], "boundary_pressure_score"]
    score_low_q, score_high_q = [float(v) for v in boundary_config["score_extreme_quantiles"]]
    score_low, score_high = np.quantile(history_scores.to_numpy(dtype=float), [score_low_q, score_high_q])

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
        "history_start": history["datetime"].min(),
        "validation_origins": len(validation),
        "reference_low": contract["reference_low"],
        "zone_low": contract["zone_low"],
        "zone_high": contract["zone_high"],
        "reference_high": contract["reference_high"],
        "score_low": float(score_low),
        "score_high": float(score_high),
        "validation_low_score_fraction": float((validation["score_group"] == "low").mean()),
        "validation_high_score_fraction": float((validation["score_group"] == "high").mean()),
    }
    return joined, summary


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
        usecols=["datetime", *validated["tree_features"], *validated["boundary_columns"]],
        parse_dates=["datetime"],
        low_memory=False,
    ).sort_values("datetime", kind="stable")
    if features["datetime"].duplicated().any():
        raise AssertionError("event feature timestamps are not unique")
    labels = pd.read_csv(validated["label_path"], parse_dates=["datetime"], low_memory=False)

    if args.preflight_only:
        rows: list[dict[str, object]] = []
        for block in validated["blocks"]:
            days = _block_history_days(features, block["start"])
            tree_config = config["tree_control"]
            horizons = [int(v) for v in config["validation"]["horizons_minutes"]]
            latest = block["start"] - pd.Timedelta(minutes=1440)
            index = features.set_index("datetime").index
            eligible = index[index <= latest]
            epoch = eligible.view("int64") // (60 * 1_000_000_000)
            stride = int(tree_config["library_origin_stride_minutes"])
            library = eligible[epoch % stride == 0]
            trajectory = build_future_trajectory_matrix(
                labels, library, horizons, str(config["validation"]["target"])
            )
            library = library[np.isfinite(trajectory).all(axis=1)]
            actual = build_future_trajectory_matrix(
                labels, block["origins"], horizons, str(config["validation"]["target"])
            )
            rows.append(
                {
                    "fold_id": block["fold_id"],
                    "boundary_history_days": round(days, 2),
                    "library_origins": len(library),
                    "validation_origins": len(block["origins"]),
                    "missing_validation_label_cells": int((~np.isfinite(actual)).sum()),
                }
            )
        print(
            "PASS preflight "
            f"blocks={len(validated['blocks'])} origins={sum(int(r['validation_origins']) for r in rows)} "
            f"tree_features={len(validated['tree_features'])} boundary_features={len(validated['boundary_columns'])} "
            f"models={int(config['budget']['model_count'])} "
            f"minimum_history_days={min(float(r['boundary_history_days']) for r in rows):.1f} "
            f"minimum_library_origins={min(int(r['library_origins']) for r in rows)} "
            f"missing_validation_label_cells={sum(int(r['missing_validation_label_cells']) for r in rows)}",
            flush=True,
        )
        return

    identity_paths = [
        config_path,
        validated["feature_manifest_path"],
        validated["feature_registry_path"],
        validated["label_manifest_path"],
        validated["feature_path"],
        validated["label_path"],
        RUNNER_SOURCE,
        *HELPER_SOURCES,
    ]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "_holder2_inverse_confirm_" + digest.hexdigest()[:10]
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
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "model_count": int(config["budget"]["model_count"]),
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths},
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False
    try:
        oof_parts: list[pd.DataFrame] = []
        per_fold: list[pd.DataFrame] = []
        contract_rows: list[dict[str, object]] = []
        fit_rows: list[dict[str, object]] = []
        for number, block in enumerate(validated["blocks"], 1):
            oof, fit_summary, _missing = _build_block_control(features, labels, block, config)
            oof_parts.append(oof)
            fit_rows.append(fit_summary)
            print(
                f"[{number}/{len(validated['blocks'])}] {block['fold_id']}: "
                f"library_origins={fit_summary['library_origins']} training_rows={fit_summary['training_rows']} "
                f"control_mape={fit_summary['control_mape']:.6f} missing_cells={fit_summary['missing_validation_label_cells']}",
                flush=True,
            )
        control_oof = pd.concat(oof_parts, ignore_index=True)
        outcomes = origin_residual_outcomes(
            control_oof,
            near_bucket=str(config["validation"]["near_bucket"]),
            far_bucket=str(config["validation"]["far_bucket"]),
        )
        for block in validated["blocks"]:
            joined, contract_summary = _build_block_boundary(features, outcomes, block, config)
            per_fold.append(joined)
            contract_rows.append(contract_summary)
        per_origin = pd.concat(per_fold, ignore_index=True)
        block_ids = [block["fold_id"] for block in validated["blocks"]]
        relationship_summary = summarize_pressure_relationships(per_origin, block_ids=block_ids)
        gate_results = evaluate_inverse_gate(relationship_summary, config["inverse_diagnostic_gate"])

        frames = {
            "oof_predictions.csv": control_oof,
            "per_origin_diagnostics.csv": per_origin,
            "fold_boundary_contracts.csv": pd.DataFrame.from_records(contract_rows),
            "fit_summary.csv": pd.DataFrame.from_records(fit_rows),
            "relationship_summary.csv": relationship_summary,
            "gate_results.csv": gate_results,
        }
        artifacts: dict[str, str] = {}
        for name, frame in frames.items():
            path = output_dir / name
            frame.to_csv(
                path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f"
            )
            stem = name.removesuffix(".csv")
            artifacts[stem] = path.relative_to(PROJECT_ROOT).as_posix()
            artifacts[stem + "_sha256"] = sha256(path)
        gate = gate_results.iloc[0]
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "inverse_diagnostic_gate_passed": bool(gate["inverse_diagnostic_gate_passed"]),
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
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "holder2_inverse_boundary_confirmation",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"inverse_diagnostic_gate_passed={bool(gate['inverse_diagnostic_gate_passed'])}; fresh blocks only; no submission",
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", inverse_diagnostic_gate_passed=bool(gate["inverse_diagnostic_gate_passed"]))
        print("\n" + relationship_summary.to_string(index=False), flush=True)
        print("\n" + gate_results.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} "
            f"inverse_diagnostic_gate_passed={bool(gate['inverse_diagnostic_gate_passed'])} "
            f"model_experiment_allowed={bool(gate['model_experiment_allowed'])}",
            flush=True,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
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
                    "model_name": "holder2_inverse_boundary_confirmation",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": manifest["ended_at"],
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"failure={type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
