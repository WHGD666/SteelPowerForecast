"""Evaluate a MAPE-aligned fuel proxy inside the frozen v28 long-g1 recipe."""

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
    FUELS_G1,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
    v28_g1_fuel_weight,
)
from src.round2_v3.event_ablation_utils import attach_event_context
from src.round2_v3.mape_fuel_utils import (
    fit_mape_median_fuel_proxy,
    predict_mape_median_fuel_proxy,
)
from src.round2_v3.run_anomaly_weight_g1 import (
    _append_event,
    _append_registry,
    _git_value,
    _summarize_metrics,
)
from src.round2_v3.run_long_g1_mape_objective import _evaluate_gate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "long_g1_mape_fuel_v1.yaml"
)
RUNNER_SOURCE = Path(__file__).resolve()
UTILITY_SOURCES = [
    RUNNER_SOURCE.with_name("mape_fuel_utils.py"),
    RUNNER_SOURCE.with_name("mape_objective_utils.py"),
    RUNNER_SOURCE.with_name("anomaly_weight_utils.py"),
    RUNNER_SOURCE.with_name("event_ablation_utils.py"),
    RUNNER_SOURCE.with_name("run_anomaly_weight_g1.py"),
    RUNNER_SOURCE.with_name("run_long_g1_mape_objective.py"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_identity(config_path: Path, inputs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *inputs]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_long_g1_mape_fuel_{digest.hexdigest()[:10]}"


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    raw_manifest_path = PROJECT_ROOT / inputs["raw_fingerprints"]
    raw_manifest = _read_json(raw_manifest_path)
    expected = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
    raw_tables = [PROJECT_ROOT / item for item in inputs["raw_train_tables"]]
    for path in raw_tables:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if expected.get(relative) != sha256(path):
            raise AssertionError(f"raw fingerprint mismatch: {relative}")

    label_manifest_path = PROJECT_ROOT / inputs["label_manifest"]
    split_manifest_path = PROJECT_ROOT / inputs["split_manifest"]
    label_manifest = _read_json(label_manifest_path)
    split_manifest = _read_json(split_manifest_path)
    label_path = PROJECT_ROOT / label_manifest["output"]
    split_path = PROJECT_ROOT / inputs["split_assignments"]
    control_path = PROJECT_ROOT / inputs["control_oof"]
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if sha256(split_path) != split_manifest["assignment_sha256"]:
        raise AssertionError("split artifact hash mismatch")
    if sha256(control_path) != inputs["control_oof_sha256"]:
        raise AssertionError("control OOF hash mismatch")

    experiment = config["experiment"]
    if experiment["target"] != "generator_1":
        raise AssertionError("target must remain generator_1")
    if experiment["tree_recipe"] != "v28_exact_frozen_from_control_oof":
        raise AssertionError("tree must stay frozen")
    if experiment["horizon_blend_recipe"] != "v28_exact_0.8_near_0.6_far":
        raise AssertionError("horizon blend differs from v28")
    if (
        experiment["future_process_observations_allowed"]
        or experiment["true_target_history_features_allowed"]
        or experiment["platform_feedback_used_for_fit"]
    ):
        raise AssertionError("information boundary violated")
    training = config["training"]
    expected_training = {
        "denominator_floor_mw": 20.0,
        "quantile": 0.5,
        "alpha": 0.0,
        "solver": "highs",
        "normalize_sample_weight_mean": True,
    }
    for key, value in expected_training.items():
        if training[key] != value:
            raise AssertionError(f"v1 frozen setting differs: {key}")
    folds = list(config["validation"]["fold_ids"])
    if len(folds) != int(config["budget"]["fuel_model_count"]):
        raise AssertionError("fuel model budget mismatch")
    if int(config["budget"]["lightgbm_model_count"]) != 0:
        raise AssertionError("this experiment must not train LightGBM")
    return {
        "raw_tables": raw_tables,
        "raw_manifest_path": raw_manifest_path,
        "label_manifest": label_manifest,
        "label_manifest_path": label_manifest_path,
        "label_path": label_path,
        "split_manifest": split_manifest,
        "split_manifest_path": split_manifest_path,
        "split_path": split_path,
        "control_path": control_path,
        "event_path": PROJECT_ROOT / inputs["event_episodes"],
        "legacy_source": PROJECT_ROOT / inputs["legacy_v28_source"],
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = _validate(config)
    if args.preflight_only:
        print(
            "PASS preflight folds=3 fuel_models=3 lightgbm_models=0 "
            "control_hash=true tree_frozen=true create_submission=false",
            flush=True,
        )
        return

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    identity_inputs = [
        *validated["raw_tables"],
        validated["raw_manifest_path"],
        validated["label_manifest_path"],
        validated["label_path"],
        validated["split_manifest_path"],
        validated["split_path"],
        validated["control_path"],
        validated["event_path"],
        validated["legacy_source"],
        RUNNER_SOURCE,
        *UTILITY_SOURCES,
    ]
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
        "experiment_version": config["experiment_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "holdout_evaluated": False,
        "submission_eligible": False,
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
        raw = merge_training_tables(validated["raw_tables"])
        grid = to_v28_grid(raw)
        split = pd.read_csv(validated["split_path"])
        split["train_end"] = pd.to_datetime(split["train_end"], errors="raise")
        split["validation_origin"] = pd.to_datetime(
            split["validation_origin"], errors="raise"
        )
        split = split.loc[
            (split["role"] == "model_selection")
            & split["fold_id"].isin(validated["folds"])
        ].copy()
        if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
            raise AssertionError("validation fold order differs")

        control = pd.read_csv(
            validated["control_path"],
            parse_dates=["datetime", "interval_start"],
            low_memory=False,
        )
        base_columns = [
            "variant",
            "fold_id",
            "datetime",
            "interval_start",
            "horizon_minutes",
            "actual_interval_mean",
            "actual_legacy_point",
            "prediction",
        ]
        control = control.loc[
            control["variant"].eq(config["experiment"]["control_variant"])
            & control["fold_id"].isin(validated["folds"]),
            base_columns,
        ].copy()
        control["variant"] = "mae_control"

        fuel_start = pd.Timestamp(config["training"]["fuel_fit_start"])
        candidate_parts: list[pd.DataFrame] = []
        fit_records: list[dict[str, object]] = []
        for fold_number, (fold_id, fold) in enumerate(
            split.groupby("fold_id", sort=False), start=1
        ):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            training = grid.loc[grid["datetime"].between(fuel_start, train_end)].copy()
            ols_coefficients = fit_linear_fuel_proxy(
                training, "generator_1", FUELS_G1
            )
            median_model = fit_mape_median_fuel_proxy(
                training,
                target="generator_1",
                fuel_columns=FUELS_G1,
                denominator_floor=float(config["training"]["denominator_floor_mw"]),
            )
            fold_control = control.loc[control["fold_id"].eq(fold_id)].copy()
            origins = fold_control[["datetime"]].drop_duplicates().sort_values("datetime")
            origin_frame = origins.merge(
                grid[["datetime", *FUELS_G1]],
                on="datetime",
                how="left",
                validate="one_to_one",
            )
            old_fuel = predict_linear_fuel(
                origin_frame, FUELS_G1, ols_coefficients
            )
            new_fuel = predict_mape_median_fuel_proxy(
                median_model, origin_frame, fuel_columns=FUELS_G1
            )
            old_map = pd.Series(old_fuel, index=origin_frame["datetime"])
            new_map = pd.Series(new_fuel, index=origin_frame["datetime"])
            candidate = fold_control.copy()
            candidate["variant"] = config["experiment"]["candidate_variant"]
            weight = candidate["horizon_minutes"].map(v28_g1_fuel_weight).to_numpy(float)
            delta = (
                candidate["datetime"].map(new_map).to_numpy(float)
                - candidate["datetime"].map(old_map).to_numpy(float)
            )
            candidate["prediction"] = np.clip(
                candidate["prediction"].to_numpy(float) + weight * delta,
                0.0,
                None,
            )
            if not np.isfinite(candidate["prediction"]).all():
                raise AssertionError(f"{fold_id}: non-finite candidate prediction")
            candidate_parts.append(candidate)
            fit_records.append(
                {
                    "fold_id": fold_id,
                    "training_rows": len(training),
                    "train_end": train_end,
                    "mean_old_fuel": float(np.mean(old_fuel)),
                    "mean_new_fuel": float(np.mean(new_fuel)),
                    "mean_delta_mw": float(np.mean(new_fuel - old_fuel)),
                    "mean_absolute_delta_mw": float(np.mean(np.abs(new_fuel - old_fuel))),
                    "max_absolute_delta_mw": float(np.max(np.abs(new_fuel - old_fuel))),
                }
            )
            print(
                f"[{fold_number}/3] {fold_id}: train_rows={len(training)} "
                f"mean_fuel_delta={np.mean(new_fuel-old_fuel):.6f} MW",
                flush=True,
            )
            _append_event(event_log, "fold_completed", fold_id=fold_id)

        candidate = pd.concat(candidate_parts, ignore_index=True)
        oof = pd.concat([control, candidate], ignore_index=True)
        oof = attach_event_context(
            oof, pd.read_csv(validated["event_path"]), context_hours=24
        )
        metrics = _summarize_metrics(oof)
        gate_result = _evaluate_gate(metrics, config)
        gate = pd.DataFrame([gate_result])
        overall = metrics.loc[
            (metrics["label_variant"] == "official_interval_mean")
            & (metrics["scope"] == "overall")
        ].copy()
        control_accuracy = overall.loc[
            overall["variant"].eq("mae_control"),
            ["period", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
        comparison = overall.merge(control_accuracy, on="period", validate="many_to_one")
        comparison["delta_accuracy_pct"] = (
            comparison["accuracy_1_minus_mape"] - comparison["control_accuracy"]
        ) * 100.0
        comparison = comparison[
            ["period", "variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]
        ].sort_values(["period", "mape"])

        paths = {
            "oof_predictions": output_dir / "oof_predictions.csv",
            "fit_summary": output_dir / "fit_summary.csv",
            "metrics": output_dir / "metrics.csv",
            "variant_comparison": output_dir / "variant_comparison.csv",
            "gate_results": output_dir / "gate_results.csv",
        }
        oof.to_csv(paths["oof_predictions"], index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
        pd.DataFrame(fit_records).to_csv(paths["fit_summary"], index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
        metrics.to_csv(paths["metrics"], index=False, float_format="%.10f")
        comparison.to_csv(paths["variant_comparison"], index=False, float_format="%.10f")
        gate.to_csv(paths["gate_results"], index=False, float_format="%.10f")

        ended = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended.isoformat(),
                "duration_seconds": round(time.perf_counter() - clock, 3),
                "fuel_model_count": int(config["budget"]["fuel_model_count"]),
                "lightgbm_model_count": 0,
                "component_gate_passed": bool(gate_result["component_gate_passed"]),
                "standalone_submission_eligible": bool(gate_result["standalone_submission_eligible"]),
                "artifacts": {
                    name: path.relative_to(PROJECT_ROOT).as_posix()
                    for name, path in paths.items()
                },
            }
        )
        manifest["artifact_sha256"] = {
            name: sha256(path) for name, path in paths.items()
        }
        write_json(output_dir / "run_manifest.json", manifest)
        candidate_long = comparison.loc[
            (comparison["period"] == "long")
            & comparison["variant"].eq(config["experiment"]["candidate_variant"])
        ].iloc[0]
        if config["output"]["append_registry"]:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "split_sha256": validated["split_manifest"]["assignment_sha256"],
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "v28_g1_mape_weighted_median_fuel",
                    "long_g1_mape": float(candidate_long["mape"]),
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": (
                        f"component_gate_passed={gate_result['component_gate_passed']}; "
                        f"standalone_submission_eligible={gate_result['standalone_submission_eligible']}"
                    ),
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", **gate_result)
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gate.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} "
            f"component_gate_passed={gate_result['component_gate_passed']} "
            f"standalone_submission_eligible={gate_result['standalone_submission_eligible']}",
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
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "split_sha256": validated["split_manifest"]["assignment_sha256"],
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "v28_g1_mape_weighted_median_fuel",
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

