"""Run an expanding-time causal gate between v16 and capped dynamic short-g1."""

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
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io_utils import sha256, write_json
from src.round2_v3.run_short_g1_dynamic_fuel import evaluate_gate, summarize_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/short_g1_expert_gate_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCE = RUNNER_SOURCE.with_name("run_short_g1_dynamic_fuel.py")


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


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in manifest["partitions"] if row["partition"] == name]
    if len(rows) != 1:
        raise AssertionError(f"feature manifest has {len(rows)} {name} partitions")
    return rows[0]


def _run_id(config_path: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *paths]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_short_expert_gate_{digest.hexdigest()[:10]}"


def build_origin_gate_table(
    source: pd.DataFrame,
    control_variant: str,
    candidate_variant: str,
    horizons: list[int],
) -> pd.DataFrame:
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    control = source.loc[source["variant"] == control_variant, key + ["actual", "prediction"]].rename(columns={"prediction": "control_prediction"})
    candidate = source.loc[source["variant"] == candidate_variant, key + ["actual", "prediction"]].rename(columns={"actual": "candidate_actual", "prediction": "candidate_prediction"})
    merged = control.merge(candidate, on=key, how="inner", validate="one_to_one")
    if len(merged) != len(control) or len(merged) != len(candidate):
        raise AssertionError("control and candidate coverage differs")
    np.testing.assert_allclose(merged["actual"], merged["candidate_actual"], rtol=0, atol=1e-10)
    if sorted(merged["horizon_minutes"].unique()) != horizons:
        raise AssertionError("source horizons differ from preregistration")
    merged["control_ape"] = np.abs(merged["control_prediction"] - merged["actual"]) / np.maximum(np.abs(merged["actual"]), 1e-9)
    merged["candidate_ape"] = np.abs(merged["candidate_prediction"] - merged["actual"]) / np.maximum(np.abs(merged["actual"]), 1e-9)
    merged["disagreement"] = merged["candidate_prediction"] - merged["control_prediction"]
    rows: list[dict[str, object]] = []
    for (fold_id, origin), part in merged.groupby(["fold_id", "datetime"], sort=False):
        part = part.sort_values("horizon_minutes")
        if part["horizon_minutes"].tolist() != horizons:
            raise AssertionError("origin does not contain the full short horizon set")
        disagreement = part["disagreement"].to_numpy(float)
        control_prediction = part["control_prediction"].to_numpy(float)
        rows.append(
            {
                "fold_id": fold_id,
                "datetime": origin,
                "expert_win": int(part["candidate_ape"].mean() < part["control_ape"].mean()),
                "control_origin_mape": float(part["control_ape"].mean()),
                "candidate_origin_mape": float(part["candidate_ape"].mean()),
                "disagreement_mean": float(disagreement.mean()),
                "disagreement_abs_mean": float(np.abs(disagreement).mean()),
                "disagreement_std": float(disagreement.std()),
                "disagreement_min": float(disagreement.min()),
                "disagreement_max": float(disagreement.max()),
                "control_mean": float(control_prediction.mean()),
                "control_std": float(control_prediction.std()),
                "control_horizon_slope": float(control_prediction[-1] - control_prediction[0]),
            }
        )
    return pd.DataFrame.from_records(rows)


def soft_gate_predictions(control: np.ndarray, candidate: np.ndarray, probability: np.ndarray) -> np.ndarray:
    control = np.asarray(control, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if control.shape != candidate.shape or probability.shape != control.shape:
        raise ValueError("control, candidate, and probability must be shape aligned")
    if not np.isfinite(control).all() or not np.isfinite(candidate).all() or not np.isfinite(probability).all():
        raise ValueError("soft gate inputs must be finite")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("gate probability must be in [0,1]")
    return control + probability * (candidate - control)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: PROJECT_ROOT / value for key, value in config["inputs"].items()}
    source_manifest = _read_json(paths["source_manifest"])
    feature_manifest = _read_json(paths["feature_manifest"])
    if source_manifest["run_id"] != config["source_run_id"] or source_manifest["status"] != "completed":
        raise AssertionError("source run identity/status mismatch")
    if sha256(paths["source_oof"]) != source_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("source OOF hash mismatch")
    train_features_path = PROJECT_ROOT / _partition(feature_manifest, "train")["output"]
    if sha256(train_features_path) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("train feature hash mismatch")
    if sha256(paths["feature_registry"]) != feature_manifest["registry_sha256"]:
        raise AssertionError("feature registry hash mismatch")
    if feature_manifest["maximum_source_time_offset_minutes"] > 0 or feature_manifest["target_history_included"]:
        raise AssertionError("gate feature artifact violates causal contract")
    scope = config["scope"]
    if scope["future_fold_training_allowed"] or scope["future_process_observations_allowed"] or scope["true_target_history_at_inference_allowed"]:
        raise AssertionError("forbidden gate information was enabled")
    if config["logistic_regression"]["threshold_search"] or not config["logistic_regression"]["soft_probability_gate"]:
        raise AssertionError("gate must remain soft and threshold-free")
    horizons = [int(value) for value in scope["horizons_minutes"]]
    if horizons != list(range(15, 121, 15)):
        raise AssertionError("short horizon contract mismatch")
    source = pd.read_csv(paths["source_oof"], parse_dates=["datetime", "interval_start"])
    origin_table = build_origin_gate_table(source, scope["control_variant"], scope["source_candidate_variant"], horizons)
    feature_columns = [*config["gate_features"]["disagreement"], *config["gate_features"]["causal_process"]]
    process_columns = ["datetime", *config["gate_features"]["causal_process"]]
    origin_features = pd.read_csv(train_features_path, usecols=process_columns, parse_dates=["datetime"], low_memory=False)
    gate_table = origin_table.merge(origin_features, on="datetime", how="left", validate="many_to_one")
    if gate_table[feature_columns].isna().all(axis=0).any():
        raise AssertionError("at least one gate feature is entirely missing")
    training_folds = list(scope["training_folds"])
    validation_folds = list(scope["validation_folds"])
    if len(training_folds) != len(validation_folds) or len(validation_folds) != int(config["budget"]["sequential_validation_fold_count"]):
        raise AssertionError("sequential gate fold budget mismatch")
    sequence = list(dict.fromkeys([*training_folds, validation_folds[-1]]))
    observed_order = [fold for fold in source["fold_id"].drop_duplicates().tolist() if fold in sequence]
    if observed_order != sequence:
        raise AssertionError("gate fold chronology differs from source OOF")
    for index, validation_fold in enumerate(validation_folds):
        expected_training = sequence[: index + 1]
        if training_folds[: index + 1] != expected_training:
            raise AssertionError("gate training folds are not strictly expanding")
        if validation_fold in expected_training:
            raise AssertionError("validation fold leaked into gate training")
    if args.preflight_only:
        prevalence = gate_table.groupby("fold_id")["expert_win"].mean().to_dict()
        print(f"PASS preflight origins={len(gate_table)} features={len(feature_columns)} sequential_fits={len(validation_folds)} prevalence={prevalence} max_source_offset=0", flush=True)
        return

    identity_paths = [*paths.values(), train_features_path, RUNNER_SOURCE, HELPER_SOURCE]
    run_id = _run_id(config_path, identity_paths)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": run_id, "status": "running", "task": config["task"], "experiment_role": config["experiment_role"], "protocol_version": "round2_v3", "experiment_version": config["experiment_version"], "control_run_id": config["source_run_id"], "started_at": started_at.isoformat(), "holdout_evaluated": False, "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]}, "git": {"branch": _git_value("branch", "--show-current"), "commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))}, "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__}, "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, *identity_paths]}, "failure": {"type": None, "message": None, "traceback": None}, "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    registry_written = False
    try:
        params = dict(config["logistic_regression"])
        for key in ("soft_probability_gate", "threshold_search"):
            params.pop(key)
        gate_rows: list[pd.DataFrame] = []
        fit_rows: list[dict[str, object]] = []
        for index, validation_fold in enumerate(validation_folds):
            eligible_training = sequence[: index + 1]
            train = gate_table.loc[gate_table["fold_id"].isin(eligible_training)].copy()
            validation = gate_table.loc[gate_table["fold_id"] == validation_fold].copy()
            if train["expert_win"].nunique() != 2 or validation["expert_win"].nunique() != 2:
                raise AssertionError("gate train/validation label lacks both classes")
            model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler()), ("classifier", LogisticRegression(**params))])
            model.fit(train[feature_columns], train["expert_win"])
            probability = model.predict_proba(validation[feature_columns])[:, 1]
            validation = validation[["fold_id", "datetime", "expert_win"]].copy()
            validation["gate_probability"] = probability
            gate_rows.append(validation)
            fit_rows.append({"validation_fold": validation_fold, "training_folds": "+".join(eligible_training), "training_origins": len(train), "validation_origins": len(validation), "train_prevalence": float(train["expert_win"].mean()), "validation_prevalence": float(validation["expert_win"].mean()), "roc_auc": float(roc_auc_score(validation["expert_win"], probability)), "average_precision": float(average_precision_score(validation["expert_win"], probability)), "mean_probability": float(probability.mean())})
        gate_predictions = pd.concat(gate_rows, ignore_index=True)
        key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
        base = source.loc[(source["variant"] == scope["control_variant"]) & source["fold_id"].isin(validation_folds)].copy()
        alternate = source.loc[(source["variant"] == scope["source_candidate_variant"]) & source["fold_id"].isin(validation_folds), key + ["prediction"]].rename(columns={"prediction": "alternate_prediction"})
        base = base.merge(alternate, on=key, how="inner", validate="one_to_one")
        base = base.merge(gate_predictions[["fold_id", "datetime", "gate_probability"]], on=["fold_id", "datetime"], how="left", validate="many_to_one")
        gated = base.copy()
        gated["prediction"] = soft_gate_predictions(base["prediction"].to_numpy(float), base["alternate_prediction"].to_numpy(float), base["gate_probability"].to_numpy(float))
        gated["variant"] = scope["promotion_candidate"]
        keep = [column for column in source.columns if column in base.columns and column != "variant"]
        control_eval = base.copy(); control_eval["variant"] = "v16_control"
        evaluation = pd.concat([control_eval[["variant", *keep]], gated[["variant", *keep]]], ignore_index=True)
        metrics = summarize_predictions(evaluation)
        gate_config = {"variants": {"promotion_candidate": scope["promotion_candidate"]}, "scope": {"fold_ids": validation_folds, "horizons_minutes": horizons}, "selection_gate": config["selection_gate"]}
        gates = evaluate_gate(metrics, gate_config)
        pooled_auc = float(roc_auc_score(gate_predictions["expert_win"], gate_predictions["gate_probability"]))
        pooled_ap = float(average_precision_score(gate_predictions["expert_win"], gate_predictions["gate_probability"]))
        gates["gate_roc_auc"] = pooled_auc
        gates["gate_average_precision"] = pooled_ap
        gates["pass_gate_auc"] = pooled_auc >= float(config["selection_gate"]["minimum_gate_roc_auc"])
        gates["component_gate_passed"] = gates["component_gate_passed"].astype(bool) & gates["pass_gate_auc"].astype(bool)
        gates["standalone_submission_eligible"] = gates["component_gate_passed"].astype(bool)
        comparison = metrics.loc[metrics["scope"] == "overall", ["variant", "mape", "accuracy_1_minus_mape"]].copy()
        base_accuracy = float(comparison.loc[comparison["variant"] == "v16_control", "accuracy_1_minus_mape"].iloc[0])
        comparison["delta_accuracy_pct"] = (comparison["accuracy_1_minus_mape"] - base_accuracy) * 100
        print("\n" + comparison.sort_values("mape").to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        oof_path, metrics_path, gates_path, fits_path, origin_path = [output_dir / name for name in ("oof_predictions.csv", "metrics.csv", "gate_results.csv", "fit_summary.csv", "origin_gate_predictions.csv")]
        evaluation.to_csv(oof_path, index=False, encoding="utf-8", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        pd.DataFrame(fit_rows).to_csv(fits_path, index=False, encoding="utf-8", float_format="%.10f")
        gate_predictions.to_csv(origin_path, index=False, encoding="utf-8", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
        passed = bool(gates.iloc[0]["component_gate_passed"])
        primary_mape = float(comparison.loc[comparison["variant"] == scope["promotion_candidate"], "mape"].iloc[0])
        manifest.update({"status": "completed", "ended_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(time.perf_counter() - started, 3), "primary_variant": scope["promotion_candidate"], "primary_short_g1_mape": primary_mape, "component_gate_passed": passed, "standalone_submission_eligible": passed, "submission_eligible": False, "gate_roc_auc": pooled_auc, "gate_average_precision": pooled_ap, "gate_features": feature_columns, "artifacts": {"oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(), "oof_predictions_sha256": sha256(oof_path), "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(), "metrics_sha256": sha256(metrics_path), "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(), "gate_results_sha256": sha256(gates_path), "fit_summary": fits_path.relative_to(PROJECT_ROOT).as_posix(), "fit_summary_sha256": sha256(fits_path), "origin_gate_predictions": origin_path.relative_to(PROJECT_ROOT).as_posix(), "origin_gate_predictions_sha256": sha256(origin_path)}})
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["source_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "expanding_logistic_soft_expert_gate", "short_g1_mape": primary_mape, "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"component_gate_passed={passed}; gate_auc={pooled_auc:.4f}; gate_ap={pooled_ap:.4f}"})
            registry_written = True
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={passed}", flush=True)
    except Exception as exc:
        manifest["status"] = "failed"; manifest["ended_at"] = datetime.now(timezone.utc).isoformat(); manifest["duration_seconds"] = round(time.perf_counter() - started, 3); manifest["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}; write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["source_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "expanding_logistic_soft_expert_gate", "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
