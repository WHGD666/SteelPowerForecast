"""Calibrate OOF comparisons against unlabeled October process-state similarity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io_utils import sha256, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/diagnostics/test_similarity_calibration_v1.yaml"
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


def normalized_similarity_weights(
    probabilities: np.ndarray,
    *,
    probability_clip: tuple[float, float],
    weight_clip: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("domain probabilities must be a finite vector")
    low_p, high_p = map(float, probability_clip)
    low_w, high_w = map(float, weight_clip)
    if not 0 < low_p < high_p < 1 or not 0 < low_w <= high_w:
        raise ValueError("invalid probability or weight clip")
    clipped = np.clip(values, low_p, high_p)
    odds = clipped / (1.0 - clipped)
    normalized = odds / odds.mean()
    bounded = np.clip(normalized, low_w, high_w)
    return bounded / bounded.mean()


def weighted_accuracy_gain_pct(
    frame: pd.DataFrame,
    *,
    control_variant: str,
    candidate_variant: str,
    actual_column: str,
    weight_column: str,
    actual_floor: float,
) -> float:
    keys = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    columns = [*keys, actual_column, "prediction", weight_column]
    control = frame.loc[frame["variant"] == control_variant, columns].rename(columns={"prediction": "control_prediction", weight_column: "control_weight"})
    candidate = frame.loc[frame["variant"] == candidate_variant, columns].rename(columns={"prediction": "candidate_prediction", weight_column: "candidate_weight", actual_column: "candidate_actual"})
    merged = control.merge(candidate, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(control) or len(merged) != len(candidate):
        raise AssertionError("control/candidate comparison keys differ")
    actual = pd.to_numeric(merged[actual_column], errors="coerce").to_numpy(dtype=float)
    if not np.allclose(actual, pd.to_numeric(merged["candidate_actual"], errors="coerce"), equal_nan=True):
        raise AssertionError("control/candidate actual values differ")
    control_weight = merged["control_weight"].to_numpy(dtype=float)
    candidate_weight = merged["candidate_weight"].to_numpy(dtype=float)
    if not np.allclose(control_weight, candidate_weight):
        raise AssertionError("control/candidate similarity weights differ")
    denominator = np.maximum(np.abs(actual), float(actual_floor))
    valid = np.isfinite(actual) & np.isfinite(control_weight) & (control_weight > 0)
    control_ape = np.abs(merged["control_prediction"].to_numpy(dtype=float) - actual) / denominator
    candidate_ape = np.abs(merged["candidate_prediction"].to_numpy(dtype=float) - actual) / denominator
    control_mape = np.average(control_ape[valid], weights=control_weight[valid])
    candidate_mape = np.average(candidate_ape[valid], weights=control_weight[valid])
    return float((control_mape - candidate_mape) * 100.0)


def _pipeline(config: dict[str, Any]) -> Pipeline:
    model = config["domain_model"]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=float(model["C"]), class_weight=model["class_weight"], max_iter=int(model["max_iter"]), random_state=int(model["random_state"]))),
        ]
    )


def _load_features(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any], Path, Path, Path]:
    manifest_path = PROJECT_ROOT / config["features"]["manifest"]
    registry_path = PROJECT_ROOT / config["features"]["registry"]
    manifest = _read_json(manifest_path)
    if sha256(registry_path) != manifest["registry_sha256"]:
        raise AssertionError("feature registry hash mismatch")
    if manifest["target_history_included"] or not manifest["all_features_causal"] or manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("feature manifest violates causal domain contract")
    registry = pd.read_csv(registry_path)
    allowed = set(config["features"]["allowed_transforms"])
    forbidden = set(config["features"]["forbidden_transforms"])
    selected = registry.loc[registry["transform"].isin(allowed) & ~registry["transform"].isin(forbidden), "feature_name"].tolist()
    if not selected or registry.loc[registry["feature_name"].isin(selected), "transform"].isin(forbidden).any():
        raise AssertionError("domain feature selection is empty or contains forbidden transforms")
    parts = {item["partition"]: item for item in manifest["partitions"]}
    train_path, test_path = PROJECT_ROOT / parts["train"]["output"], PROJECT_ROOT / parts["test"]["output"]
    if sha256(train_path) != parts["train"]["output_sha256"] or sha256(test_path) != parts["test"]["output_sha256"]:
        raise AssertionError("origin feature artifact hash mismatch")
    usecols = ["datetime", *selected]
    train = pd.read_csv(train_path, usecols=usecols, parse_dates=["datetime"], low_memory=False)
    test = pd.read_csv(test_path, usecols=usecols, parse_dates=["datetime"], low_memory=False)
    return train, test, selected, manifest, manifest_path, registry_path, train_path


def _direction_pass(expected: str, weighted_gain: float, gate: dict[str, Any]) -> bool:
    if expected == "negative":
        return weighted_gain <= float(gate["negative_max_gain_pct"])
    if expected == "neutral":
        return abs(weighted_gain) <= float(gate["neutral_max_absolute_gain_pct"])
    raise ValueError(f"unsupported expected direction: {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train, test, feature_columns, feature_manifest, feature_manifest_path, registry_path, train_path = _load_features(config)
    test_path = PROJECT_ROOT / {item["partition"]: item for item in feature_manifest["partitions"]}["test"]["output"]
    probe_paths = {name: PROJECT_ROOT / spec["oof"] for name, spec in config["platform_probes"].items()}
    for name, path in probe_paths.items():
        if sha256(path) != config["platform_probes"][name]["sha256"]:
            raise AssertionError(f"probe OOF hash mismatch: {name}")
    if args.preflight_only:
        print(f"PASS preflight train_rows={len(train)} test_rows={len(test)} domain_features={len(feature_columns)} probes={len(probe_paths)} calendar_features=false", flush=True)
        return

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    domain = pd.concat([train.assign(domain=0), test.assign(domain=1)], ignore_index=True)
    x = domain[feature_columns]
    y = domain["domain"].to_numpy(dtype=int)
    groups = domain["datetime"].dt.floor("D").astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=int(config["domain_model"]["grouped_cv_splits"]), shuffle=True, random_state=int(config["domain_model"]["random_state"]))
    oof_probability = np.full(len(domain), np.nan, dtype=float)
    for train_index, validation_index in cv.split(x, y, groups):
        model = _pipeline(config)
        model.fit(x.iloc[train_index], y[train_index])
        oof_probability[validation_index] = model.predict_proba(x.iloc[validation_index])[:, 1]
    domain_auc = float(roc_auc_score(y, oof_probability))
    model = _pipeline(config)
    model.fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    domain_scores = domain[["datetime", "domain"]].copy()
    domain_scores["test_probability"] = probability
    train_scores = domain_scores.loc[domain_scores["domain"] == 0, ["datetime", "test_probability"]].copy()

    rows, fold_rows = [], []
    folds = list(config["validation"]["fold_ids"])
    weight_config = config["weighting"]
    for name, spec in config["platform_probes"].items():
        frame = pd.read_csv(probe_paths[name], parse_dates=["datetime", "interval_start"], low_memory=False)
        frame = frame.loc[frame["fold_id"].isin(folds) & frame["variant"].isin([spec["control_variant"], spec["candidate_variant"]])].copy()
        frame = frame.merge(train_scores, on="datetime", how="left", validate="many_to_one")
        if frame["test_probability"].isna().any():
            raise AssertionError(f"{name}: OOF origins missing domain scores")
        frame["uniform_weight"] = 1.0
        frame["similarity_weight"] = np.nan
        frame["top_similarity_weight"] = 0.0
        for fold_id, index in frame.groupby("fold_id", sort=False).groups.items():
            origin = frame.loc[index, ["datetime", "test_probability"]].drop_duplicates("datetime").sort_values("datetime")
            weights = normalized_similarity_weights(origin["test_probability"].to_numpy(), probability_clip=tuple(weight_config["probability_clip"]), weight_clip=tuple(weight_config["normalized_odds_clip"]))
            threshold = float(origin["test_probability"].quantile(1.0 - float(weight_config["top_similarity_fraction"])))
            mapping = pd.Series(weights, index=origin["datetime"])
            frame.loc[index, "similarity_weight"] = frame.loc[index, "datetime"].map(mapping).to_numpy(dtype=float)
            frame.loc[index, "top_similarity_weight"] = (frame.loc[index, "test_probability"] >= threshold).astype(float)
            ess = float(weights.sum() ** 2 / np.square(weights).sum())
            fold_rows.append({"probe": name, "fold_id": fold_id, "origin_count": len(origin), "mean_test_probability": float(origin["test_probability"].mean()), "max_test_probability": float(origin["test_probability"].max()), "effective_sample_size": ess, "effective_sample_fraction": ess / len(origin), "top_similarity_threshold": threshold})
        common = {"frame": frame, "control_variant": spec["control_variant"], "candidate_variant": spec["candidate_variant"], "actual_column": spec["actual_column"], "actual_floor": float(config["validation"]["actual_floor"])}
        unweighted = weighted_accuracy_gain_pct(**common, weight_column="uniform_weight")
        weighted = weighted_accuracy_gain_pct(**common, weight_column="similarity_weight")
        top = weighted_accuracy_gain_pct(**common, weight_column="top_similarity_weight")
        pass_direction = _direction_pass(spec["expected_weighted_direction"], weighted, config["calibration_gate"])
        rows.append({"probe": name, "unweighted_accuracy_gain_pct": unweighted, "similarity_weighted_accuracy_gain_pct": weighted, "top_quartile_accuracy_gain_pct": top, "official_score_delta": float(spec["official_score_delta"]), "expected_weighted_direction": spec["expected_weighted_direction"], "direction_calibration_passed": pass_direction})

    calibration = pd.DataFrame(rows)
    fold_weights = pd.DataFrame(fold_rows)
    min_ess = float(fold_weights["effective_sample_fraction"].min())
    direction_pass = bool(calibration["direction_calibration_passed"].all())
    ess_pass = min_ess >= float(config["calibration_gate"]["minimum_fold_effective_sample_fraction"])
    gate_passed = bool(direction_pass and ess_pass)
    calibration["all_direction_calibration_passed"] = direction_pass
    calibration["minimum_effective_sample_fraction"] = min_ess
    calibration["calibration_gate_passed"] = gate_passed

    digest = hashlib.sha256()
    identity_paths = [config_path, RUNNER_SOURCE, feature_manifest_path, registry_path, train_path, test_path, *probe_paths.values()]
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_test_similarity_" + digest.hexdigest()[:10]
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    domain_path, fold_path, calibration_path = output_dir / "domain_scores.csv", output_dir / "fold_weight_diagnostics.csv", output_dir / "platform_calibration.csv"
    domain_scores.to_csv(domain_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
    fold_weights.to_csv(fold_path, index=False, encoding="utf-8", float_format="%.10f")
    calibration.to_csv(calibration_path, index=False, encoding="utf-8", float_format="%.10f")
    ended = datetime.now(timezone.utc)
    manifest = {"run_id": run_id, "status": "completed", "task": config["task"], "experiment_role": config["experiment_role"], "diagnostic_version": config["diagnostic_version"], "started_at": started.isoformat(), "ended_at": ended.isoformat(), "duration_seconds": round(time.perf_counter() - clock, 3), "holdout_evaluated": False, "submission_eligible": False, "domain_auc_grouped_cv": domain_auc, "domain_feature_count": len(feature_columns), "calendar_features_included": False, "direction_calibration_passed": direction_pass, "effective_sample_gate_passed": ess_pass, "calibration_gate_passed": gate_passed, "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))}, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}, "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths}, "artifacts": {"domain_scores": domain_path.relative_to(PROJECT_ROOT).as_posix(), "domain_scores_sha256": sha256(domain_path), "fold_weight_diagnostics": fold_path.relative_to(PROJECT_ROOT).as_posix(), "fold_weight_diagnostics_sha256": sha256(fold_path), "platform_calibration": calibration_path.relative_to(PROJECT_ROOT).as_posix(), "platform_calibration_sha256": sha256(calibration_path)}}
    write_json(output_dir / "run_manifest.json", manifest)
    if config["output"]["append_registry"]:
        _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_artifact_id"], "hypothesis": config["hypothesis"], "primary_change": "fixed process-only domain weighting calibrated against three official platform probes", "model_name": "logistic_domain_classifier", "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started.isoformat(), "ended_at": ended.isoformat(), "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"domain_auc={domain_auc:.4f}; direction_pass={direction_pass}; ess_pass={ess_pass}; calibration_gate_passed={gate_passed}"})
    print(f"domain_auc_grouped_cv={domain_auc:.6f} features={len(feature_columns)} min_ess_fraction={min_ess:.6f}", flush=True)
    print("\n" + calibration.to_string(index=False), flush=True)
    print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} calibration_gate_passed={gate_passed}", flush=True)


if __name__ == "__main__":
    main()

