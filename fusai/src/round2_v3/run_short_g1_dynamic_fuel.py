"""Evaluate causal dynamic fuel against the exact v16-lineage short-g1 OOF."""

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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io_utils import sha256, write_json
from src.round2_v3.baseline_utils import build_supervised_long
from src.round2_v3.dynamic_fuel_utils import (
    build_dynamic_fuel_features,
    latest_mature_origin,
    recent_origin_window,
)
from src.round2_v3.event_ablation_utils import attach_event_context
from src.round2_v3.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/short_g1_dynamic_fuel_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("baseline_utils.py"),
    RUNNER_SOURCE.with_name("dynamic_fuel_utils.py"),
    RUNNER_SOURCE.with_name("event_ablation_utils.py"),
    RUNNER_SOURCE.with_name("metrics.py"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in manifest["partitions"] if row["partition"] == name]
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


def _run_id(config_path: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *paths]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_short_dynamic_fuel_{digest.hexdigest()[:10]}"


def _validate(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    feature_manifest = _read_json(paths["feature_manifest"])
    label_manifest = _read_json(paths["label_manifest"])
    split_manifest = _read_json(paths["split_manifest"])
    control_manifest = _read_json(paths["control_manifest"])
    train_features = PROJECT_ROOT / _partition(feature_manifest, "train")["output"]
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(train_features) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("train origin feature hash mismatch")
    if sha256(paths["feature_registry"]) != feature_manifest["registry_sha256"]:
        raise AssertionError("feature registry hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    if feature_manifest["maximum_source_time_offset_minutes"] > 0 or feature_manifest["target_history_included"]:
        raise AssertionError("origin features violate the causal contract")
    if control_manifest["run_id"] != config["control_run_id"] or control_manifest["status"] != "completed":
        raise AssertionError("short control run identity/status mismatch")
    if sha256(paths["control_oof"]) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("short control OOF hash mismatch")
    scope = config["scope"]
    if scope["period"] != "short" or scope["target"] != "generator_1":
        raise AssertionError("experiment must remain short generator_1 only")
    if scope["generator_all_changed"] or scope["long_outputs_changed"]:
        raise AssertionError("generator_all and long outputs must remain frozen")
    if scope["future_process_observations_allowed"] or scope["true_target_history_allowed"]:
        raise AssertionError("forbidden information was enabled")
    horizons = [int(value) for value in scope["horizons_minutes"]]
    folds = list(scope["fold_ids"])
    if horizons != list(range(15, 121, 15)):
        raise AssertionError("short horizons must be 15..120 by 15")
    if config["variants"]["broad_blend_search"]:
        raise AssertionError("broad blend search is forbidden")
    weights = config["variants"][config["variants"]["promotion_candidate"]]
    if not np.isclose(float(weights["dynamic_fuel_weight"]), 0.70) or not np.isclose(float(weights["v16_control_weight"]), 0.30):
        raise AssertionError("promotion blend must remain fixed at 70/30")
    expected_models = len(folds) * len(horizons)
    if expected_models != int(config["budget"]["expected_model_count"]):
        raise AssertionError("model budget mismatch")
    return {
        "feature_manifest": feature_manifest,
        "label_manifest": label_manifest,
        "split_manifest": split_manifest,
        "control_manifest": control_manifest,
        "train_features": train_features,
        "label_path": label_path,
        "folds": folds,
        "horizons": horizons,
    }


def _load_control(path: Path, variant: str, folds: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime", "interval_start"])
    frame = frame.loc[(frame["variant"] == variant) & frame["fold_id"].isin(folds)].copy()
    frame = frame.rename(columns={"actual_interval_mean": "actual"})
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    if frame.empty or frame.duplicated(key).any():
        raise AssertionError("short control OOF is empty or has duplicate keys")
    return frame[key + ["actual", "prediction"]]


def summarize_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope, groups in (
        ("overall", ["variant"]),
        ("fold", ["variant", "fold_id"]),
        ("horizon", ["variant", "horizon_minutes"]),
        ("ordinary", ["variant"]),
        ("episode", ["variant", "target_episode_id"]),
    ):
        source = oof.loc[oof["origin_regime"] == "ordinary"].copy() if scope == "ordinary" else oof
        if scope == "episode":
            episode_id = source["target_episode_id"]
            valid_episode = (
                episode_id.notna()
                & episode_id.astype(str).str.strip().ne("")
                & episode_id.astype(str).str.lower().ne("none")
            )
            source = source.loc[valid_episode].copy()
        for keys, part in source.groupby(groups, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(groups, keys, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "variant": identity.get("variant", "all"),
                    "fold_id": identity.get("fold_id", "all"),
                    "horizon_minutes": identity.get("horizon_minutes", "all"),
                    "target_episode_id": identity.get("target_episode_id", "all"),
                    **regression_metrics(part["actual"], part["prediction"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def evaluate_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    control = "v16_control"
    candidate = config["variants"]["promotion_candidate"]

    def accuracy(scope: str, variant: str, column: str | None = None, value: object | None = None) -> float:
        selected = metrics.loc[(metrics["scope"] == scope) & (metrics["variant"] == variant)]
        if column is not None:
            selected = selected.loc[selected[column].astype(str) == str(value)]
        if len(selected) != 1:
            raise AssertionError(f"metric lookup not unique: {scope}/{variant}/{column}/{value}")
        return float(selected.iloc[0]["accuracy_1_minus_mape"])

    overall_gain = (accuracy("overall", candidate) - accuracy("overall", control)) * 100
    fold_gains = [
        (accuracy("fold", candidate, "fold_id", fold) - accuracy("fold", control, "fold_id", fold)) * 100
        for fold in config["scope"]["fold_ids"]
    ]
    horizon_gains = [
        (accuracy("horizon", candidate, "horizon_minutes", horizon) - accuracy("horizon", control, "horizon_minutes", horizon)) * 100
        for horizon in config["scope"]["horizons_minutes"]
    ]
    ordinary_gain = (accuracy("ordinary", candidate) - accuracy("ordinary", control)) * 100
    episodes = sorted(
        set(metrics.loc[(metrics["scope"] == "episode") & (metrics["variant"] == control), "target_episode_id"])
        & set(metrics.loc[(metrics["scope"] == "episode") & (metrics["variant"] == candidate), "target_episode_id"])
    )
    episode_gains = [
        (accuracy("episode", candidate, "target_episode_id", episode) - accuracy("episode", control, "target_episode_id", episode)) * 100
        for episode in episodes
    ]
    improvement_fraction = float(np.mean(np.asarray(episode_gains) > 0)) if episode_gains else 0.0
    gate = config["selection_gate"]
    checks = {
        "pass_overall": overall_gain >= float(gate["minimum_short_g1_accuracy_gain_pct"]),
        "pass_folds": min(fold_gains) >= float(gate["minimum_single_fold_gain_pct"]),
        "pass_horizons": min(horizon_gains) >= float(gate["minimum_single_horizon_gain_pct"]),
        "pass_ordinary": ordinary_gain >= -float(gate["maximum_ordinary_origin_accuracy_loss_pct"]),
        "pass_episode_count": len(episodes) >= int(gate["minimum_evaluated_event_count"]),
        "pass_episodes": improvement_fraction >= float(gate["minimum_event_improvement_fraction"]),
    }
    return pd.DataFrame(
        [
            {
                "variant": candidate,
                "overall_accuracy_gain_pct": overall_gain,
                "single_fold_min_gain_pct": min(fold_gains),
                "single_fold_mean_gain_pct": float(np.mean(fold_gains)),
                "single_horizon_min_gain_pct": min(horizon_gains),
                "ordinary_accuracy_gain_pct": ordinary_gain,
                "evaluated_episode_count": len(episodes),
                "episode_improvement_fraction": improvement_fraction,
                **checks,
                "component_gate_passed": bool(all(checks.values())),
                "standalone_submission_eligible": bool(all(checks.values())),
            }
        ]
    )


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
    validated = _validate(config, paths)
    origin = pd.read_csv(validated["train_features"], low_memory=False)
    feature_config = config["dynamic_fuel_features"]
    dynamic, registry = build_dynamic_fuel_features(
        origin,
        feature_config["fuel_columns"],
        feature_config["lag_steps"],
        feature_config["rolling_mean_steps"],
        frequency_minutes=int(feature_config["origin_frequency_minutes"]),
        include_current=bool(feature_config["include_current"]),
    )
    dynamic_columns = registry["feature_name"].tolist()
    model_columns = [*dynamic_columns, *feature_config["target_calendar_columns"]]
    dynamic_times = pd.DatetimeIndex(dynamic["datetime"])
    split = pd.read_csv(paths["split_assignments"], parse_dates=["train_end", "validation_origin"])
    split = split.loc[(split["role"] == "model_selection") & split["fold_id"].isin(validated["folds"])].copy()
    if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
        raise AssertionError("fold order/content differs from preregistration")
    for _, fold in split.groupby("fold_id", sort=False):
        train_end = pd.Timestamp(fold["train_end"].iloc[0])
        for horizon in validated["horizons"]:
            latest = latest_mature_origin(train_end, horizon, frequency_minutes=15)
            recent_origin_window(dynamic_times, latest, window_days=int(config["training"]["recent_window_days"]), frequency_minutes=15)
    control = _load_control(paths["control_oof"], config["scope"]["control_variant"], validated["folds"])
    expected_rows = sum(len(fold) for _, fold in split.groupby("fold_id", sort=False)) * len(validated["horizons"])
    if len(control) != expected_rows:
        raise AssertionError(f"control coverage {len(control)} != expected {expected_rows}")
    if args.preflight_only:
        print(
            f"PASS preflight folds={len(validated['folds'])} horizons={len(validated['horizons'])} "
            f"models={config['budget']['expected_model_count']} control_rows={len(control)} "
            f"features={len(model_columns)} max_source_offset={int(registry['source_offset_max_minutes'].max())}",
            flush=True,
        )
        return

    identity_paths = [*paths.values(), validated["train_features"], validated["label_path"], RUNNER_SOURCE, *HELPER_SOURCES]
    run_id = _run_id(config_path, identity_paths)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": config["task"],
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "experiment_version": config["experiment_version"],
        "control_run_id": config["control_run_id"],
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"branch": _git_value("branch", "--show-current"), "commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, *identity_paths]},
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    registry_written = False
    try:
        labels = pd.read_csv(validated["label_path"], low_memory=False)
        labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
        episodes = pd.read_csv(paths["event_episodes"])
        weights = config["variants"][config["variants"]["promotion_candidate"]]
        dynamic_weight = float(weights["dynamic_fuel_weight"])
        control_weight = float(weights["v16_control_weight"])
        predictions: list[pd.DataFrame] = []
        fits: list[dict[str, object]] = []
        for fold_number, (fold_id, fold) in enumerate(split.groupby("fold_id", sort=False), 1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
            print(f"[{fold_number}/{len(validated['folds'])}] {fold_id}: validation_origins={len(origins)}", flush=True)
            for horizon in validated["horizons"]:
                latest = latest_mature_origin(train_end, horizon, frequency_minutes=15)
                training_origins = recent_origin_window(dynamic_times, latest, window_days=int(config["training"]["recent_window_days"]), frequency_minutes=15)
                x_train, y_train, _ = build_supervised_long(dynamic, labels, training_origins, [horizon], "generator_1", dynamic_columns)
                x_val, y_val, meta = build_supervised_long(dynamic, labels, origins, [horizon], "generator_1", dynamic_columns)
                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                        ("regressor", LinearRegression()),
                    ]
                )
                model.fit(x_train[model_columns], y_train)
                dynamic_prediction = model.predict(x_val[model_columns])
                control_part = control.loc[(control["fold_id"] == fold_id) & (control["horizon_minutes"] == horizon)].sort_values(["datetime", "interval_start"]).reset_index(drop=True)
                meta = meta.sort_values(["datetime", "interval_start"]).reset_index(drop=True)
                if not control_part[["datetime", "interval_start"]].equals(meta[["datetime", "interval_start"]]):
                    raise AssertionError("control and dynamic validation keys differ")
                np.testing.assert_allclose(control_part["actual"], y_val, rtol=0, atol=1e-8)
                control_prediction = control_part["prediction"].to_numpy(float)
                blend_prediction = dynamic_weight * dynamic_prediction + control_weight * control_prediction
                for variant, prediction in (
                    ("v16_control", control_prediction),
                    ("dynamic_fuel_ols", dynamic_prediction),
                    (config["variants"]["promotion_candidate"], blend_prediction),
                ):
                    part = meta.copy()
                    part["actual"] = y_val
                    part["prediction"] = prediction
                    part["variant"] = variant
                    part["fold_id"] = fold_id
                    predictions.append(attach_event_context(part, episodes, context_hours=24))
                fits.append({"fold_id": fold_id, "horizon_minutes": horizon, "training_rows": len(y_train), "training_origin_start": training_origins.min(), "training_origin_end": training_origins.max()})
        oof = pd.concat(predictions, ignore_index=True)
        metrics = summarize_predictions(oof)
        gates = evaluate_gate(metrics, config)
        comparison = metrics.loc[metrics["scope"] == "overall", ["variant", "mape", "accuracy_1_minus_mape"]].copy()
        control_accuracy = float(comparison.loc[comparison["variant"] == "v16_control", "accuracy_1_minus_mape"].iloc[0])
        comparison["delta_accuracy_pct"] = (comparison["accuracy_1_minus_mape"] - control_accuracy) * 100
        print("\n" + comparison.sort_values("mape").to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        oof_path, metrics_path, gates_path, fits_path = [output_dir / name for name in ("oof_predictions.csv", "metrics.csv", "gate_results.csv", "fit_summary.csv")]
        oof.to_csv(oof_path, index=False, encoding="utf-8", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        pd.DataFrame(fits).to_csv(fits_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")
        gate_passed = bool(gates.iloc[0]["component_gate_passed"])
        manifest.update(
            {
                "status": "completed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "model_count": len(fits),
                "primary_variant": config["variants"]["promotion_candidate"],
                "primary_short_g1_mape": float(comparison.loc[comparison["variant"] == config["variants"]["promotion_candidate"], "mape"].iloc[0]),
                "component_gate_passed": gate_passed,
                "standalone_submission_eligible": gate_passed,
                "submission_eligible": False,
                "artifacts": {
                    "oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(), "oof_predictions_sha256": sha256(oof_path),
                    "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(), "metrics_sha256": sha256(metrics_path),
                    "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(), "gate_results_sha256": sha256(gates_path),
                    "fit_summary": fits_path.relative_to(PROJECT_ROOT).as_posix(), "fit_summary_sha256": sha256(fits_path),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry(
                {
                    "run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "label_version": validated["label_manifest"]["artifact_version"], "split_sha256": validated["split_manifest"]["assignment_sha256"], "control_run_id": config["control_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "horizon_specific_dynamic_fuel_ols_blend", "short_g1_mape": manifest["primary_short_g1_mape"], "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"component_gate_passed={gate_passed}; standalone_submission_eligible={gate_passed}",
                }
            )
            registry_written = True
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={gate_passed}", flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_run_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "horizon_specific_dynamic_fuel_ols_blend", "holdout_evaluated": False, "submission_eligible": False, "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
