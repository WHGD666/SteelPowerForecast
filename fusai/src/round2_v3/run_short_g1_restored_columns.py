"""Isolate restored process columns in the v16_final short-g1 recipe.

This scientific runner never creates a submission.  Both variants use the
same September point labels, LightGBM parameters and 80% fold-local fuel
proxy; only the tree feature engine differs.
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
import sklearn
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.anomaly_weight_utils import (
    FUELS_G1,
    build_v16_legacy_features,
    build_v28_features,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
)
from src.round2_v3.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/short_g1_restored_columns_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("anomaly_weight_utils.py")
METRICS_SOURCE = RUNNER_SOURCE.with_name("metrics.py")


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


def _identity(config_path: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *paths]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_short_g1_restored_{digest.hexdigest()[:10]}"


def _validate(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    raw_manifest = _read_json(paths["raw_fingerprints"])
    expected = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
    raw_tables = [PROJECT_ROOT / value for value in config["inputs"]["raw_train_tables"]]
    for path in raw_tables:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256(path) != expected.get(relative):
            raise AssertionError(f"raw table hash mismatch: {relative}")
    label_manifest = _read_json(paths["label_manifest"])
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("interval label hash mismatch")
    split_manifest = _read_json(paths["split_manifest"])
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    experiment = config["experiment"]
    if experiment["control_tree"] != "v16_legacy_columns":
        raise AssertionError("control tree differs from v16 preregistration")
    if experiment["candidate_tree"] != "v28_restored_columns":
        raise AssertionError("candidate tree differs from preregistration")
    if experiment["fuel_recipe"] != "v16_sep_ols_weight_0.8_all_short_horizons":
        raise AssertionError("fuel recipe must remain v16 exact")
    if experiment["future_process_observations_allowed"] or experiment["true_target_history_features_allowed"]:
        raise AssertionError("future process/target features are forbidden")
    horizons = [int(value) for value in config["horizons_minutes"]]
    folds = list(config["validation"]["fold_ids"])
    if horizons != list(range(15, 121, 15)):
        raise AssertionError("short horizons must be 15..120 by 15")
    expected_models = len(folds) * len(horizons) * 2
    if expected_models != int(config["budget"]["expected_model_count"]):
        raise AssertionError("model budget is inconsistent")
    if float(config["training"]["tree_weight"]) != 0.2 or float(config["training"]["fuel_weight"]) != 0.8:
        raise AssertionError("v16 tree/fuel weights must remain 0.2/0.8")
    return {
        "raw_tables": raw_tables,
        "label_manifest": label_manifest,
        "label_path": label_path,
        "split_manifest": split_manifest,
        "folds": folds,
        "horizons": horizons,
    }


def _metric_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label_variant, column in (
        ("official_interval_mean", "actual_interval_mean"),
        ("legacy_point", "actual_legacy_point"),
    ):
        source = oof.copy()
        source["actual"] = source[column]
        for scope, groups in (
            ("overall", ["variant"]),
            ("fold", ["variant", "fold_id"]),
            ("horizon", ["variant", "horizon_minutes"]),
        ):
            for keys, part in source.groupby(groups, sort=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                identity = dict(zip(groups, keys, strict=True))
                rows.append(
                    {
                        "label_variant": label_variant,
                        "scope": scope,
                        "variant": identity.get("variant", "all"),
                        "fold_id": identity.get("fold_id", "all"),
                        "horizon_minutes": identity.get("horizon_minutes", "all"),
                        **regression_metrics(part["actual"], part["prediction"]),
                    }
                )
    return pd.DataFrame.from_records(rows)


def _accuracy(metrics: pd.DataFrame, label: str, scope: str, variant: str, **identity: object) -> float:
    selected = metrics.loc[
        (metrics["label_variant"] == label)
        & (metrics["scope"] == scope)
        & (metrics["variant"] == variant)
    ]
    for column, value in identity.items():
        selected = selected.loc[selected[column].astype(str) == str(value)]
    if len(selected) != 1:
        raise AssertionError(f"metric lookup not unique: {label}/{scope}/{variant}/{identity}")
    return float(selected.iloc[0]["accuracy_1_minus_mape"])


def _gate(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    control, candidate = "v16_legacy_columns", "restored_columns"
    official_gain = (
        _accuracy(metrics, "official_interval_mean", "overall", candidate)
        - _accuracy(metrics, "official_interval_mean", "overall", control)
    ) * 100
    point_gain = (
        _accuracy(metrics, "legacy_point", "overall", candidate)
        - _accuracy(metrics, "legacy_point", "overall", control)
    ) * 100
    fold_gains = {
        fold: (
            _accuracy(metrics, "official_interval_mean", "fold", candidate, fold_id=fold)
            - _accuracy(metrics, "official_interval_mean", "fold", control, fold_id=fold)
        ) * 100
        for fold in config["validation"]["fold_ids"]
    }
    horizon_gains = {
        horizon: (
            _accuracy(metrics, "official_interval_mean", "horizon", candidate, horizon_minutes=horizon)
            - _accuracy(metrics, "official_interval_mean", "horizon", control, horizon_minutes=horizon)
        ) * 100
        for horizon in config["horizons_minutes"]
    }
    gate = config["selection_gate"]
    pass_component = official_gain >= float(gate["minimum_component_short_g1_accuracy_gain_pct"])
    pass_folds = min(fold_gains.values()) >= float(gate["minimum_single_fold_gain_pct"])
    pass_point = point_gain >= float(gate["minimum_auxiliary_legacy_point_gain_pct"])
    pass_horizons = min(horizon_gains.values()) >= -float(gate["maximum_single_horizon_loss_pct"])
    return pd.DataFrame(
        [{
            "variant": candidate,
            "official_short_g1_gain_pct": official_gain,
            "legacy_point_gain_pct": point_gain,
            "single_fold_min_gain_pct": min(fold_gains.values()),
            "single_fold_mean_gain_pct": float(np.mean(list(fold_gains.values()))),
            "single_horizon_min_gain_pct": min(horizon_gains.values()),
            **{f"{fold}_gain_pct": value for fold, value in fold_gains.items()},
            "pass_component": pass_component,
            "pass_folds": pass_folds,
            "pass_legacy_point": pass_point,
            "pass_horizons": pass_horizons,
            "component_gate_passed": bool(pass_component and pass_folds and pass_point and pass_horizons),
            "standalone_submission_eligible": bool(
                official_gain >= float(gate["minimum_standalone_short_g1_accuracy_gain_pct"])
                and pass_folds and pass_point and pass_horizons
            ),
        }]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_paths = {
        key: PROJECT_ROOT / config["inputs"][key]
        for key in ("raw_fingerprints", "label_manifest", "split_manifest", "split_assignments")
    }
    source_paths = [PROJECT_ROOT / config["inputs"][key] for key in ("legacy_v16_source", "legacy_v16_fuel_source", "restored_v28_source")]
    identity_inputs = [*input_paths.values(), *source_paths, RUNNER_SOURCE, UTILS_SOURCE, METRICS_SOURCE]
    validated = _validate(config, input_paths)
    raw = merge_training_tables(validated["raw_tables"])
    grid = to_v28_grid(raw)
    legacy_features, legacy_columns = build_v16_legacy_features(grid)
    restored_features, restored_columns = build_v28_features(grid)
    if set(legacy_columns) == set(restored_columns):
        raise AssertionError("legacy and restored feature sets are unexpectedly identical")
    if args.preflight_only:
        print(
            f"PASS preflight rows={len(grid)} legacy_features={len(legacy_columns)} "
            f"restored_features={len(restored_columns)} models={config['budget']['expected_model_count']}"
        )
        return

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    run_id = _identity(config_path, identity_inputs)
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
        "protocol_version": "round2_v3",
        "experiment_version": config["experiment_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {"lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__},
        },
        "git": {"commit": git_commit, "dirty": git_dirty},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, *identity_inputs]},
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False
    try:
        interval_labels = pd.read_csv(validated["label_path"], parse_dates=["datetime"])
        interval_mean = pd.to_numeric(interval_labels.set_index("datetime")["generator_1"], errors="coerce")
        split = pd.read_csv(input_paths["split_assignments"], parse_dates=["train_end", "validation_origin"])
        split = split.loc[(split["role"] == "model_selection") & split["fold_id"].isin(validated["folds"])].copy()
        if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
            raise AssertionError("fold order/content differs from preregistration")
        timestamps = pd.DatetimeIndex(grid["datetime"])
        positions = pd.Series(np.arange(len(grid)), index=timestamps)
        training_start = pd.Timestamp(config["training"]["g1_tree_start"])
        params = dict(config["lightgbm"])
        predictions: list[pd.DataFrame] = []
        fits: list[dict[str, object]] = []
        variants = {
            "v16_legacy_columns": (legacy_features, legacy_columns),
            "restored_columns": (restored_features, restored_columns),
        }
        groups = list(split.groupby("fold_id", sort=False))
        for fold_number, (fold_id, fold) in enumerate(groups, 1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            cutoff = int(np.flatnonzero(timestamps <= train_end)[-1])
            fuel_training = grid.loc[(grid["datetime"] >= training_start) & (grid["datetime"] <= train_end)]
            fuel_coefficients = fit_linear_fuel_proxy(fuel_training, "generator_1", FUELS_G1)
            origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
            validation_positions = positions.loc[origins].to_numpy(dtype=int)
            fuel_prediction = predict_linear_fuel(grid.iloc[validation_positions], FUELS_G1, fuel_coefficients)
            print(f"[{fold_number}/{len(groups)}] {fold_id}: validation_origins={len(origins)}", flush=True)
            for horizon in validated["horizons"]:
                step = horizon // 15
                train_positions = np.flatnonzero((timestamps >= training_start) & (np.arange(len(grid)) + step <= cutoff))
                target_positions = train_positions + step
                y_train = grid["generator_1"].to_numpy(dtype=float)[target_positions]
                valid_train = np.isfinite(y_train)
                train_positions, y_train = train_positions[valid_train], y_train[valid_train]
                if len(y_train) < int(config["training"]["minimum_rows_per_model"]):
                    raise AssertionError(f"{fold_id} h={horizon} has only {len(y_train)} training rows")
                actual_point = grid["generator_1"].to_numpy(dtype=float)[validation_positions + step]
                interval_starts = origins + pd.to_timedelta(horizon - 15, unit="min")
                actual_interval = interval_mean.reindex(interval_starts).to_numpy(dtype=float)
                for variant, (features, columns) in variants.items():
                    model = lgb.LGBMRegressor(**params)
                    model.fit(features.iloc[train_positions][columns], y_train)
                    tree_prediction = np.clip(model.predict(features.iloc[validation_positions][columns]), 0, None)
                    prediction = 0.2 * tree_prediction + 0.8 * fuel_prediction
                    predictions.append(pd.DataFrame({
                        "variant": variant,
                        "fold_id": fold_id,
                        "datetime": origins,
                        "interval_start": interval_starts,
                        "horizon_minutes": horizon,
                        "actual_interval_mean": actual_interval,
                        "actual_legacy_point": actual_point,
                        "prediction": prediction,
                    }))
                    fits.append({"fold_id": fold_id, "horizon_minutes": horizon, "variant": variant, "training_rows": len(y_train), "feature_count": len(columns), "training_origin_start": timestamps[train_positions.min()], "training_origin_end": timestamps[train_positions.max()]})
            _append_event(event_log, "fold_completed", fold_id=fold_id, validation_origins=len(origins))
        oof = pd.concat(predictions, ignore_index=True)
        expected_rows = len(split) * len(validated["horizons"]) * 2
        if len(oof) != expected_rows:
            raise AssertionError(f"OOF row count {len(oof)} != {expected_rows}")
        metrics = _metric_table(oof)
        gates = _gate(metrics, config)
        overall = metrics.loc[(metrics["label_variant"] == "official_interval_mean") & (metrics["scope"] == "overall")].copy()
        control_accuracy = float(overall.loc[overall["variant"] == "v16_legacy_columns", "accuracy_1_minus_mape"].iloc[0])
        overall["delta_accuracy_pct"] = (overall["accuracy_1_minus_mape"] - control_accuracy) * 100
        comparison = overall[["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]].sort_values("mape")
        artifacts = {
            "oof_predictions.csv": oof,
            "fit_summary.csv": pd.DataFrame.from_records(fits),
            "metrics.csv": metrics,
            "variant_comparison.csv": comparison,
            "gate_results.csv": gates,
        }
        artifact_meta = {}
        for name, frame in artifacts.items():
            path = output_dir / name
            frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
            artifact_meta[name.removesuffix(".csv")] = path.relative_to(PROJECT_ROOT).as_posix()
            artifact_meta[name.removesuffix(".csv") + "_sha256"] = sha256(path)
        ended_at = datetime.now(timezone.utc)
        gate_row = gates.iloc[0]
        candidate_mape = float(comparison.loc[comparison["variant"] == "restored_columns", "mape"].iloc[0])
        manifest.update({
            "status": "completed",
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round(time.perf_counter() - started_clock, 3),
            "model_count": int(config["budget"]["expected_model_count"]),
            "feature_counts": {"legacy": len(legacy_columns), "restored": len(restored_columns)},
            "model_parameters": params,
            "primary_variant": "restored_columns",
            "primary_short_g1_mape": candidate_mape,
            "component_gate_passed": bool(gate_row["component_gate_passed"]),
            "standalone_submission_eligible": bool(gate_row["standalone_submission_eligible"]),
            "artifacts": artifact_meta,
        })
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry({
                "run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"],
                "task": config["task"], "protocol_version": "round2_v3", "label_version": validated["label_manifest"]["artifact_version"],
                "split_sha256": validated["split_manifest"]["assignment_sha256"], "control_run_id": config["control_artifact_id"],
                "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "v16_short_g1_restored_columns",
                "short_g1_mape": candidate_mape, "holdout_evaluated": False, "submission_eligible": False,
                "git_commit": git_commit, "git_dirty": git_dirty, "started_at": started_at.isoformat(), "ended_at": ended_at.isoformat(),
                "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "notes": f"component_gate_passed={bool(gate_row['component_gate_passed'])}; scientific OOF only",
            })
            registry_written = True
        _append_event(event_log, "run_completed", duration_seconds=manifest["duration_seconds"], component_gate_passed=bool(gate_row["component_gate_passed"]))
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} component_gate_passed={bool(gate_row['component_gate_passed'])} standalone_submission_eligible={bool(gate_row['standalone_submission_eligible'])}", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "ended_at": datetime.now(timezone.utc).isoformat(), "duration_seconds": round(time.perf_counter() - started_clock, 3), "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", error_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry({"run_id": run_id, "status": "failed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_artifact_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "v16_short_g1_restored_columns", "holdout_evaluated": False, "submission_eligible": False, "git_commit": git_commit, "git_dirty": git_dirty, "started_at": started_at.isoformat(), "ended_at": manifest["ended_at"], "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"failure={type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
