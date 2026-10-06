"""Run the preregistered ten-day historical-analog long-g1 experiment."""

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
from src.round2_v3.analog_trajectory_utils import (
    build_future_trajectory_matrix,
    predict_analog_trajectories,
    robust_standardize,
)
from src.round2_v3.baseline_utils import build_prediction_long
from src.round2_v3.contracts import LONG_HORIZONS_MINUTES
from src.round2_v3.metrics import regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/analog_trajectory_g1_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("analog_trajectory_utils.py"),
    RUNNER_SOURCE.with_name("baseline_utils.py"),
    RUNNER_SOURCE.with_name("contracts.py"),
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
    matches = [row for row in manifest["partitions"] if row["partition"] == name]
    if len(matches) != 1:
        raise AssertionError(f"manifest has {len(matches)} {name} partitions")
    return matches[0]


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


def _add_origin_calendar(
    frame: pd.DataFrame,
    origins: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    values = frame.to_numpy(dtype=float)
    names = list(frame.columns)
    additions: list[np.ndarray] = []
    if config["add_origin_hour_cyclic"]:
        angle = 2 * np.pi * (origins.hour + origins.minute / 60.0) / 24.0
        additions.extend([np.sin(angle), np.cos(angle)])
        names.extend(["origin_hour_sin_weighted", "origin_hour_cos_weighted"])
    if config["add_origin_dayofweek_cyclic"]:
        angle = 2 * np.pi * origins.dayofweek / 7.0
        additions.extend([np.sin(angle), np.cos(angle)])
        names.extend(["origin_dayofweek_sin_weighted", "origin_dayofweek_cos_weighted"])
    if additions:
        values = np.column_stack([values, *additions])
    return values, names


def _apply_calendar_weights(
    library: np.ndarray,
    query: np.ndarray,
    names: list[str],
    config: dict[str, Any],
) -> None:
    """Apply declared calendar distance weights after robust scaling."""
    for suffix, weight_key in (
        ("origin_hour_", "origin_hour_weight"),
        ("origin_dayofweek_", "origin_dayofweek_weight"),
    ):
        weight = float(config[weight_key])
        for index, name in enumerate(names):
            if name.startswith(suffix):
                library[:, index] *= weight
                query[:, index] *= weight


def _horizon_bucket(horizon: pd.Series) -> pd.Series:
    return pd.cut(
        horizon,
        bins=[0, 120, 360, 720, 1440],
        labels=["h015_120", "h135_360", "h375_720", "h735_1440"],
        include_lowest=True,
    ).astype(str)


def summarize_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    views = [
        ("overall", ["variant"]),
        ("fold", ["variant", "fold_id"]),
        ("horizon_bucket", ["variant", "horizon_bucket"]),
        ("horizon", ["variant", "horizon_minutes"]),
    ]
    for scope, columns in views:
        for keys, part in oof.groupby(columns, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(columns, keys, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "variant": identity.get("variant", "all"),
                    "fold_id": identity.get("fold_id", "all"),
                    "horizon_bucket": identity.get("horizon_bucket", "all"),
                    "horizon_minutes": identity.get("horizon_minutes", "all"),
                    **regression_metrics(part["actual"], part["prediction"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def evaluate_gate(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    control = config["variants"]["control"]
    candidate = config["variants"]["promotion_candidate"]
    gate = config["selection_gate"]

    def accuracy(scope: str, variant: str, *, fold: str = "all", bucket: str = "all") -> float:
        row = metrics.loc[
            (metrics["scope"] == scope)
            & (metrics["variant"] == variant)
            & (metrics["fold_id"].astype(str) == fold)
            & (metrics["horizon_bucket"].astype(str) == bucket)
        ]
        if len(row) != 1:
            raise AssertionError(f"metric lookup not unique: {scope}/{variant}/{fold}/{bucket}")
        return float(row.iloc[0]["accuracy_1_minus_mape"])

    overall_gain = (accuracy("overall", candidate) - accuracy("overall", control)) * 100
    block_gains = {
        block["fold_id"]: (
            accuracy("fold", candidate, fold=block["fold_id"])
            - accuracy("fold", control, fold=block["fold_id"])
        )
        * 100
        for block in config["validation"]["blocks"]
    }
    bucket_gains = {
        bucket: (
            accuracy("horizon_bucket", candidate, bucket=bucket)
            - accuracy("horizon_bucket", control, bucket=bucket)
        )
        * 100
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440")
    }
    winning_fraction = float(np.mean(np.asarray(list(block_gains.values())) > 0.0))
    minimum_source_days = int(diagnostics["unique_source_days"].min())
    pass_overall = overall_gain >= float(gate["minimum_long_g1_accuracy_gain_pct"])
    pass_blocks = min(block_gains.values()) >= float(gate["minimum_single_block_gain_pct"])
    pass_fraction = winning_fraction >= float(gate["minimum_winning_block_fraction"])
    tolerance = float(gate["maximum_near_far_accuracy_loss_pct"])
    pass_near_far = min(bucket_gains["h015_120"], bucket_gains["h735_1440"]) >= -tolerance
    pass_neighbors = minimum_source_days >= int(gate["minimum_neighbor_source_days"])
    passed = bool(pass_overall and pass_blocks and pass_fraction and pass_near_far and pass_neighbors)
    return pd.DataFrame(
        [
            {
                "variant": candidate,
                "overall_accuracy_gain_pct": overall_gain,
                "single_block_min_gain_pct": min(block_gains.values()),
                "single_block_mean_gain_pct": float(np.mean(list(block_gains.values()))),
                "winning_block_fraction": winning_fraction,
                "near_gain_pct": bucket_gains["h015_120"],
                "mid1_gain_pct": bucket_gains["h135_360"],
                "mid2_gain_pct": bucket_gains["h375_720"],
                "far_gain_pct": bucket_gains["h735_1440"],
                "minimum_neighbor_source_days": minimum_source_days,
                **{f"{key}_gain_pct": value for key, value in block_gains.items()},
                "pass_overall": pass_overall,
                "pass_blocks": pass_blocks,
                "pass_winning_fraction": pass_fraction,
                "pass_near_far": pass_near_far,
                "pass_neighbor_diversity": pass_neighbors,
                "component_gate_passed": passed,
                "standalone_submission_eligible": False,
            }
        ]
    )


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
    horizons = [int(value) for value in config["validation"]["horizons_minutes"]]
    if horizons != list(LONG_HORIZONS_MINUTES):
        raise AssertionError("long horizons differ from official contract")
    registry = pd.read_csv(feature_registry_path)
    selected = list(config["analog_features"]["columns"])
    missing = set(selected) - set(registry["feature_name"])
    if missing:
        raise KeyError(f"analog features absent from registry: {sorted(missing)}")
    chosen = registry.loc[registry["feature_name"].isin(selected)]
    if not chosen["causal"].astype(bool).all() or chosen["target_history"].astype(bool).any():
        raise AssertionError("selected analog features violate causal contract")
    if len(config["validation"]["blocks"]) != int(config["budget"]["fold_count"]):
        raise AssertionError("validation block count differs from budget")
    seen: set[pd.Timestamp] = set()
    blocks: list[dict[str, Any]] = []
    for block in config["validation"]["blocks"]:
        start, end = pd.Timestamp(block["validation_start"]), pd.Timestamp(block["validation_end"])
        origins = pd.date_range(start, end, freq="15min")
        if len(origins) != 960:
            raise AssertionError(f"{block['fold_id']} is not a complete ten-day block")
        if any(value in seen for value in origins):
            raise AssertionError("pseudo-test blocks overlap")
        seen.update(origins)
        blocks.append({**block, "start": start, "end": end, "origins": origins})
    return {
        "feature_manifest_path": feature_manifest_path,
        "feature_registry_path": feature_registry_path,
        "label_manifest_path": label_manifest_path,
        "feature_path": feature_path,
        "label_path": label_path,
        "feature_manifest": feature_manifest,
        "label_manifest": label_manifest,
        "selected_features": selected,
        "horizons": horizons,
        "blocks": blocks,
        "config_path": config_path,
    }


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
        usecols=["datetime", *validated["selected_features"]],
        parse_dates=["datetime"],
        low_memory=False,
    ).sort_values("datetime")
    labels = pd.read_csv(validated["label_path"], parse_dates=["datetime"], low_memory=False)
    feature_index = features.set_index("datetime")
    analog_config = config["analog"]
    preflight_rows: list[dict[str, object]] = []
    for block in validated["blocks"]:
        latest_library_origin = block["start"] - pd.Timedelta(minutes=1440)
        eligible = feature_index.index[feature_index.index <= latest_library_origin]
        stride = int(analog_config["library_origin_stride_minutes"])
        epoch = eligible.view("int64") // (60 * 1_000_000_000)
        library_origins = eligible[epoch % stride == 0]
        trajectory = build_future_trajectory_matrix(
            labels, library_origins, validated["horizons"], config["validation"]["target"]
        )
        complete = np.isfinite(trajectory).all(axis=1)
        library_origins = library_origins[complete]
        if len(library_origins) < int(analog_config["nearest_neighbors"]):
            raise AssertionError(f"{block['fold_id']} has insufficient complete analogs")
        actual = build_future_trajectory_matrix(
            labels, block["origins"], validated["horizons"], config["validation"]["target"]
        )
        library_raw, analog_names = _add_origin_calendar(
            feature_index.loc[library_origins, validated["selected_features"]],
            library_origins,
            config["analog_features"],
        )
        query_raw, query_names = _add_origin_calendar(
            feature_index.loc[block["origins"], validated["selected_features"]],
            block["origins"],
            config["analog_features"],
        )
        if analog_names != query_names:
            raise AssertionError("preflight library/query feature names differ")
        library_scaled, query_scaled, _ = robust_standardize(
            library_raw,
            query_raw,
            scale_floor=float(config["analog_features"]["robust_scale_floor"]),
        )
        _apply_calendar_weights(
            library_scaled,
            query_scaled,
            analog_names,
            config["analog_features"],
        )
        required_days = int(np.ceil(
            int(analog_config["nearest_neighbors"])
            / int(analog_config["maximum_neighbors_per_source_day"])
        ))
        if library_origins.floor("D").nunique() < required_days:
            raise AssertionError(f"{block['fold_id']} lacks required source-day diversity")
        preflight_rows.append(
            {
                "fold_id": block["fold_id"],
                "validation_origins": len(block["origins"]),
                "library_origins": len(library_origins),
                "latest_library_origin": library_origins.max(),
                "missing_validation_label_cells": int((~np.isfinite(actual)).sum()),
            }
        )
    if args.preflight_only:
        print(
            "PASS preflight "
            f"blocks={len(validated['blocks'])} validation_origins={sum(int(row['validation_origins']) for row in preflight_rows)} "
            f"features={len(validated['selected_features'])} horizons={len(validated['horizons'])} "
            f"minimum_library_origins={min(int(row['library_origins']) for row in preflight_rows)} "
            f"missing_validation_label_cells={sum(int(row['missing_validation_label_cells']) for row in preflight_rows)}",
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
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "_analog_trajectory_g1_" + digest.hexdigest()[:10]
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
        "experiment_version": config["experiment_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
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
        predictions: list[pd.DataFrame] = []
        neighbor_parts: list[pd.DataFrame] = []
        fit_rows: list[dict[str, object]] = []
        model_config = dict(config["lightgbm"])
        weight_mode = model_config.pop("sample_weight")
        weight_floor = float(model_config.pop("sample_weight_floor"))
        for number, block in enumerate(validated["blocks"], 1):
            latest_library_origin = block["start"] - pd.Timedelta(minutes=1440)
            eligible = feature_index.index[feature_index.index <= latest_library_origin]
            stride = int(analog_config["library_origin_stride_minutes"])
            epoch = eligible.view("int64") // (60 * 1_000_000_000)
            library_origins = eligible[epoch % stride == 0]
            library_trajectory = build_future_trajectory_matrix(
                labels, library_origins, validated["horizons"], config["validation"]["target"]
            )
            complete = np.isfinite(library_trajectory).all(axis=1)
            library_origins = library_origins[complete]
            library_trajectory = library_trajectory[complete]
            validation_origins = block["origins"]
            actual = build_future_trajectory_matrix(
                labels, validation_origins, validated["horizons"], config["validation"]["target"]
            )

            library_raw, analog_names = _add_origin_calendar(
                feature_index.loc[library_origins, validated["selected_features"]],
                library_origins,
                config["analog_features"],
            )
            query_raw, query_names = _add_origin_calendar(
                feature_index.loc[validation_origins, validated["selected_features"]],
                validation_origins,
                config["analog_features"],
            )
            if analog_names != query_names:
                raise AssertionError("library/query analog feature names differ")
            library_scaled, query_scaled, _ = robust_standardize(
                library_raw,
                query_raw,
                scale_floor=float(config["analog_features"]["robust_scale_floor"]),
            )
            _apply_calendar_weights(
                library_scaled,
                query_scaled,
                analog_names,
                config["analog_features"],
            )
            analog_prediction, neighbor_diagnostics = predict_analog_trajectories(
                library_scaled,
                query_scaled,
                library_trajectory,
                library_origins.floor("D").to_numpy(),
                neighbor_count=int(analog_config["nearest_neighbors"]),
                maximum_per_day=int(analog_config["maximum_neighbors_per_source_day"]),
                distance_epsilon=float(analog_config["distance_epsilon"]),
            )
            neighbor_diagnostics["fold_id"] = block["fold_id"]
            neighbor_diagnostics["datetime"] = validation_origins[
                neighbor_diagnostics["query_index"].to_numpy(dtype=int)
            ]
            neighbor_parts.append(neighbor_diagnostics)

            x_train, _ = build_prediction_long(
                features, library_origins, validated["horizons"], validated["selected_features"]
            )
            y_train = library_trajectory.reshape(-1)
            x_validation, validation_meta = build_prediction_long(
                features, validation_origins, validated["horizons"], validated["selected_features"]
            )
            model = lgb.LGBMRegressor(**model_config)
            sample_weight = None
            if weight_mode == "inverse_absolute_target":
                sample_weight = 1.0 / np.maximum(np.abs(y_train), weight_floor)
            elif weight_mode != "none":
                raise ValueError(f"unsupported sample weight mode: {weight_mode}")
            print(
                f"[{number}/{len(validated['blocks'])}] {block['fold_id']}: "
                f"library_origins={len(library_origins)} train_rows={len(y_train)} "
                f"validation_rows={len(x_validation)} analog_features={len(analog_names)}",
                flush=True,
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
            control_prediction = model.predict(x_validation).reshape(actual.shape)
            analog_weight = float(config["variants"]["analog_weight"])
            control_weight = float(config["variants"]["control_weight"])
            if not np.isclose(analog_weight + control_weight, 1.0):
                raise AssertionError("blend weights must sum to one")
            blend_prediction = analog_weight * analog_prediction + control_weight * control_prediction

            flat_meta = validation_meta.copy()
            flat_meta["fold_id"] = block["fold_id"]
            flat_meta["actual"] = actual.reshape(-1)
            flat_meta["horizon_bucket"] = _horizon_bucket(flat_meta["horizon_minutes"])
            for variant, values in (
                (config["variants"]["control"], control_prediction),
                (config["variants"]["diagnostic_analog"], analog_prediction),
                (config["variants"]["promotion_candidate"], blend_prediction),
            ):
                part = flat_meta.copy()
                part["variant"] = variant
                part["prediction"] = values.reshape(-1)
                predictions.append(part)
            fit_rows.append(
                {
                    "fold_id": block["fold_id"],
                    "validation_start": block["start"],
                    "validation_end": block["end"],
                    "latest_library_origin": library_origins.max(),
                    "library_origins": len(library_origins),
                    "library_source_days": int(library_origins.floor("D").nunique()),
                    "training_rows": len(y_train),
                    "validation_rows": len(x_validation),
                    "process_feature_count": len(validated["selected_features"]),
                    "analog_feature_count": len(analog_names),
                    "tree_feature_count": x_train.shape[1],
                }
            )
            _append_event(event_log, "fold_completed", fold_id=block["fold_id"], library_origins=len(library_origins))

        oof = pd.concat(predictions, ignore_index=True)
        diagnostics = pd.concat(neighbor_parts, ignore_index=True)
        metrics = summarize_predictions(oof)
        gates = evaluate_gate(metrics, diagnostics, config)
        overall = metrics.loc[metrics["scope"] == "overall"].copy()
        control_accuracy = float(
            overall.loc[overall["variant"] == config["variants"]["control"], "accuracy_1_minus_mape"].iloc[0]
        )
        overall["delta_accuracy_pct"] = (overall["accuracy_1_minus_mape"] - control_accuracy) * 100
        comparison = overall[["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]].sort_values("mape")
        frames = {
            "oof_predictions.csv": oof,
            "neighbor_diagnostics.csv": diagnostics,
            "fit_summary.csv": pd.DataFrame.from_records(fit_rows),
            "metrics.csv": metrics,
            "variant_comparison.csv": comparison,
            "gate_results.csv": gates,
        }
        artifacts: dict[str, str] = {}
        for name, frame in frames.items():
            path = output_dir / name
            frame.to_csv(path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
            stem = name.removesuffix(".csv")
            artifacts[stem] = path.relative_to(PROJECT_ROOT).as_posix()
            artifacts[stem + "_sha256"] = sha256(path)
        gate = gates.iloc[0]
        candidate_mape = float(
            comparison.loc[
                comparison["variant"] == config["variants"]["promotion_candidate"], "mape"
            ].iloc[0]
        )
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "model_count": len(validated["blocks"]),
                "primary_variant": config["variants"]["promotion_candidate"],
                "primary_long_g1_mape": candidate_mape,
                "component_gate_passed": bool(gate["component_gate_passed"]),
                "standalone_submission_eligible": False,
                "model_parameters": model_config,
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
                    "model_name": "historical_analog_plus_shared_horizon_tree",
                    "long_g1_mape": candidate_mape,
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"component_gate_passed={bool(gate['component_gate_passed'])}; four ten-day pseudo-tests; no submission",
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", component_gate_passed=bool(gate["component_gate_passed"]))
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} "
            f"component_gate_passed={bool(gate['component_gate_passed'])} standalone_submission_eligible=False",
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
                    "model_name": "historical_analog_plus_shared_horizon_tree",
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
