"""Run the preregistered direct multi-step GRU long-g1 experiment."""

from __future__ import annotations

import argparse
import csv
import gc
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
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.io_utils import sha256, write_json
from src.round2_v3.analog_trajectory_utils import build_future_trajectory_matrix
from src.round2_v3.contracts import LONG_HORIZONS_MINUTES
from src.round2_v3.metrics import regression_metrics
from src.round2_v3.sequence_gru_utils import (
    DirectMultiStepGRU,
    add_step_calendar,
    build_sequence_windows,
    fit_log_target_normalization,
    fit_transform_time_features,
    inverse_log_target_normalization,
    set_reproducible_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/sequence_gru_g1_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("sequence_gru_utils.py"),
    RUNNER_SOURCE.with_name("analog_trajectory_utils.py"),
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
    rows = [row for row in manifest["partitions"] if row["partition"] == name]
    if len(rows) != 1:
        raise AssertionError(f"manifest has {len(rows)} {name} partitions")
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


def _horizon_bucket(horizon: pd.Series) -> pd.Series:
    return pd.cut(
        horizon,
        bins=[0, 120, 360, 720, 1440],
        labels=["h015_120", "h135_360", "h375_720", "h735_1440"],
        include_lowest=True,
    ).astype(str)


def _prediction_meta(
    origins: pd.DatetimeIndex,
    horizons: list[int],
    fold_id: str,
    actual: np.ndarray,
) -> pd.DataFrame:
    repeated_origins = pd.DatetimeIndex(np.repeat(origins.to_numpy(), len(horizons)))
    repeated_horizons = np.tile(np.asarray(horizons, dtype=int), len(origins))
    result = pd.DataFrame(
        {
            "fold_id": fold_id,
            "datetime": repeated_origins,
            "interval_start": repeated_origins
            + pd.to_timedelta(repeated_horizons - 15, unit="min"),
            "horizon_minutes": repeated_horizons,
            "actual": actual.reshape(-1),
        }
    )
    result["horizon_bucket"] = _horizon_bucket(result["horizon_minutes"])
    return result


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


def evaluate_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
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
    pass_overall = overall_gain >= float(gate["minimum_long_g1_accuracy_gain_pct"])
    pass_blocks = min(block_gains.values()) >= float(gate["minimum_single_block_gain_pct"])
    pass_fraction = winning_fraction >= float(gate["minimum_winning_block_fraction"])
    tolerance = float(gate["maximum_near_far_accuracy_loss_pct"])
    pass_near_far = min(bucket_gains["h015_120"], bucket_gains["h735_1440"]) >= -tolerance
    passed = bool(pass_overall and pass_blocks and pass_fraction and pass_near_far)
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
                **{f"{key}_gain_pct": value for key, value in block_gains.items()},
                "pass_overall": pass_overall,
                "pass_blocks": pass_blocks,
                "pass_winning_fraction": pass_fraction,
                "pass_near_far": pass_near_far,
                "component_gate_passed": passed,
                "standalone_submission_eligible": False,
            }
        ]
    )


def _validate(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    paths = {key: PROJECT_ROOT / value for key, value in config["inputs"].items()}
    feature_manifest = _read_json(paths["event_feature_manifest"])
    label_manifest = _read_json(paths["label_manifest"])
    control_manifest = _read_json(paths["control_manifest"])
    feature_path = PROJECT_ROOT / _partition(feature_manifest, "train")["output"]
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(feature_path) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("event feature artifact hash mismatch")
    if sha256(paths["event_feature_registry"]) != feature_manifest["registry_sha256"]:
        raise AssertionError("event feature registry hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if control_manifest["run_id"] != config["control_scientific_run_id"]:
        raise AssertionError("control run id mismatch")
    if control_manifest["status"] != "completed":
        raise AssertionError("control scientific run is incomplete")
    if sha256(paths["control_oof"]) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("control OOF hash mismatch")
    if feature_manifest["target_history_included"] or feature_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("sequence features violate causal contract")
    validation = config["validation"]
    if validation["true_target_history_allowed"] or validation["future_process_observations_allowed"]:
        raise AssertionError("target history and future process observations are forbidden")
    if not validation["full_horizon_must_end_before_block"]:
        raise AssertionError("training trajectory maturity rule must remain enabled")
    horizons = [int(value) for value in validation["horizons_minutes"]]
    if horizons != list(LONG_HORIZONS_MINUTES):
        raise AssertionError("long horizons differ from official contract")
    if int(config["model"]["output_size"]) != len(horizons):
        raise AssertionError("GRU output size differs from horizon count")
    if int(config["model"]["input_size"]) != len(config["sequence_features"]["columns"]) + 4:
        raise AssertionError("GRU input size differs from declared features plus calendar")
    if config["training"]["broad_hyperparameter_search"]:
        raise AssertionError("first GRU experiment forbids broad search")
    registry = pd.read_csv(paths["event_feature_registry"])
    selected = list(config["sequence_features"]["columns"])
    missing = set(selected) - set(registry["feature_name"])
    if missing:
        raise KeyError(f"sequence features absent from registry: {sorted(missing)}")
    chosen = registry.loc[registry["feature_name"].isin(selected)]
    if not chosen["causal"].astype(bool).all() or chosen["target_history"].astype(bool).any():
        raise AssertionError("selected sequence features violate causal contract")
    blocks: list[dict[str, Any]] = []
    seen: set[pd.Timestamp] = set()
    for block in validation["blocks"]:
        start, end = pd.Timestamp(block["validation_start"]), pd.Timestamp(block["validation_end"])
        origins = pd.date_range(start, end, freq="15min")
        if len(origins) != 960 or any(value in seen for value in origins):
            raise AssertionError("pseudo-test blocks must be disjoint complete ten-day grids")
        seen.update(origins)
        blocks.append({**block, "start": start, "end": end, "origins": origins})
    if len(blocks) != int(config["budget"]["fold_count"]):
        raise AssertionError("block count differs from budget")
    return {
        "paths": paths,
        "feature_manifest": feature_manifest,
        "label_manifest": label_manifest,
        "control_manifest": control_manifest,
        "feature_path": feature_path,
        "label_path": label_path,
        "selected_features": selected,
        "horizons": horizons,
        "blocks": blocks,
        "config_path": config_path,
    }


def _predict_batches(
    model: nn.Module,
    windows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
            parts.append(model(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = _validate(config, config_path)
    if config["training"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the preregistered GRU experiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = pd.read_csv(
        validated["feature_path"],
        usecols=["datetime", *validated["selected_features"]],
        parse_dates=["datetime"],
        low_memory=False,
    ).sort_values("datetime").reset_index(drop=True)
    feature_times = pd.DatetimeIndex(features["datetime"])
    if not feature_times.equals(pd.date_range(feature_times.min(), feature_times.max(), freq="15min")):
        raise AssertionError("sequence feature grid is not contiguous at 15 minutes")
    position_by_time = pd.Series(np.arange(len(features), dtype=int), index=feature_times)
    raw_values, input_names = add_step_calendar(
        features[validated["selected_features"]], feature_times, config["sequence_features"]
    )
    labels = pd.read_csv(validated["label_path"], parse_dates=["datetime"], low_memory=False)
    control_columns = [
        "variant", "fold_id", "datetime", "interval_start", "horizon_minutes", "actual", "prediction"
    ]
    control = pd.read_csv(validated["paths"]["control_oof"], usecols=control_columns, parse_dates=["datetime", "interval_start"], low_memory=False)
    control = control.loc[control["variant"] == config["variants"]["control"]].copy()
    key = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
    if control.duplicated(key).any():
        raise AssertionError("control OOF contains duplicate keys")

    preflight: list[dict[str, object]] = []
    window_steps = int(config["validation"]["input_window_steps"])
    stride = int(config["validation"]["training_origin_stride_minutes"])
    for block in validated["blocks"]:
        latest_origin = block["start"] - pd.Timedelta(minutes=1440)
        eligible = feature_times[feature_times <= latest_origin]
        epoch = eligible.view("int64") // (60 * 1_000_000_000)
        training_origins = eligible[epoch % stride == 0]
        positions = position_by_time.reindex(training_origins).to_numpy(dtype=int)
        training_origins = training_origins[positions >= window_steps - 1]
        trajectories = build_future_trajectory_matrix(
            labels, training_origins, validated["horizons"], config["validation"]["target"]
        )
        complete = np.isfinite(trajectories).all(axis=1) & (trajectories > 0).all(axis=1)
        training_origins = training_origins[complete]
        query_positions = position_by_time.reindex(block["origins"]).to_numpy(dtype=int)
        if np.any(query_positions < window_steps - 1):
            raise AssertionError(f"{block['fold_id']} lacks sequence history")
        actual = build_future_trajectory_matrix(
            labels, block["origins"], validated["horizons"], config["validation"]["target"]
        )
        control_fold = control.loc[control["fold_id"] == block["fold_id"]]
        if len(control_fold) != len(block["origins"]) * len(validated["horizons"]):
            raise AssertionError(f"{block['fold_id']} control row count mismatch")
        preflight.append(
            {
                "fold_id": block["fold_id"],
                "training_origins": len(training_origins),
                "validation_origins": len(block["origins"]),
                "missing_validation_cells": int((~np.isfinite(actual)).sum()),
                "latest_training_origin": training_origins.max(),
            }
        )
    if args.preflight_only:
        sample_model = DirectMultiStepGRU(
            input_size=int(config["model"]["input_size"]),
            hidden_size=int(config["model"]["hidden_size"]),
            num_layers=int(config["model"]["num_layers"]),
            dropout=float(config["model"]["dropout"]),
            output_size=int(config["model"]["output_size"]),
            layer_norm=bool(config["model"]["layer_norm"]),
        )
        parameters = sum(parameter.numel() for parameter in sample_model.parameters())
        print(
            "PASS preflight "
            f"device={device.type} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'} "
            f"blocks={len(preflight)} min_train_origins={min(int(row['training_origins']) for row in preflight)} "
            f"input_steps={window_steps} input_features={len(input_names)} outputs={len(validated['horizons'])} "
            f"parameters={parameters} missing_validation_cells={sum(int(row['missing_validation_cells']) for row in preflight)}",
            flush=True,
        )
        return

    identity_paths = [
        config_path,
        validated["paths"]["event_feature_manifest"],
        validated["paths"]["event_feature_registry"],
        validated["paths"]["label_manifest"],
        validated["paths"]["control_manifest"],
        validated["paths"]["control_oof"],
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
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "_sequence_gru_g1_" + digest.hexdigest()[:10]
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=False, exist_ok=False)
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
        "control_scientific_run_id": config["control_scientific_run_id"],
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "dependencies": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "random_seeds": [int(config["training"]["seed"])],
        "model_parameters": config["model"],
        "training_parameters": config["training"],
        "allowed_features": input_names,
        "forbidden_features": ["generator_1_history", "generator_all_history", "future_process_observations"],
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths},
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
        "models": [],
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id, device=str(device))
    registry_written = False
    try:
        predictions: list[pd.DataFrame] = []
        fit_rows: list[dict[str, object]] = []
        epoch_rows: list[dict[str, object]] = []
        model_records: list[dict[str, object]] = []
        training_config = config["training"]
        model_config = config["model"]
        for fold_number, block in enumerate(validated["blocks"], 1):
            set_reproducible_seed(
                int(training_config["seed"]),
                deterministic_cudnn=bool(training_config["deterministic_cudnn"]),
            )
            latest_origin = block["start"] - pd.Timedelta(minutes=1440)
            eligible = feature_times[feature_times <= latest_origin]
            epoch_minutes = eligible.view("int64") // (60 * 1_000_000_000)
            training_origins = eligible[epoch_minutes % stride == 0]
            training_positions = position_by_time.reindex(training_origins).to_numpy(dtype=int)
            has_history = training_positions >= window_steps - 1
            training_origins = training_origins[has_history]
            training_positions = training_positions[has_history]
            trajectories = build_future_trajectory_matrix(
                labels, training_origins, validated["horizons"], config["validation"]["target"]
            )
            complete = np.isfinite(trajectories).all(axis=1) & (trajectories > 0).all(axis=1)
            training_origins = training_origins[complete]
            training_positions = training_positions[complete]
            trajectories = trajectories[complete]
            validation_positions = position_by_time.reindex(block["origins"]).to_numpy(dtype=int)
            actual = build_future_trajectory_matrix(
                labels, block["origins"], validated["horizons"], config["validation"]["target"]
            )

            training_row_mask = feature_times <= latest_origin
            transformed, feature_fit = fit_transform_time_features(
                raw_values,
                training_row_mask,
                scale_floor=float(config["sequence_features"]["robust_scale_floor"]),
            )
            x_train = build_sequence_windows(transformed, training_positions, window_steps=window_steps)
            x_validation = build_sequence_windows(transformed, validation_positions, window_steps=window_steps)
            y_train, target_fit = fit_log_target_normalization(trajectories)

            model = DirectMultiStepGRU(
                input_size=int(model_config["input_size"]),
                hidden_size=int(model_config["hidden_size"]),
                num_layers=int(model_config["num_layers"]),
                dropout=float(model_config["dropout"]),
                output_size=int(model_config["output_size"]),
                layer_norm=bool(model_config["layer_norm"]),
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training_config["learning_rate"]),
                weight_decay=float(training_config["weight_decay"]),
            )
            criterion = nn.SmoothL1Loss(beta=float(training_config["smooth_l1_beta"]))
            generator = torch.Generator().manual_seed(int(training_config["seed"]))
            dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            loader = DataLoader(
                dataset,
                batch_size=int(training_config["batch_size"]),
                shuffle=True,
                num_workers=int(training_config["dataloader_workers"]),
                pin_memory=device.type == "cuda",
                generator=generator,
            )
            print(
                f"[{fold_number}/{len(validated['blocks'])}] {block['fold_id']}: "
                f"train_origins={len(training_origins)} validation_origins={len(block['origins'])} "
                f"window={window_steps}x{len(input_names)} device={device.type}",
                flush=True,
            )
            for epoch_index in range(1, int(training_config["epochs"]) + 1):
                model.train()
                total_loss = 0.0
                total_rows = 0
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(training_config["gradient_clip_norm"])
                    )
                    optimizer.step()
                    total_loss += float(loss.detach().cpu()) * len(batch_x)
                    total_rows += len(batch_x)
                mean_loss = total_loss / total_rows
                epoch_rows.append(
                    {"fold_id": block["fold_id"], "epoch": epoch_index, "training_loss": mean_loss}
                )
                if epoch_index == 1 or epoch_index % 5 == 0 or epoch_index == int(training_config["epochs"]):
                    print(f"  epoch={epoch_index:02d} train_loss={mean_loss:.6f}", flush=True)

            standardized_prediction = _predict_batches(
                model,
                x_validation,
                batch_size=int(training_config["batch_size"]),
                device=device,
            )
            sequence_prediction = inverse_log_target_normalization(
                standardized_prediction,
                target_fit,
                prediction_min=float(model_config["prediction_min"]),
                prediction_max=float(model_config["prediction_max"]),
            )
            meta = _prediction_meta(block["origins"], validated["horizons"], block["fold_id"], actual)
            control_fold = control.loc[control["fold_id"] == block["fold_id"]].sort_values(key).reset_index(drop=True)
            meta_sorted = meta.sort_values(key).reset_index(drop=True)
            if not control_fold[key].equals(meta_sorted[key]):
                raise AssertionError(f"control keys differ for {block['fold_id']}")
            np.testing.assert_allclose(
                control_fold["actual"].to_numpy(dtype=float),
                meta_sorted["actual"].to_numpy(dtype=float),
                equal_nan=True,
                rtol=0,
                atol=1e-10,
            )
            control_prediction = control_fold["prediction"].to_numpy(dtype=float).reshape(actual.shape)
            sequence_weight = float(config["variants"]["sequence_weight"])
            control_weight = float(config["variants"]["control_weight"])
            if not np.isclose(sequence_weight + control_weight, 1.0):
                raise AssertionError("blend weights must sum to one")
            blend_prediction = sequence_weight * sequence_prediction + control_weight * control_prediction
            if not np.isfinite(sequence_prediction).all() or not np.isfinite(blend_prediction).all():
                raise AssertionError("sequence prediction contains non-finite values")
            for variant, values in (
                (config["variants"]["control"], control_prediction),
                (config["variants"]["diagnostic_sequence"], sequence_prediction),
                (config["variants"]["promotion_candidate"], blend_prediction),
            ):
                part = meta_sorted.copy()
                part["variant"] = variant
                part["prediction"] = values.reshape(-1)
                predictions.append(part)

            model_path = model_dir / f"{block['fold_id']}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": dict(model_config),
                    "input_names": input_names,
                    "feature_median": feature_fit["median"],
                    "feature_scale": feature_fit["scale"],
                    "target_log_mean": target_fit["mean"],
                    "target_log_scale": target_fit["scale"],
                    "latest_training_origin": latest_origin.isoformat(),
                },
                model_path,
            )
            model_records.append(
                {
                    "fold_id": block["fold_id"],
                    "path": model_path.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": sha256(model_path),
                }
            )
            fit_rows.append(
                {
                    "fold_id": block["fold_id"],
                    "training_origins": len(training_origins),
                    "validation_origins": len(block["origins"]),
                    "latest_training_origin": latest_origin,
                    "input_window_steps": window_steps,
                    "input_feature_count": len(input_names),
                    "output_horizons": len(validated["horizons"]),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "final_training_loss": epoch_rows[-1]["training_loss"],
                }
            )
            _append_event(event_log, "fold_completed", fold_id=block["fold_id"], training_origins=len(training_origins))
            del model, optimizer, dataset, loader, x_train, x_validation
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        oof = pd.concat(predictions, ignore_index=True)
        metrics = summarize_predictions(oof)
        gates = evaluate_gate(metrics, config)
        overall = metrics.loc[metrics["scope"] == "overall"].copy()
        control_accuracy = float(
            overall.loc[overall["variant"] == config["variants"]["control"], "accuracy_1_minus_mape"].iloc[0]
        )
        overall["delta_accuracy_pct"] = (overall["accuracy_1_minus_mape"] - control_accuracy) * 100
        comparison = overall[["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]].sort_values("mape")
        frames = {
            "oof_predictions.csv": oof,
            "fit_summary.csv": pd.DataFrame.from_records(fit_rows),
            "epoch_history.csv": pd.DataFrame.from_records(epoch_rows),
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
                "model_count": len(model_records),
                "models": model_records,
                "primary_variant": config["variants"]["promotion_candidate"],
                "primary_long_g1_mape": candidate_mape,
                "component_gate_passed": bool(gate["component_gate_passed"]),
                "standalone_submission_eligible": False,
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
                    "control_run_id": config["control_scientific_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "direct_multistep_gru",
                    "long_g1_mape": candidate_mape,
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"component_gate_passed={bool(gate['component_gate_passed'])}; four ten-day pseudo-tests; one fixed seed; no submission",
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
                    "control_run_id": config["control_scientific_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "direct_multistep_gru",
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
