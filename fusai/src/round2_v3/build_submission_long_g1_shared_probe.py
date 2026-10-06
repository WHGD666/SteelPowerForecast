"""Build an isolated v29b long-g1 shared-horizon v2 platform probe."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
import time
import traceback
import zipfile
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
    build_v28_features,
    fit_linear_fuel_proxy,
    merge_training_tables,
    predict_linear_fuel,
    to_v28_grid,
)
from src.round2_v3.baseline_utils import TARGET_CALENDAR_COLUMNS, build_prediction_long
from src.round2_v3.build_component_ladder import frame_bytes, load_component, write_deterministic_zip
from src.round2_v3.build_submission_baseline import _append_submission_registry
from src.round2_v3.build_submission_long_gall_shared_probe import _git_value, _read_json, _resolve
from src.round2_v3.contracts import LONG_HORIZONS_MINUTES, SHORT_HORIZONS_MINUTES, validate_prediction_frame
from src.round2_v3.run_long_g1_shared_horizon import _align_calendar_to_point_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/submission_long_g1_shared_probe_v1.yaml"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("anomaly_weight_utils.py"),
    RUNNER_SOURCE.with_name("baseline_utils.py"),
    RUNNER_SOURCE.with_name("build_component_ladder.py"),
    RUNNER_SOURCE.with_name("build_submission_baseline.py"),
    RUNNER_SOURCE.with_name("build_submission_long_gall_shared_probe.py"),
    RUNNER_SOURCE.with_name("contracts.py"),
    RUNNER_SOURCE.with_name("run_long_g1_shared_horizon.py"),
]


def replace_long_generator_1(
    base: pd.DataFrame,
    predictions: np.ndarray,
    horizons: list[int],
    projection: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = base.copy()
    g1_columns = [f"generator_1_t+{h}_pred" for h in horizons]
    gall_columns = [f"generator_all_t+{h}_pred" for h in horizons]
    g1_before = base[g1_columns].to_numpy(dtype=float)
    gall_before = base[gall_columns].to_numpy(dtype=float)
    raw = np.asarray(predictions, dtype=float)
    if raw.shape != g1_before.shape:
        raise ValueError(f"long g1 matrix shape {raw.shape} != {g1_before.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("long g1 predictions contain non-finite values")
    projected = np.clip(raw, float(projection["generator_1_min"]), float(projection["generator_1_max"]))
    above_gall = projected > gall_before
    if bool(projection["enforce_generator_1_lte_generator_all"]):
        projected = np.minimum(projected, gall_before)
    result[g1_columns] = projected
    if not np.array_equal(result[gall_columns].to_numpy(dtype=float), gall_before):
        raise AssertionError("frozen v29b long generator_all changed")
    delta = np.abs(projected - g1_before)
    return result, {
        "long_generator_all_exactly_frozen": True,
        "generator_1_cells": int(delta.size),
        "generator_1_cells_changed_vs_v29b": int(np.count_nonzero(delta)),
        "generator_1_mean_abs_change_vs_v29b": float(delta.mean()),
        "generator_1_max_abs_change_vs_v29b": float(delta.max()),
        "raw_nonpositive_cells": int((raw <= 0).sum()),
        "raw_above_200_cells": int((raw > float(projection["generator_1_max"])).sum()),
        "projected_down_to_frozen_gall_cells": int(above_gall.sum()),
        "generator_1_min": float(projected.min()),
        "generator_1_max": float(projected.max()),
    }


def _validate_sources(config: dict[str, Any]) -> dict[str, Any]:
    source_config_path = _resolve(config["source_experiment_config"])
    source_manifest_path = _resolve(config["source_run_manifest"])
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    source_manifest = _read_json(source_manifest_path)
    if source_manifest["run_id"] != config["source_development_run_id"] or source_manifest["status"] != "completed":
        raise AssertionError("source scientific run identity/status mismatch")
    if source_manifest["primary_variant"] != source_config["variants"]["promotion_candidate"]:
        raise AssertionError("source primary variant differs from experiment config")
    if source_manifest["standalone_submission_eligible"]:
        raise AssertionError("this calibration contract expects a non-standalone source component")
    config_key = source_config_path.relative_to(PROJECT_ROOT).as_posix()
    if source_manifest["fingerprints"].get(config_key) != sha256(source_config_path):
        raise AssertionError("source experiment config no longer matches its run")
    source_runner = RUNNER_SOURCE.with_name("run_long_g1_shared_horizon.py")
    runner_key = source_runner.relative_to(PROJECT_ROOT).as_posix()
    if source_manifest["fingerprints"].get(runner_key) != sha256(source_runner):
        raise AssertionError("source scientific runner no longer matches its run")
    gate_path = _resolve(source_manifest["artifacts"]["gate_results"])
    if sha256(gate_path) != source_manifest["artifacts"]["gate_results_sha256"]:
        raise AssertionError("source gate results hash mismatch")
    gate = pd.read_csv(gate_path).iloc[0]
    if float(gate["overall_accuracy_gain_pct"]) < 0.5 or float(gate["single_fold_min_gain_pct"]) < 0:
        raise AssertionError("source g1 component no longer meets its preregistered gain/fold evidence")
    fingerprint_path = _resolve(config["sources"]["raw_fingerprints"])
    expected = {item["file"]: item["sha256"] for item in _read_json(fingerprint_path)["files"]}
    train_paths = [_resolve(value) for value in config["sources"]["raw_train_tables"]]
    test_paths = [_resolve(value) for value in config["sources"]["raw_test_tables"]]
    for path in [*train_paths, *test_paths]:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if sha256(path) != expected.get(relative):
            raise AssertionError(f"raw table hash mismatch: {relative}")
    bundle = _resolve(config["sources"]["v29b_bundle"]["path"])
    if sha256(bundle) != config["sources"]["v29b_bundle"]["sha256"]:
        raise AssertionError("v29b bundle hash mismatch")
    return {"source_config_path": source_config_path, "source_manifest_path": source_manifest_path, "source_config": source_config, "source_manifest": source_manifest, "gate_path": gate_path, "gate": gate.to_dict(), "fingerprint_path": fingerprint_path, "train_paths": train_paths, "test_paths": test_paths, "bundle": bundle}


def _prepare_design(sources: dict[str, Any]) -> dict[str, Any]:
    train_raw = merge_training_tables(sources["train_paths"])
    test_raw = merge_training_tables(sources["test_paths"])
    grid = to_v28_grid(pd.concat([train_raw, test_raw], ignore_index=True, sort=False).sort_values("datetime").reset_index(drop=True))
    train_end = pd.Timestamp(train_raw["datetime"].max()).floor("15min")
    test_origins = pd.DatetimeIndex(grid.loc[grid["datetime"] > train_end, "datetime"])
    expected = pd.date_range("2025-10-01 00:00:00", "2025-10-10 23:45:00", freq="15min")
    if not test_origins.equals(expected):
        raise AssertionError("test origin timeline differs from submission contract")
    features, feature_columns = build_v28_features(grid)
    feature_frame = features.copy()
    feature_frame.insert(0, "datetime", grid["datetime"].to_numpy())
    return {"grid": grid, "feature_frame": feature_frame, "feature_columns": feature_columns, "train_end": train_end, "test_origins": test_origins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources = _validate_sources(config)
    design = _prepare_design(sources)
    source_config = sources["source_config"]
    horizons = [int(value) for value in source_config["horizons_minutes"]]
    if horizons != list(LONG_HORIZONS_MINUTES):
        raise AssertionError("source horizons differ from submission contract")
    if args.preflight_only:
        print(f"PASS preflight train_end={design['train_end']} test_origins={len(design['test_origins'])} origin_features={len(design['feature_columns'])} shared_features={len(design['feature_columns']) + len(TARGET_CALENDAR_COLUMNS)} changed_block=long_generator_1", flush=True)
        return

    identity_paths = [config_path, sources["source_config_path"], sources["source_manifest_path"], sources["gate_path"], sources["fingerprint_path"], sources["bundle"], *sources["train_paths"], *sources["test_paths"], RUNNER_SOURCE, *HELPER_SOURCES]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    submission_id = created_at.astimezone().strftime("%Y%m%dT%H%M%S") + "_long_g1_shared_probe_v1_" + digest.hexdigest()[:10]
    output_dir = PROJECT_ROOT / config["output"]["root"] / submission_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite submission: {output_dir}")
    payload_dir, model_dir = output_dir / "payload", output_dir / "models"
    payload_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "submission_id": submission_id, "status": "building", "source_run_id": config["source_development_run_id"], "control_submission_id": config["control_submission_id"], "experiment_role": config["experiment_role"], "submission_version": config["submission_version"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "created_at": created_at.isoformat(), "holdout_evaluated": False, "platform_result": None,
        "known_risks": config["known_risks"],
        "git": {"branch": _git_value("branch", "--show-current"), "commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pyyaml": yaml.__version__},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths}, "failure": {"type": None, "message": None, "traceback": None}, "artifacts": {},
    }
    write_json(output_dir / "submission_manifest.json", manifest)
    try:
        grid, feature_frame, feature_columns = design["grid"], design["feature_frame"], design["feature_columns"]
        train_end, test_origins = design["train_end"], design["test_origins"]
        tree_start = pd.Timestamp(source_config["training"]["g1_tree_start"])
        training_origins = pd.DatetimeIndex(grid.loc[(grid["datetime"] >= tree_start) & (grid["datetime"] <= train_end), "datetime"])
        x_train, train_meta = build_prediction_long(feature_frame, training_origins, horizons, feature_columns)
        x_train = _align_calendar_to_point_target(x_train, train_meta)
        target_times = pd.DatetimeIndex(train_meta["datetime"]) + pd.TimedeltaIndex(pd.to_timedelta(train_meta["horizon_minutes"].to_numpy(), unit="min"))
        point_series = pd.to_numeric(grid.set_index("datetime")["generator_1"], errors="coerce")
        y_train = point_series.reindex(target_times).to_numpy(dtype=float)
        valid = np.isfinite(y_train) & (target_times <= train_end)
        x_train, y_train = x_train.loc[valid].reset_index(drop=True), y_train[valid]
        x_test, test_meta = build_prediction_long(feature_frame, test_origins, horizons, feature_columns)
        x_test = _align_calendar_to_point_target(x_test, test_meta)
        print(f"train shared long/generator_1: train_rows={len(y_train)} test_rows={len(x_test)} features={x_train.shape[1]}", flush=True)
        model = lgb.LGBMRegressor(**dict(source_config["lightgbm"]))
        model.fit(x_train, y_train)
        tree = np.clip(model.predict(x_test), 0.0, None)
        model_path = model_dir / "full__long__generator_1_shared_horizon.txt"
        model.booster_.save_model(str(model_path))
        fuel = source_config["fuel_recipe"]
        train_grid = grid.loc[grid["datetime"] <= train_end].copy()
        fuel_train = train_grid.loc[train_grid["datetime"] >= pd.Timestamp(fuel["fit_start"])]
        coefficients = fit_linear_fuel_proxy(fuel_train, "generator_1", FUELS_G1)
        test_grid = grid.set_index("datetime").loc[test_origins].reset_index()
        fuel_origin = predict_linear_fuel(test_grid, FUELS_G1, coefficients)
        origin_index = pd.DatetimeIndex(test_meta["datetime"])
        fuel_flat = pd.Series(fuel_origin, index=test_origins).reindex(origin_index).to_numpy(dtype=float)
        horizon_flat = test_meta["horizon_minutes"].to_numpy(dtype=int)
        weights = np.where(horizon_flat <= int(fuel["near_horizon_max_minutes"]), float(fuel["near_weight"]), float(fuel["far_weight"]))
        g1_matrix = ((1.0 - weights) * tree + weights * fuel_flat).reshape(len(test_origins), len(horizons))

        v29b = config["sources"]["v29b_bundle"]
        with zipfile.ZipFile(sources["bundle"], "r") as archive:
            if archive.testzip() is not None:
                raise AssertionError("v29b source bundle CRC failed")
            loaded = load_component(archive, v29b)
        v29b_short, short_bytes = loaded["s_result.csv"]
        v29b_long, _ = loaded["l_result.csv"]
        validate_prediction_frame(v29b_short, test_origins, SHORT_HORIZONS_MINUTES)
        validate_prediction_frame(v29b_long, test_origins, LONG_HORIZONS_MINUTES)
        candidate_long, change_report = replace_long_generator_1(v29b_long, g1_matrix, horizons, config["physical_projection"])
        short_validation = validate_prediction_frame(v29b_short, test_origins, SHORT_HORIZONS_MINUTES)
        long_validation = validate_prediction_frame(candidate_long, test_origins, LONG_HORIZONS_MINUTES)
        prediction_path = output_dir / "full_fit_long_g1_predictions.csv"
        prediction_frame = pd.DataFrame({"datetime": test_origins})
        for index, horizon in enumerate(horizons):
            prediction_frame[f"generator_1_t+{horizon}_pred"] = g1_matrix[:, index]
        prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8", float_format="%.10f", date_format="%Y-%m-%d %H:%M:%S")
        short_path, long_path = payload_dir / "s_result.csv", payload_dir / "l_result.csv"
        short_path.write_bytes(short_bytes)
        long_bytes = frame_bytes(candidate_long, config["output"]["float_format"])
        long_path.write_bytes(long_bytes)
        archive_path = output_dir / config["output"]["archive_filename"]
        timestamp = tuple(int(value) for value in config["output"]["zip_timestamp"])
        write_deterministic_zip(archive_path, [("s_result.csv", short_bytes), ("l_result.csv", long_bytes)], timestamp)  # type: ignore[arg-type]
        with zipfile.ZipFile(archive_path, "r") as archive:
            if archive.namelist() != ["s_result.csv", "l_result.csv"] or archive.testzip() is not None or archive.read("s_result.csv") != short_bytes or archive.read("l_result.csv") != long_bytes:
                raise AssertionError("candidate ZIP integrity/identity check failed")
        model_sha = sha256(model_path)
        manifest.update({
            "status": "ready_for_upload", "duration_seconds": round(time.perf_counter() - started, 3),
            "training": [{"period": "long", "target": "generator_1", "tree_start": tree_start.isoformat(), "train_end": train_end.isoformat(), "training_rows": int(len(y_train)), "test_rows": int(len(x_test)), "features": int(x_train.shape[1]), "model_parameters": dict(source_config["lightgbm"]), "fuel_coefficients": coefficients.tolist(), "model": model_path.relative_to(PROJECT_ROOT).as_posix(), "model_sha256": model_sha}],
            "frozen_blocks": config["frozen_blocks"], "source_oof_evidence": sources["gate"],
            "validation": {"short": short_validation, "long": long_validation, "change_vs_v29b": change_report, "short_bytes_exactly_v29b": True},
            "artifacts": {"model": model_path.relative_to(PROJECT_ROOT).as_posix(), "model_sha256": model_sha, "full_fit_predictions": prediction_path.relative_to(PROJECT_ROOT).as_posix(), "full_fit_predictions_sha256": sha256(prediction_path), "s_result": short_path.relative_to(PROJECT_ROOT).as_posix(), "s_result_sha256": sha256(short_path), "l_result": long_path.relative_to(PROJECT_ROOT).as_posix(), "l_result_sha256": sha256(long_path), "archive": archive_path.relative_to(PROJECT_ROOT).as_posix(), "archive_sha256": sha256(archive_path), "archive_members": ["s_result.csv", "l_result.csv"]},
        })
        write_json(output_dir / "submission_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_submission_registry({"submission_id": submission_id, "source_run_id": config["source_development_run_id"], "control_submission_id": config["control_submission_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "archive_filename": config["output"]["archive_filename"], "archive_sha256": manifest["artifacts"]["archive_sha256"], "s_result_sha256": manifest["artifacts"]["s_result_sha256"], "l_result_sha256": manifest["artifacts"]["l_result_sha256"], "anomaly_status": "not_submitted", "is_official_best": False, "reproducibility_status": "ready_for_upload_isolated_platform_probe", "notes": "Only long generator_1 changed; v29b short bytes and long generator_all values frozen; source OOF all folds and horizon buckets positive."})
        print(f"PASS submission_id={submission_id} archive={archive_path} sha256={manifest['artifacts']['archive_sha256']} duration_seconds={manifest['duration_seconds']}", flush=True)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output_dir / "submission_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()

