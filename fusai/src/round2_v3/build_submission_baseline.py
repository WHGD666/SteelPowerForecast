"""Fit the frozen baseline on all eligible history and build a checked ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.baseline_utils import (
    build_prediction_long,
    build_supervised_long,
    filter_origins_by_stride,
    select_feature_columns,
)
from src.round2_v3.contracts import (
    expected_result_columns,
    validate_prediction_frame,
)
from src.round2_v3.run_baseline import PROJECT_ROOT, _git_value, _validate_inputs


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "round2_v3" / "submission_baseline_v1.yaml"
SUBMISSION_REGISTRY = PROJECT_ROOT / "submissions" / "round2_v3" / "registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()


def apply_physical_projection(
    frame: pd.DataFrame,
    horizons: list[int],
    projection: dict[str, float | bool],
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    g1_columns = [f"generator_1_t+{h}_pred" for h in horizons]
    gall_columns = [f"generator_all_t+{h}_pred" for h in horizons]
    g1 = result[g1_columns].to_numpy(float)
    gall = result[gall_columns].to_numpy(float)
    pre = {
        "generator_1_nonpositive": int((g1 <= 0).sum()),
        "generator_all_nonpositive": int((gall <= 0).sum()),
        "generator_1_above_200": int((g1 > float(projection["generator_1_max"])).sum()),
        "generator_all_above_440": int((gall > float(projection["generator_all_max"])).sum()),
        "generator_1_above_generator_all": int((g1 > gall).sum()),
    }
    gall = np.clip(
        gall,
        float(projection["generator_all_min"]),
        float(projection["generator_all_max"]),
    )
    g1 = np.clip(
        g1,
        float(projection["generator_1_min"]),
        float(projection["generator_1_max"]),
    )
    if projection["enforce_generator_1_lte_generator_all"]:
        g1 = np.minimum(g1, gall)
    result[g1_columns] = g1
    result[gall_columns] = gall
    return result, pre


def predictions_to_wide(
    origins: pd.DatetimeIndex,
    horizons: list[int],
    predictions_by_target: dict[str, np.ndarray],
) -> pd.DataFrame:
    result = pd.DataFrame({"datetime": origins})
    expected_cells = len(origins) * len(horizons)
    for target in ("generator_1", "generator_all"):
        values = np.asarray(predictions_by_target[target], dtype=float)
        if len(values) != expected_cells:
            raise ValueError(f"unexpected prediction count for {target}")
        matrix = values.reshape(len(origins), len(horizons))
        for index, horizon in enumerate(horizons):
            result[f"{target}_t+{horizon}_pred"] = matrix[:, index]
    expected = expected_result_columns(horizons)
    if list(result.columns) != expected:
        raise AssertionError("wide prediction column order differs from contract")
    return result


def _append_submission_registry(row: dict[str, object]) -> None:
    with SUBMISSION_REGISTRY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        existing = {record["submission_id"] for record in reader}
    if not fieldnames:
        raise RuntimeError("submission registry has no header")
    if str(row["submission_id"]) in existing:
        raise RuntimeError(f"submission already registered: {row['submission_id']}")
    with SUBMISSION_REGISTRY.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_config_path = PROJECT_ROOT / config["source_baseline_config"]
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    input_paths = {
        key: PROJECT_ROOT / value for key, value in baseline_config["inputs"].items()
    }
    manifests = _validate_inputs(baseline_config, input_paths)
    source_run_dir = (
        PROJECT_ROOT
        / baseline_config["output"]["root"]
        / config["source_development_run_id"]
    )
    source_manifest_path = source_run_dir / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["status"] != "completed" or source_manifest["experiment_role"] != "competition":
        raise AssertionError("source development run is not an eligible completed competition run")

    test_manifest_path = PROJECT_ROOT / config["test_feature_manifest"]
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    test_feature_path = PROJECT_ROOT / test_manifest["output"]
    if sha256(test_feature_path) != test_manifest["output_sha256"]:
        raise AssertionError("test feature artifact hash mismatch")
    if not test_manifest["history_contiguous_across_boundary"]:
        raise AssertionError("test features do not preserve train-test history")
    if test_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("test features contain future sources")

    digest = hashlib.sha256()
    for path in (
        config_path,
        baseline_config_path,
        source_manifest_path,
        test_manifest_path,
        RUNNER_SOURCE,
        *input_paths.values(),
    ):
        digest.update(sha256(path).encode("ascii"))
    short_hash = digest.hexdigest()[:10]
    submission_id = (
        created_at.astimezone().strftime("%Y%m%dT%H%M%S")
        + f"_calibration_baseline_v1_{short_hash}"
    )
    output_dir = PROJECT_ROOT / config["output"]["root"] / submission_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite submission: {output_dir}")
    payload_dir = output_dir / "payload"
    model_dir = output_dir / "models"
    payload_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "submission_id": submission_id,
        "status": "building",
        "source_run_id": config["source_development_run_id"],
        "protocol_version": "round2_v3",
        "submission_version": config["submission_version"],
        "hypothesis": config["hypothesis"],
        "primary_change": config["primary_change"],
        "created_at": created_at.isoformat(),
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--porcelain")),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "fingerprints": {
            "submission_config": sha256(config_path),
            "baseline_config": sha256(baseline_config_path),
            "source_run_manifest": sha256(source_manifest_path),
            "test_feature_manifest": sha256(test_manifest_path),
            "runner_source": sha256(RUNNER_SOURCE),
            "train_features": manifests["feature"]["partitions"][0]["output_sha256"],
            "test_features": test_manifest["output_sha256"],
            "labels": manifests["label"]["output_sha256"],
        },
        "platform_result": None,
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "submission_manifest.json", manifest)

    try:
        train_feature_path = PROJECT_ROOT / manifests["feature"]["partitions"][0]["output"]
        label_path = PROJECT_ROOT / manifests["label"]["output"]
        train_features = pd.read_csv(train_feature_path, low_memory=False)
        test_features = pd.read_csv(test_feature_path, low_memory=False)
        train_features["datetime"] = pd.to_datetime(train_features["datetime"], errors="raise")
        test_features["datetime"] = pd.to_datetime(test_features["datetime"], errors="raise")
        labels = pd.read_csv(label_path, low_memory=False)
        labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
        registry = pd.read_csv(input_paths["feature_registry"])
        feature_columns = select_feature_columns(
            registry, baseline_config["feature_selection"]
        )
        test_origins = pd.DatetimeIndex(test_features["datetime"])
        if len(test_origins) != 960 or test_origins.duplicated().any():
            raise AssertionError("test origin contract failed")
        raw_train_end = labels["datetime"].max() + pd.Timedelta(minutes=14)

        model_config = dict(baseline_config["lightgbm"])
        weight_policy = model_config.pop("sample_weight")
        weight_floor = float(model_config.pop("sample_weight_floor"))
        if weight_policy != "inverse_absolute_target":
            raise ValueError(f"unsupported weight policy: {weight_policy}")
        result_frames: dict[str, pd.DataFrame] = {}
        training_records = []

        for period, period_config in baseline_config["periods"].items():
            horizons = [int(value) for value in period_config["horizons_minutes"]]
            latest_origin = (raw_train_end - pd.Timedelta(minutes=max(horizons) - 1)).floor(
                "15min"
            )
            eligible = pd.DatetimeIndex(
                train_features.loc[train_features["datetime"] <= latest_origin, "datetime"]
            )
            training_origins = filter_origins_by_stride(
                eligible, period_config["training_origin_stride_minutes"]
            )
            x_test, test_meta = build_prediction_long(
                test_features, test_origins, horizons, feature_columns
            )
            if not test_meta["datetime"].equals(
                pd.Series(np.repeat(test_origins.to_numpy(), len(horizons)))
            ):
                raise AssertionError("test design matrix origin order mismatch")
            predictions_by_target: dict[str, np.ndarray] = {}
            for target in baseline_config["targets"]:
                x_train, y_train, _ = build_supervised_long(
                    train_features,
                    labels,
                    training_origins,
                    horizons,
                    target,
                    feature_columns,
                )
                print(
                    f"train {period}/{target}: train_rows={len(y_train)} "
                    f"test_rows={len(x_test)} features={x_train.shape[1]}",
                    flush=True,
                )
                weights = 1.0 / np.maximum(np.abs(y_train), weight_floor)
                weights = weights / weights.mean()
                model = lgb.LGBMRegressor(**model_config)
                model.fit(x_train, y_train, sample_weight=weights)
                predictions_by_target[target] = model.predict(x_test)
                model_path = model_dir / f"full__{period}__{target}.txt"
                model.booster_.save_model(str(model_path))
                training_records.append(
                    {
                        "period": period,
                        "target": target,
                        "latest_train_origin": latest_origin.isoformat(),
                        "training_origins": int(len(training_origins)),
                        "training_rows": int(len(y_train)),
                        "test_rows": int(len(x_test)),
                        "model": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "model_sha256": sha256(model_path),
                    }
                )
                print(f"done  {period}/{target}", flush=True)
            wide = predictions_to_wide(test_origins, horizons, predictions_by_target)
            projected, pre_violations = apply_physical_projection(
                wide, horizons, config["physical_projection"]
            )
            validation = validate_prediction_frame(projected, test_origins, horizons)
            validation["pre_projection_violations"] = pre_violations
            validation["latest_train_origin"] = latest_origin.isoformat()
            result_frames[period] = projected
            write_json(output_dir / f"validation_{period}.json", validation)

        short_path = payload_dir / "s_result.csv"
        long_path = payload_dir / "l_result.csv"
        result_frames["short"].to_csv(
            short_path,
            index=False,
            encoding=config["output"]["encoding"],
            date_format="%Y-%m-%d %H:%M:%S",
            float_format=config["output"]["float_format"],
        )
        result_frames["long"].to_csv(
            long_path,
            index=False,
            encoding=config["output"]["encoding"],
            date_format="%Y-%m-%d %H:%M:%S",
            float_format=config["output"]["float_format"],
        )
        archive_path = output_dir / config["archive_filename"]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(short_path, arcname="s_result.csv")
            archive.write(long_path, arcname="l_result.csv")
        with zipfile.ZipFile(archive_path, "r") as archive:
            if archive.namelist() != ["s_result.csv", "l_result.csv"]:
                raise AssertionError("ZIP member contract failed")
            if archive.testzip() is not None:
                raise AssertionError("ZIP integrity test failed")

        manifest["status"] = "ready_for_upload"
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["training"] = training_records
        manifest["artifacts"] = {
            "s_result": str(short_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "s_result_sha256": sha256(short_path),
            "l_result": str(long_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "l_result_sha256": sha256(long_path),
            "archive": str(archive_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "archive_sha256": sha256(archive_path),
            "archive_members": ["s_result.csv", "l_result.csv"],
        }
        write_json(output_dir / "submission_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_submission_registry(
                {
                    "submission_id": submission_id,
                    "source_run_id": config["source_development_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "archive_filename": config["archive_filename"],
                    "archive_sha256": manifest["artifacts"]["archive_sha256"],
                    "s_result_sha256": manifest["artifacts"]["s_result_sha256"],
                    "l_result_sha256": manifest["artifacts"]["l_result_sha256"],
                    "anomaly_status": "not_submitted",
                    "is_official_best": False,
                    "reproducibility_status": "ready_for_upload",
                    "notes": "first round2_v3 external calibration candidate",
                }
            )
        print(
            f"PASS submission_id={submission_id} archive={archive_path} "
            f"sha256={manifest['artifacts']['archive_sha256']} "
            f"duration_seconds={manifest['duration_seconds']}",
            flush=True,
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["duration_seconds"] = round(time.perf_counter() - started, 3)
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "submission_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
