"""Build the user-authorized high-risk dynamic-fuel platform probe.

The builder freezes the external v29b short file and both generator_all blocks.
Only long generator_1 is replaced, using the exact development formula:
70% horizon-specific dynamic-fuel OLS + 30% frozen v3 baseline prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io_utils import sha256, write_json
from src.round2_v3.baseline_utils import build_prediction_long, build_supervised_long
from src.round2_v3.build_component_ladder import (
    bytes_sha256,
    frame_bytes,
    load_component,
    write_deterministic_zip,
)
from src.round2_v3.build_submission_baseline import (
    _append_submission_registry,
    apply_physical_projection,
)
from src.round2_v3.contracts import (
    LONG_HORIZONS_MINUTES,
    SHORT_HORIZONS_MINUTES,
    validate_prediction_frame,
)
from src.round2_v3.dynamic_fuel_utils import (
    build_dynamic_fuel_features,
    latest_mature_origin,
    recent_origin_window,
)
from src.round2_v3.run_baseline import _git_value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "submission_dynamic_fuel_probe_v1.yaml"
)
RUNNER_SOURCE = Path(__file__).resolve()
DYNAMIC_UTILS_SOURCE = RUNNER_SOURCE.with_name("dynamic_fuel_utils.py")
BASELINE_UTILS_SOURCE = RUNNER_SOURCE.with_name("baseline_utils.py")
COMPONENT_BUILDER_SOURCE = RUNNER_SOURCE.with_name("build_component_ladder.py")
SUBMISSION_BUILDER_SOURCE = RUNNER_SOURCE.with_name("build_submission_baseline.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in manifest["partitions"] if row["partition"] == name]
    if len(rows) != 1:
        raise AssertionError(f"feature manifest has {len(rows)} {name} partitions")
    return rows[0]


def _validate_sources(
    config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    source_run_dir = (
        PROJECT_ROOT
        / "outputs"
        / "round2_v3"
        / "dynamic_fuel_runs"
        / config["source_development_run_id"]
    )
    source_run_manifest_path = source_run_dir / "run_manifest.json"
    source_run = _read_json(source_run_manifest_path)
    if source_run["status"] != "completed":
        raise AssertionError("source dynamic-fuel run is not completed")
    if source_run["run_id"] != config["source_development_run_id"]:
        raise AssertionError("source dynamic-fuel run id mismatch")
    if source_run["promotion_gate_passed_variants"]:
        raise AssertionError("probe override expected a failed promotion gate")
    if not config["user_override_failed_promotion_gate"]:
        raise AssertionError("failed-gate platform probe requires explicit config override")

    dynamic_config_path = PROJECT_ROOT / config["source_dynamic_config"]
    dynamic_config = yaml.safe_load(dynamic_config_path.read_text(encoding="utf-8"))
    if dynamic_config["experiment_version"] != source_run["experiment_version"]:
        raise AssertionError("dynamic config and source run experiment versions differ")
    if int(dynamic_config["training"]["recent_window_days"]) != 21:
        raise AssertionError("this probe is frozen to the 21-day strongest OOF version")
    blend = dynamic_config["variants"]["blend_dynamic70_control30"]
    if not np.isclose(
        float(blend["dynamic_fuel_weight"]),
        float(config["blend"]["dynamic_fuel_weight"]),
    ) or not np.isclose(
        float(blend["control_weight"]),
        float(config["blend"]["baseline_control_weight"]),
    ):
        raise AssertionError("submission blend differs from source development formula")

    feature_manifest_path = PROJECT_ROOT / config["sources"]["feature_manifest"]
    label_manifest_path = PROJECT_ROOT / config["sources"]["label_manifest"]
    feature_manifest = _read_json(feature_manifest_path)
    label_manifest = _read_json(label_manifest_path)
    train_feature_path = PROJECT_ROOT / _partition(feature_manifest, "train")["output"]
    test_feature_path = PROJECT_ROOT / _partition(feature_manifest, "test")["output"]
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(train_feature_path) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("train feature hash mismatch")
    if sha256(test_feature_path) != _partition(feature_manifest, "test")["output_sha256"]:
        raise AssertionError("test feature hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label hash mismatch")
    if feature_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("feature manifest contains future observations")
    if feature_manifest["target_history_included"]:
        raise AssertionError("dynamic-fuel submission forbids target history")

    baseline_manifest_path = (
        PROJECT_ROOT / config["sources"]["baseline_submission_manifest"]
    )
    baseline_manifest = _read_json(baseline_manifest_path)
    if baseline_manifest["source_run_id"] != dynamic_config["control_run_id"]:
        raise AssertionError("baseline submission does not match OOF control run")
    baseline_long_path = PROJECT_ROOT / baseline_manifest["artifacts"]["l_result"]
    expected_baseline_hash = config["sources"]["baseline_l_result_sha256"]
    if sha256(baseline_long_path) != expected_baseline_hash:
        raise AssertionError("baseline long-result hash mismatch")

    bundle = PROJECT_ROOT / config["sources"]["v29b_bundle"]["path"]
    if sha256(bundle) != config["sources"]["v29b_bundle"]["sha256"]:
        raise AssertionError("v29b source bundle hash mismatch")
    return {
        "source_run_dir": source_run_dir,
        "source_run_manifest_path": source_run_manifest_path,
        "source_run": source_run,
        "dynamic_config_path": dynamic_config_path,
        "dynamic_config": dynamic_config,
        "feature_manifest_path": feature_manifest_path,
        "feature_manifest": feature_manifest,
        "label_manifest_path": label_manifest_path,
        "label_manifest": label_manifest,
        "train_feature_path": train_feature_path,
        "test_feature_path": test_feature_path,
        "label_path": label_path,
        "baseline_manifest_path": baseline_manifest_path,
        "baseline_manifest": baseline_manifest,
        "baseline_long_path": baseline_long_path,
        "bundle_path": bundle,
        "config_path": config_path,
    }


def replace_long_generator_1(
    v29b_long: pd.DataFrame,
    baseline_long: pd.DataFrame,
    dynamic_matrix: np.ndarray,
    horizons: list[int],
    dynamic_weight: float,
    baseline_weight: float,
    projection: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace only long g1 and preserve v29b generator_all exactly."""
    if list(v29b_long.columns) != list(baseline_long.columns):
        raise AssertionError("v29b and baseline long schemas differ")
    if not v29b_long["datetime"].equals(baseline_long["datetime"]):
        raise AssertionError("v29b and baseline long origins differ")
    expected_shape = (len(v29b_long), len(horizons))
    if dynamic_matrix.shape != expected_shape:
        raise AssertionError(
            f"dynamic matrix shape {dynamic_matrix.shape} != {expected_shape}"
        )
    g1_columns = [f"generator_1_t+{h}_pred" for h in horizons]
    gall_columns = [f"generator_all_t+{h}_pred" for h in horizons]
    baseline_values = baseline_long[g1_columns].to_numpy(dtype=float)
    raw_candidate = dynamic_weight * dynamic_matrix + baseline_weight * baseline_values
    result = v29b_long.copy()
    result[g1_columns] = raw_candidate
    frozen_gall = result[gall_columns].to_numpy(dtype=float).copy()
    projected, violations = apply_physical_projection(result, horizons, projection)
    if not np.array_equal(
        projected[gall_columns].to_numpy(dtype=float), frozen_gall
    ):
        raise AssertionError("physical projection changed frozen v29b generator_all")
    v29b_g1 = v29b_long[g1_columns].to_numpy(dtype=float)
    final_g1 = projected[g1_columns].to_numpy(dtype=float)
    return projected, {
        "formula": "0.70*dynamic_fuel_ols+0.30*v3_baseline_long_g1",
        "pre_projection_violations": violations,
        "generator_all_exactly_frozen": True,
        "g1_cells": int(final_g1.size),
        "g1_cells_changed_vs_v29b": int(np.count_nonzero(final_g1 - v29b_g1)),
        "g1_mean_abs_change_vs_v29b": float(np.abs(final_g1 - v29b_g1).mean()),
        "g1_max_abs_change_vs_v29b": float(np.abs(final_g1 - v29b_g1).max()),
        "g1_min": float(final_g1.min()),
        "g1_max": float(final_g1.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources = _validate_sources(config, config_path)

    identity_paths = [
        config_path,
        sources["source_run_manifest_path"],
        sources["dynamic_config_path"],
        sources["feature_manifest_path"],
        sources["label_manifest_path"],
        sources["baseline_manifest_path"],
        sources["baseline_long_path"],
        sources["bundle_path"],
        RUNNER_SOURCE,
        DYNAMIC_UTILS_SOURCE,
        BASELINE_UTILS_SOURCE,
        COMPONENT_BUILDER_SOURCE,
        SUBMISSION_BUILDER_SOURCE,
    ]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    submission_id = (
        created_at.astimezone().strftime("%Y%m%dT%H%M%S")
        + f"_dynamic_fuel_probe_v1_{digest.hexdigest()[:10]}"
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
        "control_submission_id": config["control_submission_id"],
        "protocol_version": "round2_v3",
        "submission_version": config["submission_version"],
        "experiment_role": config["experiment_role"],
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
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "pyyaml": yaml.__version__,
        },
        "selection_gate_override": {
            "enabled": True,
            "reason": "explicit user request to package the strongest local version",
            "source_promotion_passed": False,
            "known_risks": config["known_risks"],
        },
        "holdout_evaluated": False,
        "platform_result": None,
        "failure": {"type": None, "message": None, "traceback": None},
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in identity_paths
        },
        "training": [],
        "artifacts": {},
    }
    write_json(output_dir / "submission_manifest.json", manifest)

    try:
        dynamic_config = sources["dynamic_config"]
        feature_config = dynamic_config["dynamic_fuel_features"]
        current_columns = ["datetime", *feature_config["fuel_columns"]]
        train_origin = pd.read_csv(
            sources["train_feature_path"], usecols=current_columns, low_memory=False
        )
        test_origin = pd.read_csv(
            sources["test_feature_path"], usecols=current_columns, low_memory=False
        )
        train_origin["datetime"] = pd.to_datetime(train_origin["datetime"], errors="raise")
        test_origin["datetime"] = pd.to_datetime(test_origin["datetime"], errors="raise")
        combined = pd.concat([train_origin, test_origin], ignore_index=True)
        dynamic_all, dynamic_registry = build_dynamic_fuel_features(
            combined,
            feature_config["fuel_columns"],
            feature_config["lag_steps"],
            feature_config["rolling_mean_steps"],
            frequency_minutes=int(feature_config["origin_frequency_minutes"]),
            include_current=bool(feature_config["include_current"]),
        )
        train_end = train_origin["datetime"].max()
        dynamic_train = dynamic_all.loc[dynamic_all["datetime"] <= train_end].copy()
        dynamic_test = dynamic_all.loc[dynamic_all["datetime"] > train_end].copy()
        dynamic_columns = dynamic_registry["feature_name"].tolist()
        model_columns = [*dynamic_columns, *feature_config["target_calendar_columns"]]
        test_origins = pd.DatetimeIndex(dynamic_test["datetime"])
        expected_origins = pd.date_range(
            "2025-10-01 00:00:00", "2025-10-10 23:45:00", freq="15min"
        )
        if not test_origins.equals(expected_origins):
            raise AssertionError("dynamic test origins differ from submission contract")

        labels = pd.read_csv(sources["label_path"], low_memory=False)
        labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
        raw_train_end = labels["datetime"].max() + pd.Timedelta(minutes=14)
        horizons = [int(value) for value in LONG_HORIZONS_MINUTES]
        frequency = int(feature_config["origin_frequency_minutes"])
        window_days = int(dynamic_config["training"]["recent_window_days"])
        train_times = pd.DatetimeIndex(dynamic_train["datetime"])
        dynamic_matrix = np.empty((len(test_origins), len(horizons)), dtype=float)
        coefficient_records: list[dict[str, object]] = []
        training_records: list[dict[str, object]] = []

        for horizon_index, horizon in enumerate(horizons):
            latest_origin = latest_mature_origin(
                raw_train_end, horizon, frequency_minutes=frequency
            )
            training_origins = recent_origin_window(
                train_times,
                latest_origin,
                window_days=window_days,
                frequency_minutes=frequency,
            )
            x_train, y_train, _ = build_supervised_long(
                dynamic_train,
                labels,
                training_origins,
                [horizon],
                "generator_1",
                dynamic_columns,
            )
            x_test, test_meta = build_prediction_long(
                dynamic_test,
                test_origins,
                [horizon],
                dynamic_columns,
            )
            if not pd.DatetimeIndex(test_meta["datetime"]).equals(test_origins):
                raise AssertionError(f"test design origin order mismatch at h={horizon}")
            model = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    ),
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression()),
                ]
            )
            model.fit(x_train[model_columns], y_train)
            dynamic_matrix[:, horizon_index] = model.predict(x_test[model_columns])
            model_path = model_dir / f"dynamic_fuel_long_g1_h{horizon:04d}.joblib"
            joblib.dump(model, model_path, compress=3)
            training_records.append(
                {
                    "period": "long",
                    "target": "generator_1",
                    "horizon_minutes": horizon,
                    "latest_train_origin": latest_origin.isoformat(),
                    "training_origin_start": training_origins.min().isoformat(),
                    "training_origin_end": training_origins.max().isoformat(),
                    "training_origins": int(len(training_origins)),
                    "training_rows": int(len(y_train)),
                    "test_rows": int(len(x_test)),
                    "model": model_path.relative_to(PROJECT_ROOT).as_posix(),
                    "model_sha256": sha256(model_path),
                }
            )
            regressor = model.named_steps["regressor"]
            for feature_index, feature_name in enumerate(model_columns):
                coefficient_records.append(
                    {
                        "horizon_minutes": horizon,
                        "feature_name": feature_name,
                        "coefficient_standardized": float(regressor.coef_[feature_index]),
                        "intercept": float(regressor.intercept_),
                    }
                )
            if (horizon_index + 1) % 16 == 0:
                print(
                    f"trained horizons={horizon_index + 1}/{len(horizons)} "
                    f"latest_h={horizon}",
                    flush=True,
                )

        coefficients_path = output_dir / "coefficients.csv"
        pd.DataFrame.from_records(coefficient_records).to_csv(
            coefficients_path,
            index=False,
            encoding="utf-8",
            float_format="%.10f",
        )
        feature_registry_path = output_dir / "dynamic_fuel_feature_registry.csv"
        dynamic_registry.to_csv(feature_registry_path, index=False, encoding="utf-8")
        dynamic_prediction_path = output_dir / "dynamic_fuel_long_g1_predictions.csv"
        dynamic_wide = pd.DataFrame({"datetime": test_origins})
        for index, horizon in enumerate(horizons):
            dynamic_wide[f"generator_1_t+{horizon}_pred"] = dynamic_matrix[:, index]
        dynamic_wide.to_csv(
            dynamic_prediction_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )

        v29b_config = config["sources"]["v29b_bundle"]
        with zipfile.ZipFile(sources["bundle_path"], "r") as archive:
            if archive.testzip() is not None:
                raise AssertionError("v29b source bundle CRC failed")
            loaded = load_component(archive, v29b_config)
        v29b_short, v29b_short_bytes = loaded["s_result.csv"]
        v29b_long, _ = loaded["l_result.csv"]
        baseline_long = pd.read_csv(sources["baseline_long_path"], low_memory=False)
        validate_prediction_frame(v29b_short, test_origins, SHORT_HORIZONS_MINUTES)
        validate_prediction_frame(v29b_long, test_origins, LONG_HORIZONS_MINUTES)
        validate_prediction_frame(baseline_long, test_origins, LONG_HORIZONS_MINUTES)

        candidate_long, change_report = replace_long_generator_1(
            v29b_long,
            baseline_long,
            dynamic_matrix,
            horizons,
            float(config["blend"]["dynamic_fuel_weight"]),
            float(config["blend"]["baseline_control_weight"]),
            config["physical_projection"],
        )
        short_validation = validate_prediction_frame(
            v29b_short, test_origins, SHORT_HORIZONS_MINUTES
        )
        long_validation = validate_prediction_frame(
            candidate_long, test_origins, LONG_HORIZONS_MINUTES
        )

        short_path = payload_dir / "s_result.csv"
        long_path = payload_dir / "l_result.csv"
        short_path.write_bytes(v29b_short_bytes)
        long_bytes = frame_bytes(candidate_long, config["output"]["float_format"])
        long_path.write_bytes(long_bytes)
        archive_path = output_dir / config["output"]["archive_filename"]
        timestamp = tuple(int(value) for value in config["output"]["zip_timestamp"])
        write_deterministic_zip(
            archive_path,
            [("s_result.csv", v29b_short_bytes), ("l_result.csv", long_bytes)],
            timestamp,  # type: ignore[arg-type]
        )
        with zipfile.ZipFile(archive_path, "r") as archive:
            if archive.namelist() != ["s_result.csv", "l_result.csv"]:
                raise AssertionError("submission ZIP member order differs")
            if archive.testzip() is not None:
                raise AssertionError("submission ZIP CRC failed")
            if archive.read("s_result.csv") != v29b_short_bytes:
                raise AssertionError("ZIP short member differs from frozen v29b bytes")
            if archive.read("l_result.csv") != long_bytes:
                raise AssertionError("ZIP long member differs from candidate bytes")

        manifest.update(
            {
                "status": "ready_for_upload",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "training": training_records,
                "blend": config["blend"],
                "frozen_blocks": config["frozen_blocks"],
                "validation": {
                    "short": short_validation,
                    "long": long_validation,
                    "change_vs_v29b": change_report,
                    "short_bytes_exactly_v29b": True,
                    "long_generator_all_exactly_v29b": True,
                },
                "artifacts": {
                    "s_result": short_path.relative_to(PROJECT_ROOT).as_posix(),
                    "s_result_sha256": sha256(short_path),
                    "l_result": long_path.relative_to(PROJECT_ROOT).as_posix(),
                    "l_result_sha256": sha256(long_path),
                    "archive": archive_path.relative_to(PROJECT_ROOT).as_posix(),
                    "archive_sha256": sha256(archive_path),
                    "archive_members": ["s_result.csv", "l_result.csv"],
                    "dynamic_predictions": dynamic_prediction_path.relative_to(PROJECT_ROOT).as_posix(),
                    "dynamic_predictions_sha256": sha256(dynamic_prediction_path),
                    "coefficients": coefficients_path.relative_to(PROJECT_ROOT).as_posix(),
                    "coefficients_sha256": sha256(coefficients_path),
                    "feature_registry": feature_registry_path.relative_to(PROJECT_ROOT).as_posix(),
                    "feature_registry_sha256": sha256(feature_registry_path),
                },
            }
        )
        write_json(output_dir / "submission_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_submission_registry(
                {
                    "submission_id": submission_id,
                    "source_run_id": config["source_development_run_id"],
                    "control_submission_id": config["control_submission_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "archive_filename": config["output"]["archive_filename"],
                    "archive_sha256": manifest["artifacts"]["archive_sha256"],
                    "s_result_sha256": manifest["artifacts"]["s_result_sha256"],
                    "l_result_sha256": manifest["artifacts"]["l_result_sha256"],
                    "anomaly_status": "not_submitted_high_risk_probe",
                    "is_official_best": False,
                    "reproducibility_status": "ready_for_upload_user_override_failed_gate",
                    "notes": (
                        "Explicit user-authorized probe; source OOF promotion gate failed. "
                        "Short and long generator_all frozen to v29b; only long generator_1 changed."
                    ),
                }
            )
        print(
            "PASS "
            f"submission_id={submission_id} archive={archive_path} "
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

