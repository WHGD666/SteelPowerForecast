"""Build the user-authorized short-g1 dynamic-fuel capped platform probe.

Only short generator_1 changes. Short generator_all and the complete long file
are frozen to the external v29b submission. The source experiment failed its
recent-fold gate, so this builder requires an explicit override in its config.
"""

from __future__ import annotations

import argparse
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
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/submission_short_dynamic_cap_probe_v1.yaml"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCES = [
    RUNNER_SOURCE.with_name("baseline_utils.py"),
    RUNNER_SOURCE.with_name("dynamic_fuel_utils.py"),
    RUNNER_SOURCE.with_name("build_component_ladder.py"),
    RUNNER_SOURCE.with_name("build_submission_baseline.py"),
    RUNNER_SOURCE.with_name("contracts.py"),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    return PROJECT_ROOT / value


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in manifest["partitions"] if row["partition"] == name]
    if len(matches) != 1:
        raise AssertionError(f"feature manifest has {len(matches)} {name} partitions")
    return matches[0]


def _check_manifest_artifacts(manifest: dict[str, Any]) -> list[Path]:
    checked: list[Path] = []
    for key, relative in manifest.get("artifacts", {}).items():
        if key.endswith("_sha256"):
            continue
        expected = manifest["artifacts"].get(f"{key}_sha256")
        if expected is None:
            continue
        path = _resolve(relative)
        if sha256(path) != expected:
            raise AssertionError(f"source artifact hash mismatch: {relative}")
        checked.append(path)
    return checked


def _validate_sources(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    if not config.get("user_override_failed_promotion_gate", False):
        raise AssertionError("failed-gate probe requires explicit user override")

    cap_config_path = _resolve(config["source_cap_config"])
    cap_manifest_path = _resolve(config["source_cap_manifest"])
    cap_config = yaml.safe_load(cap_config_path.read_text(encoding="utf-8"))
    cap_manifest = _read_json(cap_manifest_path)
    if cap_manifest["run_id"] != config["source_cap_run_id"]:
        raise AssertionError("cap run id mismatch")
    if cap_manifest["status"] != "completed":
        raise AssertionError("cap source run is not completed")
    if cap_manifest.get("component_gate_passed") is not False:
        raise AssertionError("override probe expects a failed cap gate")
    cap_key = cap_config_path.relative_to(PROJECT_ROOT).as_posix()
    if cap_manifest["fingerprints"].get(cap_key) != sha256(cap_config_path):
        raise AssertionError("cap config changed after source experiment")
    cap_artifacts = _check_manifest_artifacts(cap_manifest)

    dynamic_config_path = _resolve(config["source_dynamic_config"])
    dynamic_manifest_path = _resolve(config["source_dynamic_manifest"])
    dynamic_config = yaml.safe_load(dynamic_config_path.read_text(encoding="utf-8"))
    dynamic_manifest = _read_json(dynamic_manifest_path)
    if dynamic_manifest["run_id"] != config["source_dynamic_run_id"]:
        raise AssertionError("dynamic run id mismatch")
    if dynamic_manifest["status"] != "completed":
        raise AssertionError("dynamic source run is not completed")
    if dynamic_manifest.get("component_gate_passed") is not False:
        raise AssertionError("dynamic source unexpectedly passed its gate")
    dynamic_key = dynamic_config_path.relative_to(PROJECT_ROOT).as_posix()
    if dynamic_manifest["fingerprints"].get(dynamic_key) != sha256(dynamic_config_path):
        raise AssertionError("dynamic config changed after source experiment")
    dynamic_artifacts = _check_manifest_artifacts(dynamic_manifest)
    if cap_config["control_run_id"] != dynamic_manifest["run_id"]:
        raise AssertionError("cap experiment is not linked to the dynamic source run")

    candidate = config["candidate"]
    source_blend = dynamic_config["variants"]["blend_dynamic70_v16control30"]
    if not np.isclose(
        float(source_blend["dynamic_fuel_weight"]),
        float(candidate["dynamic_fuel_weight"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError("dynamic weight differs from source experiment")
    if not np.isclose(
        float(cap_config["scope"]["correction_cap_mw"]),
        float(candidate["correction_cap_mw"]),
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError("correction cap differs from source experiment")
    if int(dynamic_config["training"]["recent_window_days"]) != int(
        candidate["recent_window_days"]
    ):
        raise AssertionError("recent training window differs from source experiment")
    if [int(value) for value in cap_config["scope"]["horizons_minutes"]] != list(
        SHORT_HORIZONS_MINUTES
    ):
        raise AssertionError("source cap horizons differ from short contract")

    feature_manifest_path = _resolve(config["sources"]["feature_manifest"])
    label_manifest_path = _resolve(config["sources"]["label_manifest"])
    feature_manifest = _read_json(feature_manifest_path)
    label_manifest = _read_json(label_manifest_path)
    train_feature_path = _resolve(_partition(feature_manifest, "train")["output"])
    test_feature_path = _resolve(_partition(feature_manifest, "test")["output"])
    label_path = _resolve(label_manifest["output"])
    if sha256(train_feature_path) != _partition(feature_manifest, "train")["output_sha256"]:
        raise AssertionError("train feature hash mismatch")
    if sha256(test_feature_path) != _partition(feature_manifest, "test")["output_sha256"]:
        raise AssertionError("test feature hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label hash mismatch")
    if feature_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("feature manifest contains future observations")
    if feature_manifest["target_history_included"]:
        raise AssertionError("target history is forbidden")

    bundle_path = _resolve(config["sources"]["v29b_bundle"]["path"])
    if sha256(bundle_path) != config["sources"]["v29b_bundle"]["sha256"]:
        raise AssertionError("v29b source bundle hash mismatch")

    return {
        "cap_config_path": cap_config_path,
        "cap_manifest_path": cap_manifest_path,
        "cap_manifest": cap_manifest,
        "cap_artifacts": cap_artifacts,
        "dynamic_config_path": dynamic_config_path,
        "dynamic_manifest_path": dynamic_manifest_path,
        "dynamic_config": dynamic_config,
        "dynamic_manifest": dynamic_manifest,
        "dynamic_artifacts": dynamic_artifacts,
        "feature_manifest_path": feature_manifest_path,
        "feature_manifest": feature_manifest,
        "label_manifest_path": label_manifest_path,
        "label_manifest": label_manifest,
        "train_feature_path": train_feature_path,
        "test_feature_path": test_feature_path,
        "label_path": label_path,
        "bundle_path": bundle_path,
        "config_path": config_path,
    }


def replace_short_generator_1_with_capped_dynamic(
    base: pd.DataFrame,
    dynamic_matrix: np.ndarray,
    horizons: list[int],
    *,
    dynamic_weight: float,
    correction_cap_mw: float,
    projection: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen capped correction while preserving generator_all."""
    expected_shape = (len(base), len(horizons))
    if dynamic_matrix.shape != expected_shape:
        raise AssertionError(
            f"dynamic matrix shape {dynamic_matrix.shape} != {expected_shape}"
        )
    if not 0.0 < dynamic_weight <= 1.0:
        raise ValueError("dynamic weight must be in (0, 1]")
    if correction_cap_mw <= 0.0:
        raise ValueError("correction cap must be positive")
    g1_columns = [f"generator_1_t+{h}_pred" for h in horizons]
    gall_columns = [f"generator_all_t+{h}_pred" for h in horizons]
    base_g1 = base[g1_columns].to_numpy(dtype=float)
    base_gall = base[gall_columns].to_numpy(dtype=float)
    raw_correction = dynamic_weight * (dynamic_matrix - base_g1)
    capped_correction = np.clip(
        raw_correction, -correction_cap_mw, correction_cap_mw
    )
    candidate = base.copy()
    candidate[g1_columns] = base_g1 + capped_correction
    projected, violations = apply_physical_projection(candidate, horizons, projection)
    final_g1 = projected[g1_columns].to_numpy(dtype=float)
    final_gall = projected[gall_columns].to_numpy(dtype=float)
    if not np.array_equal(base_gall, final_gall):
        raise AssertionError("frozen short generator_all changed")
    final_correction = final_g1 - base_g1
    if float(np.abs(final_correction).max()) > correction_cap_mw + 1e-9:
        raise AssertionError("physical projection violated the correction cap")
    return projected, {
        "formula": "v29b+clip(0.70*(dynamic_fuel_ols-v29b),-5,+5)",
        "pre_projection_violations": violations,
        "generator_all_exactly_frozen": True,
        "g1_cells": int(final_correction.size),
        "source_cells_exceeding_cap": int(
            np.count_nonzero(np.abs(raw_correction) > correction_cap_mw)
        ),
        "raw_mean_abs_correction_mw": float(np.abs(raw_correction).mean()),
        "raw_max_abs_correction_mw": float(np.abs(raw_correction).max()),
        "final_mean_abs_correction_mw": float(np.abs(final_correction).mean()),
        "final_max_abs_correction_mw": float(np.abs(final_correction).max()),
        "changed_g1_cells": int(np.count_nonzero(final_correction)),
        "positive_correction_cells": int(np.count_nonzero(final_correction > 0.0)),
        "negative_correction_cells": int(np.count_nonzero(final_correction < 0.0)),
        "generator_1_min": float(final_g1.min()),
        "generator_1_max": float(final_g1.max()),
    }


def _identity_paths(sources: dict[str, Any]) -> list[Path]:
    paths = [
        sources["config_path"],
        sources["cap_config_path"],
        sources["cap_manifest_path"],
        sources["dynamic_config_path"],
        sources["dynamic_manifest_path"],
        sources["feature_manifest_path"],
        sources["label_manifest_path"],
        sources["train_feature_path"],
        sources["test_feature_path"],
        sources["label_path"],
        sources["bundle_path"],
        RUNNER_SOURCE,
        *HELPER_SOURCES,
        *sources["cap_artifacts"],
        *sources["dynamic_artifacts"],
    ]
    return list(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sources = _validate_sources(config, config_path)
    identity_paths = _identity_paths(sources)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"source_cap_run={config['source_cap_run_id']} "
            f"source_gate_passed={sources['cap_manifest']['component_gate_passed']} "
            f"identity_files={len(identity_paths)} user_override=true"
        )
        return

    created_at = datetime.now(timezone.utc)
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    submission_id = (
        created_at.astimezone().strftime("%Y%m%dT%H%M%S")
        + f"_short_dynamic_cap_probe_v1_{digest.hexdigest()[:10]}"
    )
    output_dir = _resolve(config["output"]["root"]) / submission_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite submission: {output_dir}")
    payload_dir = output_dir / "payload"
    model_dir = output_dir / "models"
    payload_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "submission_id": submission_id,
        "status": "building",
        "source_run_id": config["source_cap_run_id"],
        "control_submission_id": config["control_submission_id"],
        "protocol_version": "round2_v3",
        "submission_version": config["submission_version"],
        "experiment_role": config["experiment_role"],
        "hypothesis": config["hypothesis"],
        "primary_change": config["primary_change"],
        "created_at": created_at.isoformat(),
        "command": {
            "working_directory": str(PROJECT_ROOT),
            "argv": [sys.executable, *sys.argv],
        },
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
            "authorization": "explicit user request on 2026-10-05",
            "source_component_gate_passed": False,
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
        horizons = [int(value) for value in SHORT_HORIZONS_MINUTES]
        frequency = int(feature_config["origin_frequency_minutes"])
        window_days = int(config["candidate"]["recent_window_days"])
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
                dynamic_test, test_origins, [horizon], dynamic_columns
            )
            if not pd.DatetimeIndex(test_meta["datetime"]).equals(test_origins):
                raise AssertionError(f"test origin order mismatch at h={horizon}")
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("regressor", LinearRegression()),
                ]
            )
            model.fit(x_train[model_columns], y_train)
            dynamic_matrix[:, horizon_index] = model.predict(x_test[model_columns])
            model_path = model_dir / f"dynamic_fuel_short_g1_h{horizon:04d}.joblib"
            joblib.dump(model, model_path, compress=3)
            training_records.append(
                {
                    "period": "short",
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
                        "coefficient_standardized": float(
                            regressor.coef_[feature_index]
                        ),
                        "intercept": float(regressor.intercept_),
                    }
                )
            print(f"trained short generator_1 h={horizon}", flush=True)

        coefficients_path = output_dir / "coefficients.csv"
        pd.DataFrame.from_records(coefficient_records).to_csv(
            coefficients_path, index=False, encoding="utf-8", float_format="%.10f"
        )
        feature_registry_path = output_dir / "dynamic_fuel_feature_registry.csv"
        dynamic_registry.to_csv(feature_registry_path, index=False, encoding="utf-8")
        dynamic_prediction_path = output_dir / "dynamic_fuel_short_g1_predictions.csv"
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
        with zipfile.ZipFile(sources["bundle_path"], "r") as source_archive:
            if source_archive.testzip() is not None:
                raise AssertionError("v29b source bundle CRC failed")
            loaded = load_component(source_archive, v29b_config)
        v29b_short, _ = loaded["s_result.csv"]
        v29b_long, v29b_long_bytes = loaded["l_result.csv"]
        validate_prediction_frame(v29b_short, test_origins, SHORT_HORIZONS_MINUTES)
        validate_prediction_frame(v29b_long, test_origins, LONG_HORIZONS_MINUTES)

        candidate_short, change_report = replace_short_generator_1_with_capped_dynamic(
            v29b_short,
            dynamic_matrix,
            horizons,
            dynamic_weight=float(config["candidate"]["dynamic_fuel_weight"]),
            correction_cap_mw=float(config["candidate"]["correction_cap_mw"]),
            projection=config["physical_projection"],
        )
        short_validation = validate_prediction_frame(
            candidate_short, test_origins, SHORT_HORIZONS_MINUTES
        )
        long_validation = validate_prediction_frame(
            v29b_long, test_origins, LONG_HORIZONS_MINUTES
        )

        short_bytes = frame_bytes(candidate_short, config["output"]["float_format"])
        short_path = payload_dir / "s_result.csv"
        long_path = payload_dir / "l_result.csv"
        short_path.write_bytes(short_bytes)
        long_path.write_bytes(v29b_long_bytes)
        archive_path = output_dir / config["output"]["archive_filename"]
        timestamp = tuple(int(value) for value in config["output"]["zip_timestamp"])
        write_deterministic_zip(
            archive_path,
            [("s_result.csv", short_bytes), ("l_result.csv", v29b_long_bytes)],
            timestamp,  # type: ignore[arg-type]
        )
        with zipfile.ZipFile(archive_path, "r") as archive:
            if archive.namelist() != ["s_result.csv", "l_result.csv"]:
                raise AssertionError("submission ZIP member order differs")
            if archive.testzip() is not None:
                raise AssertionError("submission ZIP CRC failed")
            if archive.read("s_result.csv") != short_bytes:
                raise AssertionError("ZIP short member differs from candidate bytes")
            if archive.read("l_result.csv") != v29b_long_bytes:
                raise AssertionError("ZIP long member differs from frozen v29b bytes")

        manifest.update(
            {
                "status": "ready_for_upload",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "training": training_records,
                "candidate": config["candidate"],
                "frozen_blocks": config["frozen_blocks"],
                "validation": {
                    "short": short_validation,
                    "long": long_validation,
                    "change_vs_v29b": change_report,
                    "short_generator_all_exactly_v29b": True,
                    "long_bytes_exactly_v29b": True,
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
                    "source_run_id": config["source_cap_run_id"],
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
                        "Explicit user-authorized failed-gate probe. Only short generator_1 "
                        "changed; short generator_all and complete long file frozen to v29b."
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
