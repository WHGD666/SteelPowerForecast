"""Run the first formal round2 v3 calendar and LightGBM OOF baselines."""

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

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.baseline_utils import (
    TARGET_CALENDAR_COLUMNS,
    build_supervised_long,
    filter_origins_by_stride,
    fit_calendar_profile,
    predict_calendar_profile,
    select_feature_columns,
    summarize_oof,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "round2_v3" / "baseline_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "round2_v3" / "registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
BASELINE_UTILS_SOURCE = RUNNER_SOURCE.with_name("baseline_utils.py")
METRICS_SOURCE = RUNNER_SOURCE.with_name("metrics.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _run_identity(config_path: Path, input_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *input_paths]:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_r2v3_baseline_{digest.hexdigest()[:10]}"


def _append_event(path: Path, event: str, **values: object) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        existing = {record["run_id"] for record in reader}
    if not fieldnames:
        raise RuntimeError("experiment registry has no header")
    if str(row["run_id"]) in existing:
        raise RuntimeError(f"run_id already registered: {row['run_id']}")
    normalized = {field: row.get(field, "") for field in fieldnames}
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(normalized)


def _validate_inputs(config: dict, paths: dict[str, Path]) -> dict[str, dict]:
    feature_manifest = _read_json(paths["feature_manifest"])
    label_manifest = _read_json(paths["label_manifest"])
    split_manifest = _read_json(paths["split_manifest"])
    for partition in feature_manifest["partitions"]:
        artifact = PROJECT_ROOT / partition["output"]
        if sha256(artifact) != partition["output_sha256"]:
            raise AssertionError(f"feature artifact hash mismatch: {artifact}")
    label_artifact = PROJECT_ROOT / label_manifest["output"]
    if sha256(label_artifact) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    if sha256(paths["feature_registry"]) != feature_manifest["registry_sha256"]:
        raise AssertionError("feature registry hash mismatch")
    if feature_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("feature artifact contains future-dependent sources")
    if feature_manifest["target_history_included"]:
        raise AssertionError("v1 baseline forbids observed target history")
    if config["holdout_evaluated"] or config["submission_eligible"]:
        raise AssertionError("OOF baseline cannot claim holdout or submission status")
    return {
        "feature": feature_manifest,
        "label": label_manifest,
        "split": split_manifest,
    }


def _prediction_part(
    meta: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
    model: str,
    period: str,
    fold_id: str,
) -> pd.DataFrame:
    result = meta.copy()
    result["actual"] = actual
    result["prediction"] = prediction
    result["model"] = model
    result["period"] = period
    result["fold_id"] = fold_id
    return result[
        [
            "model",
            "period",
            "fold_id",
            "datetime",
            "interval_start",
            "target",
            "horizon_minutes",
            "actual",
            "prediction",
        ]
    ]


def _metric_value(
    metrics: pd.DataFrame, period: str, target: str = "all"
) -> float | str:
    scope = "overall" if target == "all" else "target"
    rows = metrics.loc[
        (metrics["model"] == "lightgbm_shared_horizon")
        & (metrics["period"] == period)
        & (metrics["scope"] == scope)
        & (metrics["target"] == target)
    ]
    return "" if rows.empty else float(rows.iloc[0]["mape"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {
        key: PROJECT_ROOT / value for key, value in config["inputs"].items()
    }
    manifests = _validate_inputs(config, paths)
    identity_inputs = [
        *paths.values(),
        RUNNER_SOURCE,
        BASELINE_UTILS_SOURCE,
        METRICS_SOURCE,
    ]
    run_id = _run_identity(config_path, identity_inputs)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")
    (output_dir / "models").mkdir(parents=True, exist_ok=False)
    (output_dir / "predictions" / "parts").mkdir(parents=True, exist_ok=False)
    event_path = output_dir / "events.jsonl"

    git_commit = _git_value("rev-parse", "HEAD")
    git_branch = _git_value("branch", "--show-current")
    git_dirty = bool(_git_value("status", "--porcelain"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": "round2_short_and_long_forecast",
        "target_variant": "generator_1_and_generator_all",
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "baseline_version": config["baseline_version"],
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "command": {
            "working_directory": str(PROJECT_ROOT),
            "argv": [sys.executable, *sys.argv],
        },
        "git": {
            "branch": git_branch,
            "commit": git_commit,
            "dirty": git_dirty,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "dependencies": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "lightgbm": lgb.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "fingerprints": {
            "run_config": sha256(config_path),
            "feature_manifest": sha256(paths["feature_manifest"]),
            "feature_artifact_train": manifests["feature"]["partitions"][0][
                "output_sha256"
            ],
            "labels": manifests["label"]["output_sha256"],
            "split": manifests["split"]["assignment_sha256"],
            "feature_registry": manifests["feature"]["registry_sha256"],
            "runner_source": sha256(RUNNER_SOURCE),
            "baseline_utils_source": sha256(BASELINE_UTILS_SOURCE),
            "metrics_source": sha256(METRICS_SOURCE),
        },
        "holdout_evaluated": False,
        "submission_eligible": False,
        "models": config["models"],
        "lightgbm_parameters": config["lightgbm"],
        "artifacts": {},
        "failure": {"type": None, "message": None, "traceback": None},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_path, "run_started", run_id=run_id)
    predictions: list[pd.DataFrame] = []
    registry_written = False

    try:
        feature_path = PROJECT_ROOT / manifests["feature"]["partitions"][0]["output"]
        label_path = PROJECT_ROOT / manifests["label"]["output"]
        origin_features = pd.read_csv(feature_path, low_memory=False)
        origin_features["datetime"] = pd.to_datetime(
            origin_features["datetime"], errors="raise"
        )
        interval_targets = pd.read_csv(label_path, low_memory=False)
        interval_targets["datetime"] = pd.to_datetime(
            interval_targets["datetime"], errors="raise"
        )
        split_assignments = pd.read_csv(paths["split_assignments"])
        for column in (
            "train_end",
            "latest_eligible_train_origin",
            "validation_origin",
            "short_label_end",
            "long_label_end",
        ):
            split_assignments[column] = pd.to_datetime(
                split_assignments[column], errors="raise"
            )
        roles = set(config["fold_roles"])
        split_assignments = split_assignments.loc[
            split_assignments["role"].isin(roles)
        ].copy()
        registry = pd.read_csv(paths["feature_registry"])
        feature_columns = select_feature_columns(
            registry, config["feature_selection"]
        )
        if not feature_columns:
            raise ValueError("baseline selected zero origin features")
        (output_dir / "feature_names.txt").write_text(
            "\n".join([*feature_columns, *TARGET_CALENDAR_COLUMNS]) + "\n",
            encoding="utf-8",
        )
        manifest["selected_origin_feature_count"] = len(feature_columns)
        manifest["model_feature_count"] = len(feature_columns) + len(
            TARGET_CALENDAR_COLUMNS
        )
        _append_event(
            event_path,
            "inputs_loaded",
            selected_origin_features=len(feature_columns),
            fold_count=int(split_assignments["fold_id"].nunique()),
        )

        model_config = dict(config["lightgbm"])
        weight_policy = model_config.pop("sample_weight")
        weight_floor = float(model_config.pop("sample_weight_floor"))
        if weight_policy != "inverse_absolute_target":
            raise ValueError(f"unsupported sample weight policy: {weight_policy}")

        for fold_number, (fold_id, fold) in enumerate(
            split_assignments.groupby("fold_id", sort=False), start=1
        ):
            train_end = fold["train_end"].iloc[0]
            latest_train_origin = fold["latest_eligible_train_origin"].iloc[0]
            validation_origins = pd.DatetimeIndex(fold["validation_origin"])
            eligible_train_origins = pd.DatetimeIndex(
                origin_features.loc[
                    origin_features["datetime"] <= latest_train_origin, "datetime"
                ]
            )
            print(
                f"[{fold_number}/{split_assignments['fold_id'].nunique()}] {fold_id} "
                f"train_end={train_end} val_origins={len(validation_origins)}",
                flush=True,
            )
            for period, period_config in config["periods"].items():
                horizons = period_config["horizons_minutes"]
                training_origins = filter_origins_by_stride(
                    eligible_train_origins,
                    period_config["training_origin_stride_minutes"],
                )
                for target in config["targets"]:
                    x_train, y_train, _ = build_supervised_long(
                        origin_features,
                        interval_targets,
                        training_origins,
                        horizons,
                        target,
                        feature_columns,
                    )
                    x_validation, y_validation, validation_meta = build_supervised_long(
                        origin_features,
                        interval_targets,
                        validation_origins,
                        horizons,
                        target,
                        feature_columns,
                    )
                    expected_validation_rows = len(validation_origins) * len(horizons)
                    if len(y_validation) != expected_validation_rows:
                        raise AssertionError(
                            f"validation labels incomplete for {fold_id}/{period}/{target}: "
                            f"{len(y_validation)} != {expected_validation_rows}"
                        )

                    print(
                        f"  train {period}/{target}: train_rows={len(y_train)} "
                        f"validation_rows={len(y_validation)} features={x_train.shape[1]}",
                        flush=True,
                    )
                    profile = fit_calendar_profile(interval_targets, target, train_end)
                    calendar_prediction = predict_calendar_profile(
                        profile, validation_meta["interval_start"]
                    )
                    calendar_part = _prediction_part(
                        validation_meta,
                        y_validation,
                        calendar_prediction,
                        "calendar_profile",
                        period,
                        fold_id,
                    )
                    predictions.append(calendar_part)

                    weights = 1.0 / np.maximum(np.abs(y_train), weight_floor)
                    weights = weights / weights.mean()
                    model = lgb.LGBMRegressor(**model_config)
                    model.fit(x_train, y_train, sample_weight=weights)
                    model_prediction = model.predict(x_validation)
                    print(
                        f"  done  {period}/{target}",
                        flush=True,
                    )
                    model_part = _prediction_part(
                        validation_meta,
                        y_validation,
                        model_prediction,
                        "lightgbm_shared_horizon",
                        period,
                        fold_id,
                    )
                    predictions.append(model_part)
                    part_name = f"{fold_id}__{period}__{target}.csv"
                    pd.concat([calendar_part, model_part], ignore_index=True).to_csv(
                        output_dir / "predictions" / "parts" / part_name,
                        index=False,
                        encoding="utf-8",
                        date_format="%Y-%m-%d %H:%M:%S",
                        float_format="%.10f",
                    )
                    if config["output"]["save_models"]:
                        model.booster_.save_model(
                            str(output_dir / "models" / f"{fold_id}__{period}__{target}.txt")
                        )
                    _append_event(
                        event_path,
                        "model_completed",
                        fold_id=fold_id,
                        period=period,
                        target=target,
                        train_rows=len(y_train),
                        validation_rows=len(y_validation),
                        feature_count=x_train.shape[1],
                    )
                    del x_train, x_validation, y_train, y_validation, model
                    gc.collect()

        oof = pd.concat(predictions, ignore_index=True)
        oof_path = output_dir / "predictions" / "oof_predictions.csv"
        if config["output"]["save_oof_predictions"]:
            oof.to_csv(
                oof_path,
                index=False,
                encoding="utf-8",
                date_format="%Y-%m-%d %H:%M:%S",
                float_format="%.10f",
            )
        metrics = summarize_oof(oof)
        metrics_path = output_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        summary = metrics.loc[metrics["scope"].isin(["overall", "target"])].copy()
        summary_path = output_dir / "summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8", float_format="%.10f")

        ended_at = datetime.now(timezone.utc)
        manifest["status"] = "completed"
        manifest["ended_at"] = ended_at.isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - started_clock, 3)
        manifest["artifacts"] = {
            "oof_predictions": str(oof_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "oof_predictions_sha256": sha256(oof_path),
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "metrics_sha256": sha256(metrics_path),
            "summary": str(summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "summary_sha256": sha256(summary_path),
            "models_directory": str((output_dir / "models").relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
        }
        write_json(output_dir / "run_manifest.json", manifest)
        registry_row = {
            "run_id": run_id,
            "status": "completed",
            "experiment_role": config["experiment_role"],
            "task": "round2_short_and_long_forecast",
            "protocol_version": "round2_v3",
            "label_version": manifests["label"]["artifact_version"],
            "split_sha256": manifests["split"]["assignment_sha256"],
            "control_run_id": "",
            "hypothesis": config["hypothesis"],
            "primary_change": config["primary_change"],
            "model_name": "calendar_profile+lightgbm_shared_horizon",
            "short_mape": _metric_value(metrics, "short"),
            "long_mape": _metric_value(metrics, "long"),
            "short_g1_mape": _metric_value(metrics, "short", "generator_1"),
            "short_gall_mape": _metric_value(metrics, "short", "generator_all"),
            "long_g1_mape": _metric_value(metrics, "long", "generator_1"),
            "long_gall_mape": _metric_value(metrics, "long", "generator_all"),
            "holdout_evaluated": False,
            "submission_eligible": False,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": manifest["duration_seconds"],
            "artifact_directory": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "notes": "first formal v3 OOF baseline; platform test not evaluated",
        }
        if config["output"]["append_registry"]:
            _append_registry(registry_row)
            registry_written = True
        _append_event(event_path, "run_completed", duration_seconds=manifest["duration_seconds"])
        lightgbm_overall = summary.loc[
            (summary["model"] == "lightgbm_shared_horizon")
            & (summary["scope"] == "overall")
        ][["period", "mape", "accuracy_1_minus_mape"]]
        print("\n" + lightgbm_overall.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']}",
            flush=True,
        )
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        manifest["status"] = "failed"
        manifest["ended_at"] = ended_at.isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - started_clock, 3)
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(
            event_path,
            "run_failed",
            failure_type=type(exc).__name__,
            message=str(exc),
        )
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": "round2_short_and_long_forecast",
                    "protocol_version": "round2_v3",
                    "label_version": manifests["label"]["artifact_version"],
                    "split_sha256": manifests["split"]["assignment_sha256"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "calendar_profile+lightgbm_shared_horizon",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": str(output_dir.relative_to(PROJECT_ROOT)).replace(
                        "\\", "/"
                    ),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
