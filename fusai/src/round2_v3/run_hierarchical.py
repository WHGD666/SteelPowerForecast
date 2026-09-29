"""Controlled hierarchical-target experiment built on the frozen v3 baseline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
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
    build_supervised_long,
    filter_origins_by_stride,
    select_feature_columns,
    summarize_oof,
)
from src.round2_v3.run_baseline import (
    PROJECT_ROOT,
    _append_event,
    _append_registry,
    _git_value,
    _validate_inputs,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "round2_v3" / "hierarchical_v1.yaml"
RUNNER_SOURCE = Path(__file__).resolve()


def derive_small_unit_prediction(
    generator_all_prediction: np.ndarray,
    large_units_prediction: np.ndarray,
    constrained: bool,
    projection: dict[str, float | bool],
) -> np.ndarray:
    """Reconstruct small-unit load, optionally applying declared physical bounds."""
    total = np.asarray(generator_all_prediction, dtype=float)
    large = np.asarray(large_units_prediction, dtype=float)
    if total.shape != large.shape:
        raise ValueError("total and large-unit prediction shapes differ")
    if constrained:
        large = np.maximum(large, float(projection["large_units_min"]))
    small = total - large
    if constrained:
        upper = np.minimum(float(projection["generator_1_max"]), total)
        small = np.minimum(
            np.maximum(small, float(projection["generator_1_min"])), upper
        )
    return small


def _metric_value(metrics: pd.DataFrame, model: str, period: str, target: str = "all") -> float | str:
    scope = "overall" if target == "all" else "target"
    rows = metrics.loc[
        (metrics["model"] == model)
        & (metrics["period"] == period)
        & (metrics["scope"] == scope)
        & (metrics["target"] == target)
    ]
    return "" if rows.empty else float(rows.iloc[0]["mape"])


def _candidate_rows(
    model_name: str,
    fold_id: str,
    period: str,
    meta: pd.DataFrame,
    actual_large: np.ndarray,
    predicted_large: np.ndarray,
    control_total: pd.DataFrame,
    constrained: bool,
    projection: dict[str, float | bool],
) -> pd.DataFrame:
    keys = ["fold_id", "period", "datetime", "horizon_minutes"]
    local = meta[["datetime", "horizon_minutes"]].copy()
    local["fold_id"] = fold_id
    local["period"] = period
    local["actual_large"] = actual_large
    local["predicted_large"] = predicted_large
    merged = local.merge(
        control_total[
            [*keys, "interval_start", "actual", "prediction"]
        ].rename(columns={"actual": "actual_total", "prediction": "predicted_total"}),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if merged[["actual_total", "predicted_total"]].isna().any().any():
        raise AssertionError("control generator_all OOF join is incomplete")
    predicted_small = derive_small_unit_prediction(
        merged["predicted_total"].to_numpy(),
        merged["predicted_large"].to_numpy(),
        constrained,
        projection,
    )
    actual_small = merged["actual_total"].to_numpy() - merged["actual_large"].to_numpy()
    common = {
        "model": model_name,
        "period": period,
        "fold_id": fold_id,
        "datetime": merged["datetime"],
        "interval_start": merged["interval_start"],
        "horizon_minutes": merged["horizon_minutes"],
    }
    small = pd.DataFrame(
        {
            **common,
            "target": "generator_1",
            "actual": actual_small,
            "prediction": predicted_small,
        }
    )
    total = pd.DataFrame(
        {
            **common,
            "target": "generator_all",
            "actual": merged["actual_total"],
            "prediction": merged["predicted_total"],
        }
    )
    return pd.concat([small, total], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    control_config_path = PROJECT_ROOT / config["control_config"]
    control_config = yaml.safe_load(control_config_path.read_text(encoding="utf-8"))
    input_paths = {
        key: PROJECT_ROOT / value for key, value in control_config["inputs"].items()
    }
    manifests = _validate_inputs(control_config, input_paths)
    control_run_dir = (
        PROJECT_ROOT
        / control_config["output"]["root"]
        / config["control_run_id"]
    )
    control_manifest_path = control_run_dir / "run_manifest.json"
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    if control_manifest["status"] != "completed":
        raise AssertionError("control run is not completed")
    control_oof_path = PROJECT_ROOT / control_manifest["artifacts"]["oof_predictions"]
    if sha256(control_oof_path) != control_manifest["artifacts"]["oof_predictions_sha256"]:
        raise AssertionError("control OOF hash mismatch")

    identity_paths = [
        config_path,
        control_config_path,
        control_manifest_path,
        RUNNER_SOURCE,
        *input_paths.values(),
    ]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(sha256(path).encode("ascii"))
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_r2v3_hierarchy_{digest.hexdigest()[:10]}"
    )
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
        "task": "round2_hierarchical_target_experiment",
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "experiment_version": config["experiment_version"],
        "control_run_id": config["control_run_id"],
        "started_at": started_at.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"branch": git_branch, "commit": git_commit, "dirty": git_dirty},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "lightgbm": lgb.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "fingerprints": {
            "experiment_config": sha256(config_path),
            "control_config": sha256(control_config_path),
            "control_manifest": sha256(control_manifest_path),
            "control_oof": sha256(control_oof_path),
            "runner_source": sha256(RUNNER_SOURCE),
            "features": manifests["feature"]["partitions"][0]["output_sha256"],
            "labels": manifests["label"]["output_sha256"],
            "split": manifests["split"]["assignment_sha256"],
        },
        "holdout_evaluated": False,
        "submission_eligible": False,
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_path, "run_started", run_id=run_id)
    registry_written = False

    try:
        feature_path = PROJECT_ROOT / manifests["feature"]["partitions"][0]["output"]
        label_path = PROJECT_ROOT / manifests["label"]["output"]
        features = pd.read_csv(feature_path, low_memory=False)
        features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
        labels = pd.read_csv(label_path, low_memory=False)
        labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
        labels[config["derived_target"]["name"]] = (
            labels["generator_all"] - labels["generator_1"]
        )
        valid_large = labels[config["derived_target"]["name"]].dropna()
        if config["derived_target"]["require_nonnegative_actual"] and (valid_large < 0).any():
            raise AssertionError("derived large-unit actual contains negative values")

        splits = pd.read_csv(input_paths["split_assignments"])
        for column in ("train_end", "latest_eligible_train_origin", "validation_origin"):
            splits[column] = pd.to_datetime(splits[column], errors="raise")
        splits = splits.loc[splits["role"].isin(control_config["fold_roles"])].copy()
        registry = pd.read_csv(input_paths["feature_registry"])
        feature_columns = select_feature_columns(
            registry, control_config["feature_selection"]
        )
        control_oof = pd.read_csv(
            control_oof_path, parse_dates=["datetime", "interval_start"]
        )
        control_lgbm = control_oof.loc[
            control_oof["model"].eq("lightgbm_shared_horizon")
        ].copy()
        control_total = control_lgbm.loc[control_lgbm["target"].eq("generator_all")].copy()
        control_direct = control_lgbm.copy()
        control_direct["model"] = "control_direct"

        model_config = dict(control_config["lightgbm"])
        weight_policy = model_config.pop("sample_weight")
        weight_floor = float(model_config.pop("sample_weight_floor"))
        if weight_policy != "inverse_absolute_target":
            raise ValueError(f"unsupported weight policy: {weight_policy}")
        candidate_parts: list[pd.DataFrame] = [control_direct]

        for fold_number, (fold_id, fold) in enumerate(
            splits.groupby("fold_id", sort=False), start=1
        ):
            latest_train_origin = fold["latest_eligible_train_origin"].iloc[0]
            validation_origins = pd.DatetimeIndex(fold["validation_origin"])
            eligible_train_origins = pd.DatetimeIndex(
                features.loc[features["datetime"] <= latest_train_origin, "datetime"]
            )
            print(
                f"[{fold_number}/{splits['fold_id'].nunique()}] {fold_id} "
                f"val_origins={len(validation_origins)}",
                flush=True,
            )
            for period, period_config in control_config["periods"].items():
                horizons = period_config["horizons_minutes"]
                training_origins = filter_origins_by_stride(
                    eligible_train_origins,
                    period_config["training_origin_stride_minutes"],
                )
                x_train, y_train, _ = build_supervised_long(
                    features,
                    labels,
                    training_origins,
                    horizons,
                    config["derived_target"]["name"],
                    feature_columns,
                )
                x_validation, y_validation, meta = build_supervised_long(
                    features,
                    labels,
                    validation_origins,
                    horizons,
                    config["derived_target"]["name"],
                    feature_columns,
                )
                print(
                    f"  train {period}/large_units: train_rows={len(y_train)} "
                    f"validation_rows={len(y_validation)} features={x_train.shape[1]}",
                    flush=True,
                )
                weights = 1.0 / np.maximum(np.abs(y_train), weight_floor)
                weights = weights / weights.mean()
                model = lgb.LGBMRegressor(**model_config)
                model.fit(x_train, y_train, sample_weight=weights)
                predicted_large = model.predict(x_validation)
                fold_total = control_total.loc[
                    control_total["fold_id"].eq(fold_id)
                    & control_total["period"].eq(period)
                ]
                for variant in config["candidate_variants"]:
                    constrained = variant == "hierarchical_constrained"
                    part = _candidate_rows(
                        variant,
                        fold_id,
                        period,
                        meta,
                        y_validation,
                        predicted_large,
                        fold_total,
                        constrained,
                        config["physical_projection"],
                    )
                    candidate_parts.append(part)
                    part.to_csv(
                        output_dir
                        / "predictions"
                        / "parts"
                        / f"{fold_id}__{period}__{variant}.csv",
                        index=False,
                        encoding="utf-8",
                        date_format="%Y-%m-%d %H:%M:%S",
                        float_format="%.10f",
                    )
                if config["output"]["save_models"]:
                    model.booster_.save_model(
                        str(output_dir / "models" / f"{fold_id}__{period}__large_units.txt")
                    )
                _append_event(
                    event_path,
                    "large_unit_model_completed",
                    fold_id=fold_id,
                    period=period,
                    train_rows=len(y_train),
                    validation_rows=len(y_validation),
                )
                print(f"  done  {period}/large_units", flush=True)
                del x_train, x_validation, y_train, y_validation, model
                gc.collect()

        predictions = pd.concat(candidate_parts, ignore_index=True)
        predictions_path = output_dir / "predictions" / "oof_predictions.csv"
        predictions.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        metrics = summarize_oof(predictions)
        metrics_path = output_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        summary = metrics.loc[metrics["scope"].isin(["overall", "target"])]
        summary_path = output_dir / "summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8", float_format="%.10f")

        primary_candidate = "hierarchical_constrained"
        ended_at = datetime.now(timezone.utc)
        manifest["status"] = "completed"
        manifest["ended_at"] = ended_at.isoformat()
        manifest["duration_seconds"] = round(time.perf_counter() - started_clock, 3)
        manifest["artifacts"] = {
            "predictions": str(predictions_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "predictions_sha256": sha256(predictions_path),
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "metrics_sha256": sha256(metrics_path),
            "summary": str(summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "summary_sha256": sha256(summary_path),
        }
        write_json(output_dir / "run_manifest.json", manifest)
        registry_row = {
            "run_id": run_id,
            "status": "completed",
            "experiment_role": config["experiment_role"],
            "task": "round2_hierarchical_target_experiment",
            "protocol_version": "round2_v3",
            "label_version": manifests["label"]["artifact_version"],
            "split_sha256": manifests["split"]["assignment_sha256"],
            "control_run_id": config["control_run_id"],
            "hypothesis": config["hypothesis"],
            "primary_change": config["primary_change"],
            "model_name": primary_candidate,
            "short_mape": _metric_value(metrics, primary_candidate, "short"),
            "long_mape": _metric_value(metrics, primary_candidate, "long"),
            "short_g1_mape": _metric_value(metrics, primary_candidate, "short", "generator_1"),
            "short_gall_mape": _metric_value(metrics, primary_candidate, "short", "generator_all"),
            "long_g1_mape": _metric_value(metrics, primary_candidate, "long", "generator_1"),
            "long_gall_mape": _metric_value(metrics, primary_candidate, "long", "generator_all"),
            "holdout_evaluated": False,
            "submission_eligible": False,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": manifest["duration_seconds"],
            "artifact_directory": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "notes": "controlled target-representation experiment; generator_all reused from control",
        }
        if config["output"]["append_registry"]:
            _append_registry(registry_row)
            registry_written = True
        _append_event(event_path, "run_completed", duration_seconds=manifest["duration_seconds"])
        print("\n" + summary.to_string(index=False), flush=True)
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
        _append_event(event_path, "run_failed", failure_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": "round2_hierarchical_target_experiment",
                    "protocol_version": "round2_v3",
                    "label_version": manifests["label"]["artifact_version"],
                    "split_sha256": manifests["split"]["assignment_sha256"],
                    "control_run_id": config["control_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "hierarchical_constrained",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
