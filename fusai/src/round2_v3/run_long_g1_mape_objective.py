"""Evaluate an inverse-target weighted L1 tree inside the frozen v28 long-g1 recipe."""

from __future__ import annotations

import argparse
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
    v28_g1_fuel_weight,
)
from src.round2_v3.event_ablation_utils import attach_event_context
from src.round2_v3.mape_objective_utils import normalized_mape_weights
from src.round2_v3.run_anomaly_weight_g1 import (
    _append_event,
    _append_registry,
    _git_value,
    _summarize_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "long_g1_mape_objective_v1.yaml"
)
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("mape_objective_utils.py")
ANOMALY_UTILS_SOURCE = RUNNER_SOURCE.with_name("anomaly_weight_utils.py")
EVENT_UTILS_SOURCE = RUNNER_SOURCE.with_name("event_ablation_utils.py")
CONTROL_RUNNER_SOURCE = RUNNER_SOURCE.with_name("run_anomaly_weight_g1.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_identity(config_path: Path, inputs: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in [config_path, *inputs]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_long_g1_mape_{digest.hexdigest()[:10]}"


def _validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    raw_manifest_path = PROJECT_ROOT / inputs["raw_fingerprints"]
    raw_manifest = _read_json(raw_manifest_path)
    expected_hashes = {item["file"]: item["sha256"] for item in raw_manifest["files"]}
    raw_tables = [PROJECT_ROOT / item for item in inputs["raw_train_tables"]]
    for path in raw_tables:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if expected_hashes.get(relative) != sha256(path):
            raise AssertionError(f"raw table fingerprint mismatch: {relative}")

    label_manifest_path = PROJECT_ROOT / inputs["label_manifest"]
    label_manifest = _read_json(label_manifest_path)
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    split_manifest_path = PROJECT_ROOT / inputs["split_manifest"]
    split_manifest = _read_json(split_manifest_path)
    split_path = PROJECT_ROOT / inputs["split_assignments"]
    if sha256(split_path) != split_manifest["assignment_sha256"]:
        raise AssertionError("split artifact hash mismatch")
    control_path = PROJECT_ROOT / inputs["control_oof"]
    if sha256(control_path) != inputs["control_oof_sha256"]:
        raise AssertionError("frozen control OOF hash mismatch")

    experiment = config["experiment"]
    if experiment["target"] != "generator_1":
        raise AssertionError("experiment must remain generator_1 only")
    if experiment["tree_recipe"] != "v28_exact":
        raise AssertionError("tree recipe must remain v28_exact")
    if experiment["fuel_recipe"] != "v28_exact_0.8_near_0.6_far":
        raise AssertionError("fuel recipe differs from v28")
    if experiment["future_process_observations_allowed"]:
        raise AssertionError("future process observations are forbidden")
    if experiment["true_target_history_features_allowed"]:
        raise AssertionError("target history features are forbidden")
    if experiment["platform_feedback_used_for_fit"]:
        raise AssertionError("platform feedback may not be used for fitting")

    horizons = [int(value) for value in config["horizons_minutes"]]
    folds = list(config["validation"]["fold_ids"])
    if horizons != list(range(15, 1441, 15)):
        raise AssertionError("horizons must remain 15..1440 by 15 minutes")
    if len(folds) * len(horizons) != int(config["budget"]["expected_model_count"]):
        raise AssertionError("model budget mismatch")
    if float(config["training"]["denominator_floor_mw"]) != 20.0:
        raise AssertionError("v1 denominator floor must remain 20 MW")
    if not config["training"]["normalize_sample_weight_mean"]:
        raise AssertionError("sample weights must remain mean-one normalized")
    if config["lightgbm"]["objective"] != "regression_l1":
        raise AssertionError("base objective must remain regression_l1")

    paths = {
        key: PROJECT_ROOT / value
        for key, value in inputs.items()
        if key not in {"raw_train_tables", "control_oof_sha256"}
    }
    return {
        "raw_tables": raw_tables,
        "raw_manifest": raw_manifest,
        "raw_manifest_path": raw_manifest_path,
        "label_manifest": label_manifest,
        "label_manifest_path": label_manifest_path,
        "label_path": label_path,
        "split_manifest": split_manifest,
        "split_manifest_path": split_manifest_path,
        "split_path": split_path,
        "control_path": control_path,
        "event_path": paths["event_episodes"],
        "legacy_source": paths["legacy_v28_source"],
        "folds": folds,
        "horizons": horizons,
    }


def _metric_value(
    metrics: pd.DataFrame,
    variant: str,
    *,
    label: str,
    period: str,
    scope: str,
    fold: str = "all",
    bucket: str = "all",
    regime: str = "all",
) -> float:
    selected = metrics.loc[
        (metrics["variant"] == variant)
        & (metrics["label_variant"] == label)
        & (metrics["period"] == period)
        & (metrics["scope"] == scope)
        & (metrics["fold_id"].astype(str) == str(fold))
        & (metrics["horizon_bucket"].astype(str) == str(bucket))
        & (metrics["origin_regime"].astype(str) == str(regime))
    ]
    if len(selected) != 1:
        raise AssertionError(f"metric lookup returned {len(selected)} rows")
    return float(selected.iloc[0]["accuracy_1_minus_mape"])


def _evaluate_gate(metrics: pd.DataFrame, config: dict[str, Any]) -> dict[str, object]:
    control = "mae_control"
    candidate = config["experiment"]["candidate_variant"]
    label = "official_interval_mean"

    def gain(**kwargs: str) -> float:
        return (
            _metric_value(metrics, candidate, label=label, **kwargs)
            - _metric_value(metrics, control, label=label, **kwargs)
        ) * 100.0

    overall = gain(period="long", scope="overall")
    short = gain(period="short", scope="overall")
    folds = {
        fold: gain(period="long", scope="fold", fold=fold)
        for fold in config["validation"]["fold_ids"]
    }
    buckets = {
        bucket: gain(period="long", scope="horizon_bucket", bucket=bucket)
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440")
    }
    ordinary = gain(period="long", scope="origin_regime", regime="ordinary")
    legacy = (
        _metric_value(
            metrics,
            candidate,
            label="legacy_point",
            period="long",
            scope="overall",
        )
        - _metric_value(
            metrics,
            control,
            label="legacy_point",
            period="long",
            scope="overall",
        )
    ) * 100.0

    episodes = metrics.loc[
        (metrics["label_variant"] == label)
        & (metrics["period"] == "long")
        & (metrics["scope"] == "episode")
        & (metrics["variant"].isin([control, candidate])),
        ["variant", "episode_id", "accuracy_1_minus_mape"],
    ].pivot(index="episode_id", columns="variant", values="accuracy_1_minus_mape")
    episode_fraction = float((episodes[candidate] > episodes[control]).mean())

    gate = config["selection_gate"]
    pass_component = overall >= float(gate["minimum_component_long_g1_accuracy_gain_pct"])
    pass_folds = min(folds.values()) >= float(gate["minimum_single_fold_gain_pct"])
    pass_ordinary = ordinary >= -float(gate["maximum_ordinary_origin_accuracy_loss_pct"])
    pass_near_far = min(buckets["h015_120"], buckets["h735_1440"]) >= -float(
        gate["maximum_near_far_accuracy_loss_pct"]
    )
    pass_short = short >= -float(gate["maximum_short_accuracy_loss_pct"])
    pass_episodes = episode_fraction > 0.5
    component_gate = bool(
        pass_component
        and pass_folds
        and pass_ordinary
        and pass_near_far
        and pass_short
        and pass_episodes
    )
    standalone = bool(
        component_gate
        and overall >= float(gate["minimum_standalone_long_g1_accuracy_gain_pct"])
    )
    return {
        "variant": candidate,
        "official_long_gain_pct": overall,
        "official_short_gain_pct": short,
        "legacy_point_long_gain_pct": legacy,
        "single_fold_min_gain_pct": min(folds.values()),
        "single_fold_mean_gain_pct": float(np.mean(list(folds.values()))),
        **{f"{key}_gain_pct": value for key, value in folds.items()},
        "ordinary_gain_pct": ordinary,
        "episode_improvement_fraction": episode_fraction,
        "evaluated_episode_count": len(episodes),
        "near_gain_pct": buckets["h015_120"],
        "mid1_gain_pct": buckets["h135_360"],
        "mid2_gain_pct": buckets["h375_720"],
        "far_gain_pct": buckets["h735_1440"],
        "pass_component_gain": pass_component,
        "pass_folds": pass_folds,
        "pass_ordinary": pass_ordinary,
        "pass_episodes": pass_episodes,
        "pass_near_far": pass_near_far,
        "pass_short": pass_short,
        "component_gate_passed": component_gate,
        "standalone_submission_eligible": standalone,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = _validate_inputs(config)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"folds={len(validated['folds'])} horizons={len(validated['horizons'])} "
            f"models={config['budget']['expected_model_count']} control_hash=true "
            "candidate=mape_weighted_tree create_submission=false",
            flush=True,
        )
        return

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    identity_inputs = [
        *validated["raw_tables"],
        validated["raw_manifest_path"],
        validated["label_manifest_path"],
        validated["label_path"],
        validated["split_manifest_path"],
        validated["split_path"],
        validated["control_path"],
        validated["event_path"],
        validated["legacy_source"],
        RUNNER_SOURCE,
        UTILS_SOURCE,
        ANOMALY_UTILS_SOURCE,
        EVENT_UTILS_SOURCE,
        CONTROL_RUNNER_SOURCE,
    ]
    run_id = _run_identity(config_path, identity_inputs)
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
        "experiment_version": config["experiment_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started.isoformat(),
        "ended_at": None,
        "duration_seconds": None,
        "holdout_evaluated": False,
        "submission_eligible": False,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "environment": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in [config_path, *identity_inputs]
        },
        "failure": None,
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False

    try:
        raw = merge_training_tables(validated["raw_tables"])
        grid = to_v28_grid(raw)
        features, feature_columns = build_v28_features(grid)
        timestamps = pd.DatetimeIndex(grid["datetime"])
        position = pd.Series(np.arange(len(grid), dtype=int), index=timestamps)
        labels = pd.read_csv(validated["label_path"], parse_dates=["datetime"])
        interval_mean = pd.to_numeric(
            labels.set_index("datetime")["generator_1"], errors="coerce"
        )
        split = pd.read_csv(validated["split_path"])
        split["train_end"] = pd.to_datetime(split["train_end"], errors="raise")
        split["validation_origin"] = pd.to_datetime(
            split["validation_origin"], errors="raise"
        )
        split = split.loc[
            (split["role"] == "model_selection")
            & split["fold_id"].isin(validated["folds"])
        ].copy()
        if split["fold_id"].drop_duplicates().tolist() != validated["folds"]:
            raise AssertionError("validation fold order differs from frozen config")

        training_start = pd.Timestamp(config["training"]["g1_tree_start"])
        target_values = grid["generator_1"].to_numpy(dtype=float)
        model_parameters = dict(config["lightgbm"])
        candidate_frames: list[pd.DataFrame] = []
        fit_records: list[dict[str, object]] = []
        fold_groups = list(split.groupby("fold_id", sort=False))
        for fold_number, (fold_id, fold) in enumerate(fold_groups, start=1):
            train_end = pd.Timestamp(fold["train_end"].iloc[0])
            cutoff_position = int(np.flatnonzero(timestamps <= train_end)[-1])
            validation_origins = pd.DatetimeIndex(fold["validation_origin"]).sort_values()
            validation_positions = position.loc[validation_origins].to_numpy(dtype=int)
            fuel_training = grid.loc[
                grid["datetime"].between(training_start, train_end)
            ].copy()
            fuel_coefficients = fit_linear_fuel_proxy(
                fuel_training, "generator_1", FUELS_G1
            )
            fuel_validation = predict_linear_fuel(
                grid.iloc[validation_positions], FUELS_G1, fuel_coefficients
            )
            print(
                f"[{fold_number}/{len(fold_groups)}] {fold_id}: "
                f"validation_origins={len(validation_origins)}",
                flush=True,
            )
            for horizon_number, horizon in enumerate(validated["horizons"], start=1):
                step = horizon // 15
                eligible = np.flatnonzero(
                    (timestamps >= training_start)
                    & (np.arange(len(grid)) + step <= cutoff_position)
                )
                y_train = target_values[eligible + step]
                valid = np.isfinite(y_train)
                eligible = eligible[valid]
                y_train = y_train[valid]
                if len(y_train) < int(config["training"]["minimum_rows_per_model"]):
                    raise AssertionError(f"{fold_id} h={horizon}: too few training rows")
                sample_weight = normalized_mape_weights(
                    y_train,
                    denominator_floor=float(
                        config["training"]["denominator_floor_mw"]
                    ),
                )
                model = lgb.LGBMRegressor(**model_parameters)
                model.fit(
                    features.iloc[eligible][feature_columns],
                    y_train,
                    sample_weight=sample_weight,
                )
                tree_prediction = np.clip(
                    model.predict(features.iloc[validation_positions][feature_columns]),
                    0.0,
                    None,
                )
                fuel_weight = v28_g1_fuel_weight(horizon)
                prediction = (
                    (1.0 - fuel_weight) * tree_prediction
                    + fuel_weight * fuel_validation
                )
                target_positions = validation_positions + step
                interval_starts = validation_origins + pd.to_timedelta(
                    horizon - 15, unit="min"
                )
                candidate_frames.append(
                    pd.DataFrame(
                        {
                            "variant": config["experiment"]["candidate_variant"],
                            "fold_id": fold_id,
                            "datetime": validation_origins,
                            "interval_start": interval_starts,
                            "horizon_minutes": horizon,
                            "actual_interval_mean": interval_mean.reindex(
                                interval_starts
                            ).to_numpy(dtype=float),
                            "actual_legacy_point": target_values[target_positions],
                            "prediction": prediction,
                        }
                    )
                )
                fit_records.append(
                    {
                        "fold_id": fold_id,
                        "horizon_minutes": horizon,
                        "training_rows": len(y_train),
                        "weight_min": float(sample_weight.min()),
                        "weight_mean": float(sample_weight.mean()),
                        "weight_max": float(sample_weight.max()),
                        "fuel_weight": fuel_weight,
                    }
                )
                if horizon_number % 16 == 0 or horizon_number == len(
                    validated["horizons"]
                ):
                    print(
                        f"  horizons={horizon_number}/{len(validated['horizons'])} "
                        f"latest_h={horizon}",
                        flush=True,
                    )
            _append_event(event_log, "fold_completed", fold_id=fold_id)

        candidate = pd.concat(candidate_frames, ignore_index=True)
        control = pd.read_csv(
            validated["control_path"],
            parse_dates=["datetime", "interval_start"],
            low_memory=False,
        )
        base_columns = [
            "variant",
            "fold_id",
            "datetime",
            "interval_start",
            "horizon_minutes",
            "actual_interval_mean",
            "actual_legacy_point",
            "prediction",
        ]
        control = control.loc[
            control["variant"].eq(config["experiment"]["control_variant"])
            & control["fold_id"].isin(validated["folds"]),
            base_columns,
        ].copy()
        control["variant"] = "mae_control"
        key_columns = ["fold_id", "datetime", "interval_start", "horizon_minutes"]
        paired = control.merge(
            candidate,
            on=key_columns,
            suffixes=("_control", "_candidate"),
            validate="one_to_one",
        )
        if len(paired) != len(control) or len(paired) != len(candidate):
            raise AssertionError("candidate/control OOF keys differ")
        for actual in ("actual_interval_mean", "actual_legacy_point"):
            if not np.allclose(
                paired[f"{actual}_control"],
                paired[f"{actual}_candidate"],
                equal_nan=True,
            ):
                raise AssertionError(f"candidate/control {actual} differs")

        oof = pd.concat([control, candidate], ignore_index=True)
        episodes = pd.read_csv(validated["event_path"])
        oof = attach_event_context(oof, episodes, context_hours=24)
        metrics = _summarize_metrics(oof)
        gate_result = _evaluate_gate(metrics, config)
        gate = pd.DataFrame([gate_result])

        overall = metrics.loc[
            (metrics["label_variant"] == "official_interval_mean")
            & (metrics["scope"] == "overall")
        ].copy()
        control_accuracy = overall.loc[
            overall["variant"].eq("mae_control"),
            ["period", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
        comparison = overall.merge(control_accuracy, on="period", validate="many_to_one")
        comparison["delta_accuracy_pct"] = (
            comparison["accuracy_1_minus_mape"] - comparison["control_accuracy"]
        ) * 100.0
        comparison = comparison[
            [
                "period",
                "variant",
                "mape",
                "accuracy_1_minus_mape",
                "delta_accuracy_pct",
            ]
        ].sort_values(["period", "mape"])

        oof_path = output_dir / "oof_predictions.csv"
        fit_path = output_dir / "fit_summary.csv"
        metrics_path = output_dir / "metrics.csv"
        comparison_path = output_dir / "variant_comparison.csv"
        gate_path = output_dir / "gate_results.csv"
        oof.to_csv(oof_path, index=False, date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
        pd.DataFrame(fit_records).to_csv(fit_path, index=False, float_format="%.10f")
        metrics.to_csv(metrics_path, index=False, float_format="%.10f")
        comparison.to_csv(comparison_path, index=False, float_format="%.10f")
        gate.to_csv(gate_path, index=False, float_format="%.10f")

        ended = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended.isoformat(),
                "duration_seconds": round(time.perf_counter() - clock, 3),
                "model_count": int(config["budget"]["expected_model_count"]),
                "feature_count": len(feature_columns),
                "component_gate_passed": bool(gate_result["component_gate_passed"]),
                "standalone_submission_eligible": bool(
                    gate_result["standalone_submission_eligible"]
                ),
                "artifacts": {
                    name: path.relative_to(PROJECT_ROOT).as_posix()
                    for name, path in {
                        "oof_predictions": oof_path,
                        "fit_summary": fit_path,
                        "metrics": metrics_path,
                        "variant_comparison": comparison_path,
                        "gate_results": gate_path,
                    }.items()
                },
            }
        )
        manifest["artifact_sha256"] = {
            name: sha256(PROJECT_ROOT / path)
            for name, path in manifest["artifacts"].items()
        }
        write_json(output_dir / "run_manifest.json", manifest)
        candidate_long = comparison.loc[
            (comparison["period"] == "long")
            & (comparison["variant"] == config["experiment"]["candidate_variant"])
        ].iloc[0]
        if config["output"]["append_registry"]:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "split_sha256": validated["split_manifest"]["assignment_sha256"],
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "v28_g1_inverse_target_weighted_l1",
                    "long_g1_mape": float(candidate_long["mape"]),
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": (
                        f"component_gate_passed={gate_result['component_gate_passed']}; "
                        f"standalone_submission_eligible={gate_result['standalone_submission_eligible']}"
                    ),
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", **gate_result)
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gate.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} "
            f"component_gate_passed={gate_result['component_gate_passed']} "
            f"standalone_submission_eligible={gate_result['standalone_submission_eligible']}",
            flush=True,
        )
    except Exception as exc:
        ended = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "failed",
                "ended_at": ended.isoformat(),
                "duration_seconds": round(time.perf_counter() - clock, 3),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", failure_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "label_version": validated["label_manifest"]["artifact_version"],
                    "split_sha256": validated["split_manifest"]["assignment_sha256"],
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "v28_g1_inverse_target_weighted_l1",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started.isoformat(),
                    "ended_at": ended.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()

