"""Run the pre-registered long-g1 process-feature ablation.

The run is scientific OOF evidence only.  It never builds a submission and it
does not claim equivalence to the external v29b model.
"""

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
    select_feature_columns,
)
from src.round2_v3.event_ablation_utils import (
    attach_event_context,
    gate_results,
    summarize_ablation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "event_signal_ablation_v1.yaml"
)
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "round2_v3" / "registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("event_ablation_utils.py")
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
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_event_signal_ablation_{digest.hexdigest()[:10]}"


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


def _partition(manifest: dict[str, Any], partition: str) -> dict[str, Any]:
    matches = [item for item in manifest["partitions"] if item["partition"] == partition]
    if len(matches) != 1:
        raise AssertionError(f"manifest has {len(matches)} {partition} partitions")
    return matches[0]


def _validate_inputs(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    baseline_config = yaml.safe_load(paths["baseline_config"].read_text(encoding="utf-8"))
    base_manifest = _read_json(paths["base_feature_manifest"])
    event_manifest = _read_json(paths["event_feature_manifest"])
    label_manifest = _read_json(paths["label_manifest"])
    split_manifest = _read_json(paths["split_manifest"])

    base_train = PROJECT_ROOT / _partition(base_manifest, "train")["output"]
    event_train = PROJECT_ROOT / _partition(event_manifest, "train")["output"]
    label_path = PROJECT_ROOT / label_manifest["output"]
    if sha256(base_train) != _partition(base_manifest, "train")["output_sha256"]:
        raise AssertionError("base feature train artifact hash mismatch")
    if sha256(event_train) != _partition(event_manifest, "train")["output_sha256"]:
        raise AssertionError("event feature train artifact hash mismatch")
    if sha256(label_path) != label_manifest["output_sha256"]:
        raise AssertionError("label artifact hash mismatch")
    if sha256(paths["base_feature_registry"]) != base_manifest["registry_sha256"]:
        raise AssertionError("base feature registry hash mismatch")
    if sha256(paths["event_feature_registry"]) != event_manifest["registry_sha256"]:
        raise AssertionError("event feature registry hash mismatch")
    if sha256(paths["split_assignments"]) != split_manifest["assignment_sha256"]:
        raise AssertionError("split assignment hash mismatch")
    if event_manifest["maximum_source_time_offset_minutes"] > 0:
        raise AssertionError("event process features use future observations")
    if event_manifest["target_history_included"]:
        raise AssertionError("event process artifact contains target history")
    if config["frozen_scope"]["true_target_history_allowed"]:
        raise AssertionError("formal test target history must remain forbidden")
    if config["frozen_scope"]["predicted_target_feedback_allowed"]:
        raise AssertionError("first ablation must not add predicted feedback")
    if config["frozen_scope"]["short_outputs_changed"]:
        raise AssertionError("short output must remain frozen")
    if config["frozen_scope"]["generator_all_changed"]:
        raise AssertionError("generator_all must remain frozen")
    episodes = pd.read_csv(paths["event_episodes"])
    if len(episodes) != int(config["event_protocol"]["independent_episode_count"]):
        raise AssertionError("event episode count differs from the pre-registration")

    event_registry = pd.read_csv(paths["event_feature_registry"])
    if not event_registry["causal"].astype(bool).all():
        raise AssertionError("event feature registry contains a non-causal feature")
    if event_registry["target_history"].astype(bool).any():
        raise AssertionError("event feature registry contains target history")
    declared = [
        feature
        for features in config["feature_groups"].values()
        for feature in features
    ]
    if len(declared) != len(set(declared)):
        raise AssertionError("feature groups overlap; attribution would be ambiguous")
    missing = set(declared) - set(event_registry["feature_name"])
    if missing:
        raise KeyError(f"declared event features absent from registry: {sorted(missing)}")

    expected_variants = ["control"] + [
        f"control_plus_{name}" for name in config["feature_groups"]
    ] + ["control_plus_compact_union"]
    if config["comparison_order"] != expected_variants:
        raise AssertionError("comparison_order differs from declared feature groups")
    if len(expected_variants) - 1 > int(config["budget"]["maximum_feature_group_comparisons"]):
        raise AssertionError("configured comparison budget exceeded")
    return {
        "baseline_config": baseline_config,
        "base_manifest": base_manifest,
        "event_manifest": event_manifest,
        "label_manifest": label_manifest,
        "split_manifest": split_manifest,
        "base_train": base_train,
        "event_train": event_train,
        "label_path": label_path,
        "episodes": episodes,
        "declared_features": declared,
        "expected_variants": expected_variants,
    }


def _prediction_part(
    meta: pd.DataFrame,
    actual: np.ndarray,
    prediction: np.ndarray,
    variant: str,
    fold_id: str,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    result = meta.copy()
    result["actual"] = actual
    result["prediction"] = prediction
    result["variant"] = variant
    result["fold_id"] = fold_id
    return attach_event_context(result, episodes, context_hours=24)[
        [
            "variant",
            "fold_id",
            "datetime",
            "interval_start",
            "horizon_minutes",
            "actual",
            "prediction",
            "origin_regime",
            "origin_episode_id",
            "target_in_event",
            "target_episode_id",
            "horizon_bucket",
        ]
    ]


def _write_report(
    path: Path,
    run_id: str,
    comparison: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    lines = [
        "# Long-g1 过程特征组消融",
        "",
        f"- run_id: `{run_id}`",
        "- 角色：scientific OOF；不是 v29b 复现，也不是平台提交成绩。",
        "",
        "## 总体结果",
        "",
        "| variant | MAPE | accuracy | delta accuracy (pct) |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.mape:.6f} | "
            f"{row.accuracy_1_minus_mape:.6f} | {row.delta_accuracy_pct:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 预登记门禁",
            "",
            "| variant | overall gain | ordinary loss | recent gain | episode improved | near | far | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gates.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {row.overall_accuracy_gain_pct:.4f} | "
            f"{row.ordinary_accuracy_loss_pct:.4f} | "
            f"{row.recent_fold_mean_gain_pct:.4f} | "
            f"{row.episode_improvement_fraction:.1%} | "
            f"{row.near_gain_pct:.4f} | {row.far_gain_pct:.4f} | "
            f"{row.promotion_gate_passed} |"
        )
    lines.extend(
        [
            "",
            "通过本地门禁也不自动生成提交。只有先把通过的单一 feature group 集成到冻结 v29b 的 long-g1，",
            "重新完成来源、差异和安全检查后，才可以成为平台候选。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {key: PROJECT_ROOT / value for key, value in config["inputs"].items()}
    validated = _validate_inputs(config, paths)
    if args.preflight_only:
        split_assignments = pd.read_csv(paths["split_assignments"])
        fold_count = int((split_assignments["role"] == "model_selection").groupby(
            split_assignments["fold_id"]
        ).any().sum())
        variant_count = len(validated["expected_variants"])
        print(
            "PASS preflight "
            f"folds={fold_count} variants={variant_count} "
            f"models={fold_count * variant_count} "
            f"event_features={len(validated['declared_features'])} "
            f"episodes={len(validated['episodes'])}"
        )
        return
    identity_inputs = [
        *paths.values(),
        RUNNER_SOURCE,
        UTILS_SOURCE,
        BASELINE_UTILS_SOURCE,
        METRICS_SOURCE,
    ]
    run_id = _run_identity(config_path, identity_inputs)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")
    (output_dir / "models").mkdir(parents=True, exist_ok=False)
    event_log = output_dir / "events.jsonl"

    git_commit = _git_value("rev-parse", "HEAD")
    git_branch = _git_value("branch", "--show-current")
    git_dirty = bool(_git_value("status", "--porcelain"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": config["task"],
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "experiment_version": config["experiment_version"],
        "control_run_id": config["control_run_id"],
        "external_reference_artifact_id": config["external_reference_artifact_id"],
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
        "holdout_evaluated": False,
        "submission_eligible": False,
        "external_v29b_oof_available": False,
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in identity_inputs
        },
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False

    try:
        base_features = pd.read_csv(validated["base_train"], low_memory=False)
        event_features = pd.read_csv(validated["event_train"], low_memory=False)
        for frame in (base_features, event_features):
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        origin_features = base_features.merge(
            event_features, on="datetime", how="inner", validate="one_to_one"
        )
        if len(origin_features) != len(base_features):
            raise AssertionError("base/event feature origin grids differ")

        interval_targets = pd.read_csv(validated["label_path"], low_memory=False)
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
        split_assignments = split_assignments.loc[
            split_assignments["role"] == "model_selection"
        ].copy()

        base_registry = pd.read_csv(paths["base_feature_registry"])
        baseline_config = validated["baseline_config"]
        base_columns = select_feature_columns(
            base_registry, baseline_config["feature_selection"]
        )
        union_features = list(dict.fromkeys(validated["declared_features"]))
        variants: dict[str, list[str]] = {"control": []}
        for group, features in config["feature_groups"].items():
            variants[f"control_plus_{group}"] = list(features)
        variants["control_plus_compact_union"] = union_features
        if list(variants) != config["comparison_order"]:
            raise AssertionError("variant construction differs from pre-registration")
        (output_dir / "variant_features.json").write_text(
            json.dumps(variants, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        horizons = list(config["frozen_scope"]["horizons_minutes"])
        if horizons != baseline_config["periods"]["long"]["horizons_minutes"]:
            raise AssertionError("long horizons differ from the frozen baseline")
        stride = int(
            baseline_config["periods"]["long"]["training_origin_stride_minutes"]
        )
        model_config = dict(baseline_config["lightgbm"])
        weight_policy = model_config.pop("sample_weight")
        weight_floor = float(model_config.pop("sample_weight_floor"))
        if weight_policy != "inverse_absolute_target":
            raise ValueError(f"unsupported sample weight policy: {weight_policy}")

        episodes = validated["episodes"].copy()
        predictions: list[pd.DataFrame] = []
        all_model_columns = [*base_columns, *union_features]
        fold_count = split_assignments["fold_id"].nunique()
        for fold_number, (fold_id, fold) in enumerate(
            split_assignments.groupby("fold_id", sort=False), start=1
        ):
            latest_train_origin = fold["latest_eligible_train_origin"].iloc[0]
            validation_origins = pd.DatetimeIndex(fold["validation_origin"])
            eligible_train_origins = pd.DatetimeIndex(
                origin_features.loc[
                    origin_features["datetime"] <= latest_train_origin, "datetime"
                ]
            )
            training_origins = filter_origins_by_stride(
                eligible_train_origins, stride
            )
            x_train, y_train, _ = build_supervised_long(
                origin_features,
                interval_targets,
                training_origins,
                horizons,
                "generator_1",
                all_model_columns,
            )
            x_validation, y_validation, validation_meta = build_supervised_long(
                origin_features,
                interval_targets,
                validation_origins,
                horizons,
                "generator_1",
                all_model_columns,
            )
            expected_rows = len(validation_origins) * len(horizons)
            if len(y_validation) != expected_rows:
                raise AssertionError(
                    f"incomplete validation labels for {fold_id}: {len(y_validation)} != {expected_rows}"
                )
            weights = 1.0 / np.maximum(np.abs(y_train), weight_floor)
            weights = weights / weights.mean()
            print(
                f"[{fold_number}/{fold_count}] {fold_id}: "
                f"train_rows={len(y_train)} validation_rows={len(y_validation)}",
                flush=True,
            )
            for variant_index, (variant, extra_features) in enumerate(
                variants.items(), start=1
            ):
                columns = [*base_columns, *extra_features, *TARGET_CALENDAR_COLUMNS]
                model = lgb.LGBMRegressor(**model_config)
                model.fit(x_train[columns], y_train, sample_weight=weights)
                prediction = model.predict(x_validation[columns])
                predictions.append(
                    _prediction_part(
                        validation_meta,
                        y_validation,
                        prediction,
                        variant,
                        fold_id,
                        episodes,
                    )
                )
                if config["output"]["save_models"]:
                    model.booster_.save_model(
                        str(output_dir / "models" / f"{fold_id}__{variant}.txt")
                    )
                print(
                    f"  [{variant_index}/{len(variants)}] done {variant} "
                    f"features={len(columns)}",
                    flush=True,
                )
                _append_event(
                    event_log,
                    "model_completed",
                    fold_id=fold_id,
                    variant=variant,
                    train_rows=len(y_train),
                    validation_rows=len(y_validation),
                    feature_count=len(columns),
                )
                del model
                gc.collect()
            del x_train, x_validation, y_train, y_validation
            gc.collect()

        oof = pd.concat(predictions, ignore_index=True)
        oof_path = output_dir / "oof_predictions.csv"
        if config["output"]["save_oof_predictions"]:
            oof.to_csv(
                oof_path,
                index=False,
                encoding="utf-8",
                date_format="%Y-%m-%d %H:%M:%S",
                float_format="%.10f",
            )
        metrics = summarize_ablation(oof)
        metrics_path = output_dir / "metrics.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")

        overall = metrics.loc[metrics["scope"] == "overall"].copy()
        control_accuracy = float(
            overall.loc[
                overall["variant"] == "control", "accuracy_1_minus_mape"
            ].iloc[0]
        )
        overall["delta_accuracy_pct"] = (
            overall["accuracy_1_minus_mape"] - control_accuracy
        ) * 100
        comparison = overall[
            ["variant", "mape", "accuracy_1_minus_mape", "delta_accuracy_pct"]
        ].sort_values("mape")
        comparison_path = output_dir / "variant_comparison.csv"
        comparison.to_csv(
            comparison_path, index=False, encoding="utf-8", float_format="%.10f"
        )

        recent_folds = ["wf_03_bf3_active", "wf_04_late_september", "wf_05_holder1_active"]
        gates = gate_results(
            metrics,
            "control",
            [name for name in variants if name != "control"],
            config["selection_gate"],
            recent_folds,
        )
        gates_path = output_dir / "gate_results.csv"
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        report_path = output_dir / "report.md"
        _write_report(report_path, run_id, comparison, gates)

        best = comparison.iloc[0]
        passed = gates.loc[gates["promotion_gate_passed"], "variant"].tolist()
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "variants": variants,
                "model_parameters": model_config,
                "selected_by_overall_oof_only": str(best["variant"]),
                "selected_overall_mape": float(best["mape"]),
                "promotion_gate_passed_variants": passed,
                "submission_eligible": False,
                "artifacts": {
                    "oof_predictions": oof_path.relative_to(PROJECT_ROOT).as_posix(),
                    "oof_predictions_sha256": sha256(oof_path),
                    "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                    "metrics_sha256": sha256(metrics_path),
                    "variant_comparison": comparison_path.relative_to(PROJECT_ROOT).as_posix(),
                    "variant_comparison_sha256": sha256(comparison_path),
                    "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(),
                    "gate_results_sha256": sha256(gates_path),
                    "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                    "report_sha256": sha256(report_path),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        registry_row = {
            "run_id": run_id,
            "status": "completed",
            "experiment_role": config["experiment_role"],
            "task": config["task"],
            "protocol_version": "round2_v3",
            "label_version": validated["label_manifest"]["artifact_version"],
            "split_sha256": validated["split_manifest"]["assignment_sha256"],
            "control_run_id": config["control_run_id"],
            "hypothesis": config["hypothesis"],
            "primary_change": "five pre-registered causal process feature groups for long generator_1",
            "model_name": "lightgbm_shared_horizon_event_feature_ablation",
            "long_g1_mape": float(best["mape"]),
            "holdout_evaluated": False,
            "submission_eligible": False,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": manifest["duration_seconds"],
            "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "notes": (
                f"best_oof={best['variant']}; promotion_passed={passed}; "
                "scientific comparison only, not v29b OOF"
            ),
        }
        if config["output"]["append_registry"]:
            _append_registry(registry_row)
            registry_written = True
        _append_event(
            event_log,
            "run_completed",
            duration_seconds=manifest["duration_seconds"],
            best_variant=str(best["variant"]),
            passed_variants=passed,
        )
        print("\n" + comparison.to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} promotion_passed={passed}",
            flush=True,
        )
    except Exception as exc:
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "failed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(
            event_log,
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
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "control_run_id": config["control_run_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": "five pre-registered causal process feature groups for long generator_1",
                    "model_name": "lightgbm_shared_horizon_event_feature_ablation",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": git_commit,
                    "git_dirty": git_dirty,
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
