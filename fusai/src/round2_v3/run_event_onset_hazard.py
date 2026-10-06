"""Diagnose whether causal process history predicts held-out episode onsets."""

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

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io_utils import sha256, write_json
from src.round2_v3.event_hazard_utils import (
    calendar_features,
    episode_detection_rows,
    onset_within_horizon,
    quantile_alert_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "experiments"
    / "event_onset_hazard_v1.yaml"
)
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "round2_v3" / "registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
UTILS_SOURCE = RUNNER_SOURCE.with_name("event_hazard_utils.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _partition(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in manifest["partitions"] if item["partition"] == name]
    if len(matches) != 1:
        raise AssertionError(f"manifest has {len(matches)} {name} partitions")
    return matches[0]


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
    return f"{stamp}_event_onset_hazard_{digest.hexdigest()[:10]}"


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


def _validate_inputs(
    config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    manifest = _read_json(paths["event_feature_manifest"])
    if not manifest["all_features_causal"]:
        raise AssertionError("event features are not all causal")
    if manifest["future_observations_included"]:
        raise AssertionError("event features contain future observations")
    if manifest["target_history_included"]:
        raise AssertionError("event features contain target history")
    if int(manifest["maximum_source_time_offset_minutes"]) > 0:
        raise AssertionError("event feature source offset exceeds origin")
    train_feature_path = PROJECT_ROOT / _partition(manifest, "train")["output"]
    if sha256(train_feature_path) != _partition(manifest, "train")["output_sha256"]:
        raise AssertionError("event feature train hash mismatch")
    if sha256(paths["event_feature_registry"]) != manifest["registry_sha256"]:
        raise AssertionError("event feature registry hash mismatch")

    registry = pd.read_csv(paths["event_feature_registry"])
    process_columns = list(config["feature_groups"]["compact_process"])
    selected = registry.loc[registry["feature_name"].isin(process_columns)].copy()
    if set(selected["feature_name"]) != set(process_columns):
        missing = set(process_columns) - set(selected["feature_name"])
        raise AssertionError(f"configured process features missing: {sorted(missing)}")
    if not selected["causal"].astype(bool).all():
        raise AssertionError("selected process features are not causal")
    if not selected["available_in_formal_test"].astype(bool).all():
        raise AssertionError("selected process features are unavailable in test")
    if (pd.to_numeric(selected["source_offset_max_minutes"]) > 0).any():
        raise AssertionError("selected process feature has positive source offset")

    episodes = pd.read_csv(paths["event_episodes"])
    episodes["episode_start"] = pd.to_datetime(
        episodes["episode_start"], errors="raise"
    )
    if episodes["episode_id"].duplicated().any():
        raise AssertionError("episode ids are not unique")
    configured_ids = [
        episode_id
        for fold in config["episode_folds"]
        for episode_id in fold["validation_episode_ids"]
    ]
    if len(configured_ids) != len(set(configured_ids)):
        raise AssertionError("validation episodes overlap across folds")
    if not set(configured_ids) <= set(episodes["episode_id"]):
        raise AssertionError("configured fold references unknown episode")

    if config["models"]["broad_model_search"]:
        raise AssertionError("broad model search is forbidden")
    horizons = [int(value) for value in config["horizons_minutes"]]
    if horizons != [120, 360, 720, 1440]:
        raise AssertionError("hazard horizon set differs from preregistration")
    expected_models = (
        len(config["episode_folds"])
        * len(horizons)
        * int(config["budget"]["model_count_per_fold_horizon"])
    )
    if expected_models != int(config["budget"]["expected_model_count"]):
        raise AssertionError("hazard model budget is inconsistent")
    if float(config["alert_policy"]["training_probability_top_fraction"]) != 0.10:
        raise AssertionError("alert budget must remain fixed at top 10 percent")
    return {
        "manifest": manifest,
        "train_feature_path": train_feature_path,
        "registry": registry,
        "episodes": episodes,
        "process_columns": process_columns,
        "horizons": horizons,
    }


def _binary_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    alert: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=int)
    p = np.asarray(probability, dtype=float)
    flagged = np.asarray(alert, dtype=bool)
    prevalence = float(y.mean())
    true_positive = int(((y == 1) & flagged).sum())
    false_positive = int(((y == 0) & flagged).sum())
    positive = int((y == 1).sum())
    alerts = int(flagged.sum())
    precision = float(true_positive / alerts) if alerts else 0.0
    recall = float(true_positive / positive) if positive else np.nan
    average_precision = float(average_precision_score(y, p))
    return {
        "rows": int(len(y)),
        "positives": positive,
        "prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift": float(average_precision / prevalence)
        if prevalence > 0
        else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "alert_rows": alerts,
        "alert_fraction": float(alerts / len(y)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
        "precision_lift": float(precision / prevalence) if prevalence > 0 else np.nan,
        "recall": recall,
    }


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, horizon), part in predictions.groupby(
        ["model", "horizon_minutes"], sort=False
    ):
        rows.append(
            {
                "scope": "pooled",
                "fold_id": "all",
                "model": model,
                "horizon_minutes": int(horizon),
                **_binary_metrics(
                    part["actual"].to_numpy(),
                    part["probability"].to_numpy(),
                    part["alert"].to_numpy(),
                ),
            }
        )
    for (fold_id, model, horizon), part in predictions.groupby(
        ["fold_id", "model", "horizon_minutes"], sort=False
    ):
        rows.append(
            {
                "scope": "fold",
                "fold_id": fold_id,
                "model": model,
                "horizon_minutes": int(horizon),
                **_binary_metrics(
                    part["actual"].to_numpy(),
                    part["probability"].to_numpy(),
                    part["alert"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _evaluate_gate(
    metrics: pd.DataFrame,
    detections: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    gate = config["selection_gate"]
    candidate = config["models"]["candidate"]
    control = config["models"]["control"]
    primary_horizon = int(gate["primary_horizon_minutes"])
    secondary_horizon = int(gate["secondary_horizon_minutes"])

    def pooled(model: str, horizon: int) -> pd.Series:
        selected = metrics.loc[
            (metrics["scope"] == "pooled")
            & (metrics["model"] == model)
            & (metrics["horizon_minutes"] == horizon)
        ]
        if len(selected) != 1:
            raise AssertionError(f"missing pooled metric for {model} h={horizon}")
        return selected.iloc[0]

    primary = pooled(candidate, primary_horizon)
    primary_control = pooled(control, primary_horizon)
    secondary = pooled(candidate, secondary_horizon)
    fold_metrics = metrics.loc[
        (metrics["scope"] == "fold")
        & (metrics["horizon_minutes"] == primary_horizon)
        & (metrics["model"].isin([candidate, control])),
        ["fold_id", "model", "average_precision"],
    ].pivot(index="fold_id", columns="model", values="average_precision")
    fold_win_fraction = float((fold_metrics[candidate] > fold_metrics[control]).mean())

    def detection_fraction(horizon: int) -> float:
        selected = detections.loc[
            (detections["model"] == candidate)
            & (detections["horizon_minutes"] == horizon)
        ]
        return float(selected["detected"].mean()) if len(selected) else np.nan

    primary_detection = detection_fraction(primary_horizon)
    secondary_detection = detection_fraction(secondary_horizon)
    checks = {
        "pass_primary_ap_lift": float(primary["average_precision_lift"])
        >= float(gate["minimum_primary_average_precision_lift"]),
        "pass_primary_ap_gain": float(
            primary["average_precision"] - primary_control["average_precision"]
        )
        >= float(gate["minimum_primary_ap_gain_vs_calendar"]),
        "pass_primary_precision_lift": float(primary["precision_lift"])
        >= float(gate["minimum_primary_precision_lift"]),
        "pass_primary_recall": float(primary["recall"])
        >= float(gate["minimum_primary_recall"]),
        "pass_primary_episode_detection": primary_detection
        >= float(gate["minimum_primary_episode_detection_fraction"]),
        "pass_primary_fold_wins": fold_win_fraction
        >= float(gate["minimum_primary_fold_win_fraction"]),
        "pass_secondary_ap_lift": float(secondary["average_precision_lift"])
        >= float(gate["minimum_secondary_average_precision_lift"]),
        "pass_secondary_episode_detection": secondary_detection
        >= float(gate["minimum_secondary_episode_detection_fraction"]),
        "pass_alert_budget": float(primary["alert_fraction"])
        <= float(gate["maximum_validation_alert_fraction"]),
    }
    return pd.DataFrame.from_records(
        [
            {
                "candidate": candidate,
                "primary_horizon_minutes": primary_horizon,
                "primary_prevalence": float(primary["prevalence"]),
                "primary_average_precision": float(primary["average_precision"]),
                "primary_average_precision_lift": float(
                    primary["average_precision_lift"]
                ),
                "primary_ap_gain_vs_calendar": float(
                    primary["average_precision"]
                    - primary_control["average_precision"]
                ),
                "primary_precision": float(primary["precision"]),
                "primary_precision_lift": float(primary["precision_lift"]),
                "primary_recall": float(primary["recall"]),
                "primary_alert_fraction": float(primary["alert_fraction"]),
                "primary_episode_detection_fraction": primary_detection,
                "primary_fold_win_fraction": fold_win_fraction,
                "secondary_horizon_minutes": secondary_horizon,
                "secondary_average_precision_lift": float(
                    secondary["average_precision_lift"]
                ),
                "secondary_episode_detection_fraction": secondary_detection,
                **checks,
                "promotion_gate_passed": bool(all(checks.values())),
            }
        ]
    )


def _write_report(
    path: Path,
    run_id: str,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    pooled = metrics.loc[metrics["scope"] == "pooled"].sort_values(
        ["horizon_minutes", "model"]
    )
    lines = [
        "# Causal episode-onset hazard diagnostic",
        "",
        f"- run_id: `{run_id}`",
        "- 角色：diagnostic；不修改 v29b，不生成提交。",
        "",
        "## Pooled results",
        "",
        "| horizon | model | prevalence | AP | AP lift | precision | recall | alerts |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled.itertuples(index=False):
        lines.append(
            f"| {row.horizon_minutes} | `{row.model}` | {row.prevalence:.4f} | "
            f"{row.average_precision:.4f} | {row.average_precision_lift:.2f} | "
            f"{row.precision:.4f} | {row.recall:.4f} | {row.alert_fraction:.4f} |"
        )
    gate = gates.iloc[0]
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- 24h AP gain vs calendar: {gate.primary_ap_gain_vs_calendar:.4f}",
            f"- 24h episode detection: {gate.primary_episode_detection_fraction:.1%}",
            f"- 24h fold win fraction: {gate.primary_fold_win_fraction:.1%}",
            f"- 6h AP lift: {gate.secondary_average_precision_lift:.2f}",
            f"- promotion_gate_passed: `{gate.promotion_gate_passed}`",
            "",
            "通过只允许进入 fold-local residual correction；本 run 本身永远不是提交候选。",
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
        print(
            "PASS preflight "
            f"folds={len(config['episode_folds'])} "
            f"horizons={len(validated['horizons'])} models={config['budget']['expected_model_count']} "
            f"process_features={len(validated['process_columns'])} heldout_episodes=8"
        )
        return

    identity_inputs = [*paths.values(), validated["train_feature_path"], RUNNER_SOURCE, UTILS_SOURCE]
    run_id = _run_identity(config_path, identity_inputs)
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
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
                "scikit_learn": sklearn.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "holdout_evaluated": False,
        "submission_eligible": False,
        "fingerprints": {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256(path)
            for path in [config_path, *identity_inputs]
        },
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False

    try:
        features = pd.read_csv(validated["train_feature_path"], low_memory=False)
        features["datetime"] = pd.to_datetime(features["datetime"], errors="raise")
        if features["datetime"].duplicated().any():
            raise AssertionError("event feature origins are not unique")
        calendar = calendar_features(features["datetime"])
        calendar.index = features.index
        design = pd.concat([features, calendar], axis=1)
        calendar_columns = list(config["feature_groups"]["calendar_only"])
        process_columns = validated["process_columns"]
        model_columns = {
            config["models"]["control"]: calendar_columns,
            config["models"]["candidate"]: [*calendar_columns, *process_columns],
        }
        episodes = validated["episodes"].copy().sort_values("episode_start")
        episode_starts = pd.DatetimeIndex(episodes["episode_start"])
        logistic_config = dict(config["logistic_regression"])
        alert_budget = float(config["alert_policy"]["training_probability_top_fraction"])
        max_horizon = max(validated["horizons"])
        prediction_parts: list[pd.DataFrame] = []
        detection_parts: list[pd.DataFrame] = []
        coefficient_records: list[dict[str, object]] = []

        for fold_number, fold in enumerate(config["episode_folds"], start=1):
            fold_id = fold["fold_id"]
            heldout = episodes.loc[
                episodes["episode_id"].isin(fold["validation_episode_ids"])
            ].sort_values("episode_start")
            if heldout["episode_id"].tolist() != fold["validation_episode_ids"]:
                raise AssertionError(f"episode order differs in {fold_id}")
            first_start = pd.Timestamp(heldout["episode_start"].min())
            last_start = pd.Timestamp(heldout["episode_start"].max())
            validation_start = first_start - pd.Timedelta(minutes=max_horizon)
            validation_mask = (
                (features["datetime"] >= validation_start)
                & (features["datetime"] < last_start)
            )
            validation_rows = design.loc[validation_mask].copy()
            print(
                f"[{fold_number}/{len(config['episode_folds'])}] {fold_id}: "
                f"episodes={','.join(heldout['episode_id'])} validation_rows={len(validation_rows)}",
                flush=True,
            )
            for horizon in validated["horizons"]:
                latest_training_origin = validation_start - pd.Timedelta(minutes=horizon)
                training_mask = features["datetime"] <= latest_training_origin
                training_rows = design.loc[training_mask].copy()
                y_train = onset_within_horizon(
                    training_rows["datetime"], episode_starts, horizon
                )
                y_validation = onset_within_horizon(
                    validation_rows["datetime"], episode_starts, horizon
                )
                if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
                    raise AssertionError(
                        f"{fold_id} h={horizon} does not contain both classes"
                    )
                for model_name, columns in model_columns.items():
                    model = Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="median", keep_empty_features=True),
                            ),
                            ("scaler", StandardScaler()),
                            ("classifier", LogisticRegression(**logistic_config)),
                        ]
                    )
                    model.fit(training_rows[columns], y_train)
                    training_probability = model.predict_proba(training_rows[columns])[:, 1]
                    threshold = quantile_alert_threshold(training_probability, alert_budget)
                    probability = model.predict_proba(validation_rows[columns])[:, 1]
                    alert = probability >= threshold
                    part = pd.DataFrame(
                        {
                            "fold_id": fold_id,
                            "model": model_name,
                            "horizon_minutes": horizon,
                            "datetime": validation_rows["datetime"].to_numpy(),
                            "actual": y_validation,
                            "probability": probability,
                            "threshold": threshold,
                            "alert": alert,
                            "validation_start": validation_start,
                            "latest_training_origin": latest_training_origin,
                        }
                    )
                    prediction_parts.append(part)
                    detection_parts.append(
                        episode_detection_rows(part, heldout, horizon)
                    )
                    classifier = model.named_steps["classifier"]
                    for feature_name, coefficient in zip(
                        columns, classifier.coef_[0], strict=True
                    ):
                        coefficient_records.append(
                            {
                                "fold_id": fold_id,
                                "model": model_name,
                                "horizon_minutes": horizon,
                                "feature_name": feature_name,
                                "coefficient_standardized": float(coefficient),
                                "intercept": float(classifier.intercept_[0]),
                                "training_rows": int(len(training_rows)),
                                "training_positives": int(y_train.sum()),
                                "validation_rows": int(len(validation_rows)),
                                "validation_positives": int(y_validation.sum()),
                                "threshold": threshold,
                            }
                        )
            _append_event(
                event_log,
                "fold_completed",
                fold_id=fold_id,
                heldout_episode_ids=heldout["episode_id"].tolist(),
                validation_rows=len(validation_rows),
            )

        predictions = pd.concat(prediction_parts, ignore_index=True)
        detections = pd.concat(detection_parts, ignore_index=True)
        coefficients = pd.DataFrame.from_records(coefficient_records)
        metrics = _summarize_predictions(predictions)
        gates = _evaluate_gate(metrics, detections, config)

        predictions_path = output_dir / "hazard_predictions.csv"
        detections_path = output_dir / "episode_detections.csv"
        coefficients_path = output_dir / "coefficients.csv"
        metrics_path = output_dir / "metrics.csv"
        gates_path = output_dir / "gate_results.csv"
        report_path = output_dir / "report.md"
        predictions.to_csv(
            predictions_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        detections.to_csv(
            detections_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format="%.10f",
        )
        coefficients.to_csv(
            coefficients_path, index=False, encoding="utf-8", float_format="%.10f"
        )
        metrics.to_csv(metrics_path, index=False, encoding="utf-8", float_format="%.10f")
        gates.to_csv(gates_path, index=False, encoding="utf-8", float_format="%.10f")
        _write_report(report_path, run_id, metrics, gates)

        gate_passed = bool(gates.iloc[0]["promotion_gate_passed"])
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "model_count": int(config["budget"]["expected_model_count"]),
                "process_feature_count": len(process_columns),
                "heldout_episode_count": 8,
                "promotion_gate_passed": gate_passed,
                "submission_eligible": False,
                "artifacts": {
                    "predictions": predictions_path.relative_to(PROJECT_ROOT).as_posix(),
                    "predictions_sha256": sha256(predictions_path),
                    "episode_detections": detections_path.relative_to(PROJECT_ROOT).as_posix(),
                    "episode_detections_sha256": sha256(detections_path),
                    "coefficients": coefficients_path.relative_to(PROJECT_ROOT).as_posix(),
                    "coefficients_sha256": sha256(coefficients_path),
                    "metrics": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
                    "metrics_sha256": sha256(metrics_path),
                    "gate_results": gates_path.relative_to(PROJECT_ROOT).as_posix(),
                    "gate_results_sha256": sha256(gates_path),
                    "report": report_path.relative_to(PROJECT_ROOT).as_posix(),
                    "report_sha256": sha256(report_path),
                },
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        primary = gates.iloc[0]
        registry_row = {
            "run_id": run_id,
            "status": "completed",
            "experiment_role": config["experiment_role"],
            "task": config["task"],
            "protocol_version": "round2_v3",
            "control_run_id": "calendar_only_hazard",
            "hypothesis": config["hypothesis"],
            "primary_change": config["primary_change"],
            "model_name": config["models"]["candidate"],
            "holdout_evaluated": False,
            "submission_eligible": False,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": manifest["duration_seconds"],
            "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "notes": (
                f"24h_ap_lift={primary['primary_average_precision_lift']:.4f}; "
                f"24h_detection={primary['primary_episode_detection_fraction']:.4f}; "
                f"promotion_gate_passed={gate_passed}; diagnostic only"
            ),
        }
        if config["output"]["append_registry"]:
            _append_registry(registry_row)
            registry_written = True
        _append_event(
            event_log,
            "run_completed",
            duration_seconds=manifest["duration_seconds"],
            promotion_gate_passed=gate_passed,
        )
        print("\n" + metrics.loc[metrics["scope"] == "pooled"].to_string(index=False), flush=True)
        print("\n" + gates.to_string(index=False), flush=True)
        print(
            f"PASS run_id={run_id} output={output_dir} "
            f"duration_seconds={manifest['duration_seconds']} "
            f"promotion_gate_passed={gate_passed}",
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
        _append_event(event_log, "run_failed", failure_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "control_run_id": "calendar_only_hazard",
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": config["models"]["candidate"],
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
