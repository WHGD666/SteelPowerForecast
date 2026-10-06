"""Evaluate causal consistency across overlapping rolling forecasts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.event_ablation_utils import summarize_ablation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/experiments/causal_overlap_reconciliation_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()


def _git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {item["run_id"] for item in reader}
    if not fields or str(row["run_id"]) in existing:
        raise RuntimeError("invalid or duplicate experiment registry entry")
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow({field: row.get(field, "") for field in fields})


def causal_overlap_reconcile(
    frame: pd.DataFrame,
    *,
    alpha_current: float,
    horizons: list[int],
    frequency_minutes: int,
) -> pd.DataFrame:
    """Recursively blend current and previous-origin forecasts of the same target time."""
    alpha = float(alpha_current)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha_current must be in [0, 1]")
    required = {"fold_id", "datetime", "horizon_minutes", "prediction"}
    if not required <= set(frame.columns):
        raise KeyError(f"missing columns: {sorted(required - set(frame.columns))}")
    keys = ["fold_id", "datetime", "horizon_minutes"]
    if frame.duplicated(keys).any():
        raise ValueError("forecast keys are duplicated")
    expected_horizons = [int(value) for value in horizons]
    parts: list[pd.DataFrame] = []
    for fold_id, fold in frame.groupby("fold_id", sort=False):
        source = fold.copy()
        source["datetime"] = pd.to_datetime(source["datetime"], errors="raise")
        pivot = source.pivot(index="datetime", columns="horizon_minutes", values="prediction").sort_index()
        if [int(value) for value in pivot.columns] != expected_horizons:
            raise AssertionError(f"{fold_id}: horizon grid differs")
        if pivot.isna().any().any():
            raise AssertionError(f"{fold_id}: incomplete origin-horizon matrix")
        base = pivot.to_numpy(dtype=float)
        reconciled = base.copy()
        origins = pd.DatetimeIndex(pivot.index)
        expected_delta = pd.Timedelta(minutes=int(frequency_minutes))
        for row in range(1, len(origins)):
            if origins[row] - origins[row - 1] != expected_delta:
                continue
            reconciled[row, :-1] = (
                alpha * base[row, :-1]
                + (1.0 - alpha) * reconciled[row - 1, 1:]
            )
        lookup = (
            pd.DataFrame(reconciled, index=origins, columns=expected_horizons)
            .rename_axis(index="datetime", columns="horizon_minutes")
            .stack(future_stack=True)
            .rename("prediction_reconciled")
            .reset_index()
        )
        source = source.merge(lookup, on=["datetime", "horizon_minutes"], how="left", validate="one_to_one")
        if source["prediction_reconciled"].isna().any():
            raise AssertionError(f"{fold_id}: reconciled lookup is incomplete")
        source["prediction"] = source.pop("prediction_reconciled")
        parts.append(source)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != len(frame):
        raise AssertionError("reconciliation changed row count")
    return result


def _load_control(spec: dict[str, Any], folds: list[str]) -> pd.DataFrame:
    path = PROJECT_ROOT / spec["oof"]
    if sha256(path) != spec["sha256"]:
        raise AssertionError(f"OOF hash mismatch: {path}")
    frame = pd.read_csv(
        path,
        parse_dates=["datetime", "interval_start"],
        low_memory=False,
    )
    frame = frame.loc[(frame["variant"] == spec["source_variant"]) & frame["fold_id"].isin(folds)].copy()
    if frame.empty:
        raise AssertionError("control OOF selection is empty")
    return frame


def _metrics(frame: pd.DataFrame, actual_column: str) -> pd.DataFrame:
    source = frame.copy()
    source["actual"] = source[actual_column]
    return summarize_ablation(source)


def _gate(metrics: pd.DataFrame, config: dict[str, Any], candidate: str) -> dict[str, Any]:
    gate = config["selection_gate"]
    control = config["reconciliation"]["control_variant"]

    def accuracy(scope: str, variant: str, *, fold: str = "all", bucket: str = "all", regime: str = "all") -> float:
        selected = metrics.loc[(metrics["scope"] == scope) & (metrics["variant"] == variant) & (metrics["fold_id"].astype(str) == fold) & (metrics["horizon_bucket"].astype(str) == bucket) & (metrics["origin_regime"].astype(str) == regime)]
        if len(selected) != 1:
            raise AssertionError(f"metric lookup not unique: {scope}/{variant}/{fold}/{bucket}/{regime}")
        return float(selected.iloc[0]["accuracy_1_minus_mape"])

    overall = (accuracy("overall", candidate) - accuracy("overall", control)) * 100
    fold_gains = {fold: (accuracy("fold", candidate, fold=fold) - accuracy("fold", control, fold=fold)) * 100 for fold in config["validation"]["fold_ids"]}
    ordinary_loss = (accuracy("origin_regime", control, regime="ordinary") - accuracy("origin_regime", candidate, regime="ordinary")) * 100
    buckets = {bucket: (accuracy("horizon_bucket", candidate, bucket=bucket) - accuracy("horizon_bucket", control, bucket=bucket)) * 100 for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440")}
    ce = metrics.loc[(metrics["scope"] == "episode") & (metrics["variant"] == control), ["episode_id", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "control"})
    pe = metrics.loc[(metrics["scope"] == "episode") & (metrics["variant"] == candidate), ["episode_id", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "candidate"})
    episodes = ce.merge(pe, on="episode_id", how="inner", validate="one_to_one")
    episode_fraction = float((episodes["candidate"] > episodes["control"]).mean()) if len(episodes) else np.nan
    pass_gain = overall >= float(gate["minimum_accuracy_gain_pct"])
    pass_folds = min(fold_gains.values()) >= float(gate["minimum_single_fold_gain_pct"])
    pass_ordinary = ordinary_loss <= float(gate["maximum_ordinary_accuracy_loss_pct"])
    pass_near_far = min(buckets["h015_120"], buckets["h735_1440"]) >= -float(gate["maximum_near_far_accuracy_loss_pct"])
    pass_episodes = (not gate["require_majority_episode_improvement"]) or (np.isfinite(episode_fraction) and episode_fraction > 0.5)
    return {"variant": candidate, "overall_accuracy_gain_pct": overall, "single_fold_min_gain_pct": min(fold_gains.values()), "single_fold_mean_gain_pct": float(np.mean(list(fold_gains.values()))), "ordinary_accuracy_loss_pct": ordinary_loss, "episode_improvement_fraction": episode_fraction, "evaluated_episode_count": len(episodes), "near_gain_pct": buckets["h015_120"], "mid1_gain_pct": buckets["h135_360"], "mid2_gain_pct": buckets["h375_720"], "far_gain_pct": buckets["h735_1440"], **{f"{fold}_gain_pct": value for fold, value in fold_gains.items()}, "pass_gain": pass_gain, "pass_folds": pass_folds, "pass_ordinary": pass_ordinary, "pass_episodes": pass_episodes, "pass_near_far": pass_near_far, "promotion_gate_passed": bool(pass_gain and pass_folds and pass_ordinary and pass_episodes and pass_near_far)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    folds = list(config["validation"]["fold_ids"])
    horizons = [int(value) for value in config["validation"]["horizons_minutes"]]
    controls = {target: _load_control(spec, folds) for target, spec in config["inputs"]["targets"].items()}
    if args.preflight_only:
        print("PASS preflight " + " ".join(f"{target}_rows={len(frame)}" for target, frame in controls.items()) + f" horizons={len(horizons)} primary_alpha={config['reconciliation']['primary_alpha_current']}", flush=True)
        return

    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    digest = hashlib.sha256()
    for path in [config_path, RUNNER_SOURCE, PROJECT_ROOT / config["inputs"]["event_episodes"], *[PROJECT_ROOT / spec["oof"] for spec in config["inputs"]["targets"].values()]]:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + "_overlap_reconcile_" + digest.hexdigest()[:10]
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    alphas = [float(config["reconciliation"]["primary_alpha_current"]), *[float(value) for value in config["reconciliation"]["sensitivity_alpha_current"]]]
    all_oof, all_metrics, gate_rows = [], [], []
    for target, control in controls.items():
        base = control.copy()
        base["variant"] = config["reconciliation"]["control_variant"]
        variants = [base]
        for alpha in alphas:
            candidate = causal_overlap_reconcile(control, alpha_current=alpha, horizons=horizons, frequency_minutes=int(config["validation"]["origin_frequency_minutes"]))
            candidate["variant"] = f"causal_reconcile_a{int(round(alpha * 100)):03d}"
            variants.append(candidate)
        combined = pd.concat(variants, ignore_index=True)
        combined.insert(0, "target", target)
        metrics = _metrics(combined, config["validation"]["primary_label"])
        metrics.insert(0, "target", target)
        all_oof.append(combined)
        all_metrics.append(metrics)
        primary_variant = f"causal_reconcile_a{int(round(float(config['reconciliation']['primary_alpha_current']) * 100)):03d}"
        row = _gate(metrics, config, primary_variant)
        row["target"] = target
        row["promotion_eligible"] = bool(target == config["selection_gate"]["primary_target"] and row["promotion_gate_passed"])
        gate_rows.append(row)
    oof = pd.concat(all_oof, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    gates = pd.DataFrame(gate_rows)
    overall = metrics.loc[metrics["scope"] == "overall", ["target", "variant", "mape", "accuracy_1_minus_mape"]].copy()
    control_accuracy = overall.loc[overall["variant"] == config["reconciliation"]["control_variant"], ["target", "accuracy_1_minus_mape"]].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
    overall = overall.merge(control_accuracy, on="target", validate="many_to_one")
    overall["delta_accuracy_pct"] = (overall["accuracy_1_minus_mape"] - overall["control_accuracy"]) * 100
    overall = overall.drop(columns="control_accuracy").sort_values(["target", "mape"])
    for name, frame in (("oof_predictions.csv", oof), ("metrics.csv", metrics), ("variant_comparison.csv", overall), ("gate_results.csv", gates)):
        frame.to_csv(output_dir / name, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
    ended = datetime.now(timezone.utc)
    manifest = {"run_id": run_id, "status": "completed", "task": config["task"], "experiment_role": config["experiment_role"], "experiment_version": config["experiment_version"], "control_artifact_id": config["control_artifact_id"], "started_at": started.isoformat(), "ended_at": ended.isoformat(), "duration_seconds": round(time.perf_counter() - clock, 3), "holdout_evaluated": False, "submission_eligible": bool(gates["promotion_eligible"].any()), "primary_alpha_current": float(config["reconciliation"]["primary_alpha_current"]), "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))}, "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in [config_path, RUNNER_SOURCE, *[PROJECT_ROOT / spec["oof"] for spec in config["inputs"]["targets"].values()]]}, "artifacts": {name.removesuffix(".csv"): (output_dir / name).relative_to(PROJECT_ROOT).as_posix() for name in ("oof_predictions.csv", "metrics.csv", "variant_comparison.csv", "gate_results.csv")}}
    for name in ("oof_predictions", "metrics", "variant_comparison", "gate_results"):
        manifest["artifacts"][name + "_sha256"] = sha256(PROJECT_ROOT / manifest["artifacts"][name])
    write_json(output_dir / "run_manifest.json", manifest)
    if config["output"]["append_registry"]:
        primary = gates.loc[gates["target"] == config["selection_gate"]["primary_target"]].iloc[0]
        _append_registry({"run_id": run_id, "status": "completed", "experiment_role": config["experiment_role"], "task": config["task"], "protocol_version": "round2_v3", "control_run_id": config["control_artifact_id"], "hypothesis": config["hypothesis"], "primary_change": config["primary_change"], "model_name": "causal_overlap_reconciliation", "long_g1_mape": float(overall.loc[(overall["target"] == "generator_1") & (overall["variant"] == f"causal_reconcile_a{int(round(float(config['reconciliation']['primary_alpha_current']) * 100)):03d}"), "mape"].iloc[0]), "holdout_evaluated": False, "submission_eligible": bool(primary["promotion_eligible"]), "git_commit": manifest["git"]["commit"], "git_dirty": manifest["git"]["dirty"], "started_at": started.isoformat(), "ended_at": ended.isoformat(), "duration_seconds": manifest["duration_seconds"], "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(), "notes": f"primary_alpha=0.50; promotion_gate_passed={bool(primary['promotion_gate_passed'])}; previous issued predictions only"})
    print("\n" + overall.to_string(index=False), flush=True)
    print("\n" + gates.to_string(index=False), flush=True)
    print(f"PASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} promotion_eligible={gates.loc[gates['promotion_eligible'], 'target'].tolist()}", flush=True)


if __name__ == "__main__":
    main()

