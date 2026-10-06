"""Locate the process variables whose October distributions leave the training support.

Pure diagnostic: per raw column, compare the October test window against the full
training support and the September reference window.  No model, no submission.
"""

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
import yaml

from src.common.io_utils import sha256, write_json
from src.round2_v3.october_shift_utils import column_shift_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/round2_v3/diagnostics/october_shift_audit_v1.yaml"
REGISTRY_PATH = PROJECT_ROOT / "experiments/round2_v3/registry.csv"
RUNNER_SOURCE = Path(__file__).resolve()
HELPER_SOURCE = RUNNER_SOURCE.with_name("october_shift_utils.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _append_registry(row: dict[str, object]) -> None:
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        existing = {item["run_id"] for item in reader}
    if not fields or str(row["run_id"]) in existing:
        raise RuntimeError("invalid or duplicate experiment registry entry")
    with REGISTRY_PATH.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(
            {field: row.get(field, "") for field in fields}
        )


def _append_event(path: Path, event: str, **values: object) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _validate(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    identity: dict[str, Path] = {}
    for family, paths in config["inputs"]["families"].items():
        for side in ("train", "test"):
            path = PROJECT_ROOT / paths[side]
            if not path.exists():
                raise AssertionError(f"missing raw file for {family}/{side}: {path}")
            identity[f"{family}_{side}"] = path
    windows = config["windows"]
    train_start, train_end = pd.Timestamp(windows["train_start"]), pd.Timestamp(windows["train_end"])
    ref_start, ref_end = pd.Timestamp(windows["reference_start"]), pd.Timestamp(windows["reference_end"])
    test_start, test_end = pd.Timestamp(windows["test_start"]), pd.Timestamp(windows["test_end"])
    if not (train_start <= ref_start <= ref_end < test_start <= test_end):
        raise AssertionError("windows must be ordered train <= reference < test")
    if config["metrics"]["soft_support_quantiles"][0] >= config["metrics"]["soft_support_quantiles"][1]:
        raise AssertionError("soft support quantiles must be ordered")
    return {"config_path": config_path, "identity": identity, "windows": windows}


def _load_window(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime"], low_memory=False)
    mask = (frame["datetime"] >= start) & (frame["datetime"] <= end)
    return frame.loc[mask]


def _column_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validated = _validate(config, config_path)
    windows = validated["windows"]
    train_start, train_end = pd.Timestamp(windows["train_start"]), pd.Timestamp(windows["train_end"])
    ref_start, ref_end = pd.Timestamp(windows["reference_start"]), pd.Timestamp(windows["reference_end"])
    test_start, test_end = pd.Timestamp(windows["test_start"]), pd.Timestamp(windows["test_end"])
    excluded = set(config["inputs"]["excluded_columns"])
    soft_low, soft_high = [float(v) for v in config["metrics"]["soft_support_quantiles"]]
    minimum_finite = int(config["metrics"]["minimum_finite_values"])

    family_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for family in config["inputs"]["families"]:
        train = _load_window(PROJECT_ROOT / config["inputs"]["families"][family]["train"], train_start, train_end)
        test = _load_window(PROJECT_ROOT / config["inputs"]["families"][family]["test"], test_start, test_end)
        family_frames[family] = (train, test)
    total_columns = 0
    for family, (train, test) in family_frames.items():
        columns = [c for c in train.columns if c != "datetime" and c not in excluded]
        if not columns:
            raise AssertionError(f"family {family} has no auditable columns")
        missing_test = [c for c in columns if c not in test.columns]
        if missing_test:
            raise AssertionError(f"family {family} test file lacks columns: {missing_test[:5]}")
        total_columns += len(columns)
    if args.preflight_only:
        print(
            "PASS preflight "
            f"families={len(family_frames)} columns={total_columns} "
            f"train_rows={sum(len(t) for t, _ in family_frames.values())} "
            f"test_rows={sum(len(s) for _, s in family_frames.values())} models=0",
            flush=True,
        )
        return

    identity_paths = [config_path, RUNNER_SOURCE, HELPER_SOURCE, *validated["identity"].values()]
    digest = hashlib.sha256()
    for path in identity_paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "_october_shift_" + digest.hexdigest()[:10]
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    event_log = output_dir / "events.jsonl"
    started_clock = time.perf_counter()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "task": config["task"],
        "experiment_role": config["experiment_role"],
        "protocol_version": "round2_v3",
        "diagnostic_version": config["diagnostic_version"],
        "control_artifact_id": config["control_artifact_id"],
        "started_at": started_at.isoformat(),
        "holdout_evaluated": False,
        "submission_eligible": False,
        "model_count": 0,
        "command": {"working_directory": str(PROJECT_ROOT), "argv": [sys.executable, *sys.argv]},
        "git": {"commit": _git_value("rev-parse", "HEAD"), "dirty": bool(_git_value("status", "--porcelain"))},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "pyyaml": yaml.__version__},
        "fingerprints": {path.relative_to(PROJECT_ROOT).as_posix(): sha256(path) for path in identity_paths},
        "failure": {"type": None, "message": None, "traceback": None},
        "artifacts": {},
    }
    write_json(output_dir / "run_manifest.json", manifest)
    _append_event(event_log, "run_started", run_id=run_id)
    registry_written = False
    try:
        rows: list[dict[str, object]] = []
        for family, (train, test) in family_frames.items():
            reference = train.loc[(train["datetime"] >= ref_start) & (train["datetime"] <= ref_end)]
            for column in [c for c in train.columns if c != "datetime" and c not in excluded]:
                train_values = _column_values(train, column)
                reference_values = _column_values(reference, column)
                test_values = _column_values(test, column)
                if np.isfinite(test_values).sum() < 3 or np.isfinite(train_values).sum() < minimum_finite:
                    continue
                metrics = column_shift_metrics(
                    train_values,
                    reference_values,
                    test_values,
                    soft_low_quantile=soft_low,
                    soft_high_quantile=soft_high,
                )
                rows.append({"family": family, "column": column, **metrics})
        if not rows:
            raise AssertionError("no auditable columns produced metrics")
        metrics_frame = pd.DataFrame.from_records(rows)
        metrics_frame["abs_median_shift"] = metrics_frame["median_shift_in_reference_iqr"].abs()
        metrics_frame = metrics_frame.sort_values(
            ["out_of_support_rate", "abs_median_shift"], ascending=False, kind="stable"
        )
        path = output_dir / "column_shift_metrics.csv"
        metrics_frame.to_csv(path, index=False, encoding="utf-8", float_format="%.10f")
        artifacts = {
            "column_shift_metrics": path.relative_to(PROJECT_ROOT).as_posix(),
            "column_shift_metrics_sha256": sha256(path),
        }
        ended_at = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "audited_columns": int(len(metrics_frame)),
                "artifacts": artifacts,
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        if config["output"]["append_registry"]:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "completed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "october_distribution_shift_localization",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"audited_columns={len(metrics_frame)}; diagnostic only; no submission",
                }
            )
            registry_written = True
        _append_event(event_log, "run_completed", audited_columns=int(len(metrics_frame)))
        top = metrics_frame.head(20)
        print("\n" + top[
            ["family", "column", "test_missing_rate", "median_shift_in_reference_iqr",
             "out_of_support_below_rate", "out_of_support_above_rate", "hard_out_of_range_rate"]
        ].to_string(index=False), flush=True)
        print(
            f"\nPASS run_id={run_id} output={output_dir} duration_seconds={manifest['duration_seconds']} "
            f"audited_columns={len(metrics_frame)}",
            flush=True,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.perf_counter() - started_clock, 3),
                "failure": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        _append_event(event_log, "run_failed", error_type=type(exc).__name__, message=str(exc))
        if config["output"]["append_registry"] and not registry_written:
            _append_registry(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "experiment_role": config["experiment_role"],
                    "task": config["task"],
                    "protocol_version": "round2_v3",
                    "control_run_id": config["control_artifact_id"],
                    "hypothesis": config["hypothesis"],
                    "primary_change": config["primary_change"],
                    "model_name": "october_distribution_shift_localization",
                    "holdout_evaluated": False,
                    "submission_eligible": False,
                    "git_commit": manifest["git"]["commit"],
                    "git_dirty": manifest["git"]["dirty"],
                    "started_at": started_at.isoformat(),
                    "ended_at": manifest["ended_at"],
                    "duration_seconds": manifest["duration_seconds"],
                    "artifact_directory": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "notes": f"failure={type(exc).__name__}: {exc}",
                }
            )
        raise


if __name__ == "__main__":
    main()
