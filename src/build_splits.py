"""Materialize and fingerprint the frozen IronFlow rolling-origin split assignments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import yaml


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_assignments(config: dict) -> pd.DataFrame:
    frequency = pd.Timedelta(minutes=int(config["frequency_minutes"]))
    max_horizon = int(config["maximum_horizon_steps"])
    records: list[dict[str, object]] = []

    specifications = [
        (item, "development", False) for item in config["development_folds"]
    ] + [(config["sealed_holdout"], "holdout", True)]
    for spec, role, sealed in specifications:
        origins = pd.date_range(
            pd.Timestamp(spec["validation_origin_start"]),
            pd.Timestamp(spec["validation_origin_end"]),
            freq=frequency,
        )
        for origin in origins:
            records.append(
                {
                    "fold_id": spec["fold_id"],
                    "role": role,
                    "sealed": sealed,
                    "train_end": pd.Timestamp(spec["train_end"]),
                    "origin": origin,
                    "max_target_time": origin + max_horizon * frequency,
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.duplicated(["fold_id", "origin"]).any():
        raise AssertionError("duplicate fold/origin assignment")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/splits_v1.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assignments = build_assignments(config)

    output_path = root / config["output"]["assignment_file"]
    fingerprint_path = root / config["output"]["fingerprint_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
        lineterminator="\n",
    )
    fingerprint = file_hash(output_path)
    fingerprint_path.write_text(f"{fingerprint}  {output_path.name}\n", encoding="ascii")
    print(
        f"PASS rows={len(assignments)} dev_rows={(assignments['role'] == 'development').sum()} "
        f"holdout_rows={(assignments['role'] == 'holdout').sum()} sha256={fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
