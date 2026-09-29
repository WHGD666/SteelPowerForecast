"""Materialize and validate the frozen round2 v3 walk-forward origins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.common.io_utils import sha256, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "round2_v3" / "splits_v3.yaml"
OUTPUT_CSV = PROJECT_ROOT / "manifests" / "round2_v3" / "validation_origins_v3.csv"
OUTPUT_SHA = PROJECT_ROOT / "manifests" / "round2_v3" / "validation_origins_v3.sha256"
OUTPUT_MANIFEST = PROJECT_ROOT / "manifests" / "round2_v3" / "splits_v3.json"


def materialize(config: dict) -> pd.DataFrame:
    """Expand fold date ranges to one row per validation origin."""
    if config["strategy"] != "expanding_window_walk_forward":
        raise ValueError("only expanding_window_walk_forward is accepted")
    if config.get("random_split") is not False:
        raise ValueError("random_split must be false")

    raw_start = pd.Timestamp(config["raw_train_start"])
    raw_end = pd.Timestamp(config["raw_train_end"])
    max_horizon = int(config["maximum_horizon_minutes"])
    records: list[dict[str, object]] = []
    validation_sets: list[set[pd.Timestamp]] = []

    for fold in config["folds"]:
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = pd.Timestamp(fold["validation_start"])
        validation_end = pd.Timestamp(fold["validation_end"])
        origins = pd.date_range(validation_start, validation_end, freq="15min")
        if len(origins) == 0:
            raise ValueError(f"empty validation range: {fold['fold_id']}")
        if train_end >= origins.min():
            raise ValueError(f"train/validation boundary overlap: {fold['fold_id']}")
        if origins.min() < raw_start:
            raise ValueError(f"validation starts before raw data: {fold['fold_id']}")
        max_label_end = origins.max() + pd.Timedelta(minutes=max_horizon - 1)
        if max_label_end > raw_end:
            raise ValueError(f"validation labels exceed raw data: {fold['fold_id']}")
        if any(origins.minute % 15) or any(origins.second):
            raise ValueError(f"origin grid is not quarter-hour aligned: {fold['fold_id']}")

        latest_train_origin = train_end - pd.Timedelta(minutes=max_horizon - 1)
        current_set = set(origins)
        if any(current_set & previous for previous in validation_sets):
            raise ValueError(f"validation origins overlap: {fold['fold_id']}")
        validation_sets.append(current_set)

        for origin in origins:
            records.append(
                {
                    "split_version": config["split_version"],
                    "fold_id": fold["fold_id"],
                    "role": fold["role"],
                    "train_end": train_end,
                    "latest_eligible_train_origin": latest_train_origin,
                    "validation_origin": origin,
                    "short_label_end": origin + pd.Timedelta(minutes=119),
                    "long_label_end": origin + pd.Timedelta(minutes=max_horizon - 1),
                    "regime_note": fold["regime_note"],
                }
            )

    result = pd.DataFrame.from_records(records)
    if result["validation_origin"].duplicated().any():
        raise ValueError("validation origins must be globally disjoint")
    return result


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    result = materialize(config)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )
    digest = sha256(OUTPUT_CSV)
    OUTPUT_SHA.write_text(f"{digest}  {OUTPUT_CSV.name}\n", encoding="ascii")
    fold_summary = []
    for fold_id, group in result.groupby("fold_id", sort=False):
        fold_summary.append(
            {
                "fold_id": fold_id,
                "role": group["role"].iloc[0],
                "origins": int(len(group)),
                "validation_start": group["validation_origin"].min().isoformat(),
                "validation_end": group["validation_origin"].max().isoformat(),
                "train_end": group["train_end"].iloc[0].isoformat(),
                "latest_eligible_train_origin": group[
                    "latest_eligible_train_origin"
                ].iloc[0].isoformat(),
                "maximum_label_end": group["long_label_end"].max().isoformat(),
            }
        )
    manifest = {
        "split_version": config["split_version"],
        "status": config["status"],
        "strategy": config["strategy"],
        "random_split": config["random_split"],
        "config_sha256": sha256(CONFIG_PATH),
        "assignment_sha256": digest,
        "validation_origin_rows": int(len(result)),
        "fold_count": int(result["fold_id"].nunique()),
        "globally_disjoint_validation_origins": True,
        "folds": fold_summary,
    }
    write_json(OUTPUT_MANIFEST, manifest)
    print(
        "PASS "
        f"folds={manifest['fold_count']} origins={manifest['validation_origin_rows']} "
        f"sha256={digest}"
    )


if __name__ == "__main__":
    main()
