"""Build the canonical 15-minute interval target artifact.

This is deliberately a label-only step.  It does not clean covariates, create
features, fit a model or infer missing target values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io_utils import read_csv_strict, sha256, write_json
from src.round2_v3.contracts import TARGETS, interval_mean_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "train" / "Semi_load.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "prepared" / "round2_v3_labels_1"
)
CENTRAL_MANIFEST = (
    PROJECT_ROOT / "manifests" / "round2_v3" / "prepared_labels_v3.json"
)


def build_interval_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a complete 15-minute grid with strict target interval means."""
    if "datetime" not in frame:
        raise KeyError("Semi_load.csv must contain datetime")
    timestamps = pd.to_datetime(frame["datetime"], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("duplicate datetime values are not allowed")
    start = timestamps.min().floor("15min")
    end = timestamps.max().floor("15min")
    interval_index = pd.date_range(start, end, freq="15min")
    result = pd.DataFrame({"datetime": interval_index})
    for target in TARGETS:
        result[target] = interval_mean_series(frame, target).reindex(interval_index).to_numpy()
    result["label_complete"] = result[list(TARGETS)].notna().all(axis=1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_csv = output_dir / "interval_targets_15min.csv"
    local_manifest = output_dir / "manifest.json"
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite versioned prepared data: {output_dir}"
        )

    frame, encoding = read_csv_strict(source, usecols=["datetime", *TARGETS])
    interval_table = build_interval_table(frame)
    output_dir.mkdir(parents=True, exist_ok=False)
    interval_table.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.10f",
    )

    incomplete = interval_table.loc[~interval_table["label_complete"], "datetime"]
    manifest = {
        "artifact_version": "round2_v3_labels_1",
        "source": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_encoding": encoding,
        "source_sha256": sha256(source),
        "output": str(output_csv.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "output_sha256": sha256(output_csv),
        "rows": int(len(interval_table)),
        "datetime_min": interval_table["datetime"].min().isoformat(),
        "datetime_max": interval_table["datetime"].max().isoformat(),
        "complete_intervals": int(interval_table["label_complete"].sum()),
        "incomplete_intervals": int((~interval_table["label_complete"]).sum()),
        "incomplete_interval_starts": [value.isoformat() for value in incomplete],
        "targets": list(TARGETS),
        "interval_minutes": 15,
        "interval_closed": "left",
        "require_all_source_minutes": True,
        "missing_target_fill": "forbidden",
        "label_formula": "t+h uses mean(t+h-15min through t+h-1min)",
    }
    write_json(local_manifest, manifest)
    write_json(CENTRAL_MANIFEST, manifest)
    print(
        "PASS "
        f"rows={manifest['rows']} complete={manifest['complete_intervals']} "
        f"incomplete={manifest['incomplete_intervals']} output={output_csv}"
    )


if __name__ == "__main__":
    main()
