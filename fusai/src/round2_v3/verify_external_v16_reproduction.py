"""Compare regenerated v16 CSV files with the frozen external candidate."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_RELATIVE = Path(
    "submissions/round2_v3/20261001_external_best_61_pending_exact/这模型调的有力量_gas_predict_semi.zip"
)
EXPECTED_MEMBER_HASHES = {
    "s_result.csv": "e96e633801c2d577006670ea178904e253f5e1da75faaa000a0e264eaf3b4a4f",
    "l_result.csv": "3c6e55d4d92b8ecee65609abb6784a9fe15178afc2f1f7cf04966372bf28de28",
}


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_member(candidate_bytes: bytes, reference_bytes: bytes) -> dict[str, object]:
    candidate = pd.read_csv(io.BytesIO(candidate_bytes))
    reference = pd.read_csv(io.BytesIO(reference_bytes))
    columns_identical = list(candidate.columns) == list(reference.columns)
    shape_identical = candidate.shape == reference.shape
    datetime_identical = bool(
        columns_identical
        and shape_identical
        and candidate["datetime"].equals(reference["datetime"])
    )
    result: dict[str, object] = {
        "candidate_sha256": bytes_sha256(candidate_bytes),
        "reference_sha256": bytes_sha256(reference_bytes),
        "byte_identical": candidate_bytes == reference_bytes,
        "columns_identical": columns_identical,
        "shape_identical": shape_identical,
        "datetime_identical": datetime_identical,
        "candidate_shape": list(candidate.shape),
        "reference_shape": list(reference.shape),
    }
    if columns_identical and shape_identical and datetime_identical:
        candidate_values = candidate.drop(columns="datetime").to_numpy(dtype=float)
        reference_values = reference.drop(columns="datetime").to_numpy(dtype=float)
        delta = np.abs(candidate_values - reference_values)
        result.update(
            {
                "numeric_identical": bool(np.array_equal(candidate_values, reference_values)),
                "cells_differing_at_1e_9": int(np.count_nonzero(delta > 1e-9)),
                "mean_absolute_difference": float(delta.mean()),
                "maximum_absolute_difference": float(delta.max()),
            }
        )
    else:
        result["numeric_identical"] = False
    return result


def short_long_consistent(candidate_dir: Path) -> bool:
    short = pd.read_csv(candidate_dir / "s_result.csv")
    long = pd.read_csv(candidate_dir / "l_result.csv")
    if not short["datetime"].equals(long["datetime"]):
        return False
    for target in ("generator_1", "generator_all"):
        for minute in range(15, 121, 15):
            column = f"{target}_t+{minute}_pred"
            if not np.array_equal(short[column].to_numpy(), long[column].to_numpy()):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-zip", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    candidate_dir = args.candidate_dir
    if not candidate_dir.is_absolute():
        candidate_dir = root / candidate_dir
    reference_zip = args.reference_zip or (root / REFERENCE_RELATIVE)
    if not reference_zip.is_absolute():
        reference_zip = root / reference_zip

    results: dict[str, object] = {
        "candidate_dir": str(candidate_dir),
        "reference_zip": str(reference_zip),
        "members": {},
    }
    with zipfile.ZipFile(reference_zip) as archive:
        for name, expected_hash in EXPECTED_MEMBER_HASHES.items():
            candidate_path = candidate_dir / name
            if not candidate_path.is_file():
                raise FileNotFoundError(candidate_path)
            reference_bytes = archive.read(name)
            if bytes_sha256(reference_bytes) != expected_hash:
                raise AssertionError(f"frozen reference member hash changed: {name}")
            comparison = compare_member(candidate_path.read_bytes(), reference_bytes)
            results["members"][name] = comparison

    results["short_equals_long_first_eight"] = short_long_consistent(candidate_dir)
    results["exact_reproduction"] = bool(
        all(item["byte_identical"] for item in results["members"].values())
        and results["short_equals_long_first_eight"]
    )
    results["status"] = "PASS" if results["exact_reproduction"] else "DIFFERS"
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results["exact_reproduction"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
