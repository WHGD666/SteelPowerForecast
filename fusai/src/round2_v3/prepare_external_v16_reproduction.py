"""Prepare an isolated workspace for reproducing the external v16 candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path


SNAPSHOT_RELATIVE = Path("legacy/junior_handover_20261001_v16_candidate")
WORKSPACE_RELATIVE = Path("outputs/reproduction_workspaces/external_v16_61_candidate")
REFERENCE_RELATIVE = Path(
    "submissions/round2_v3/20261001_external_best_61_pending_exact/这模型调的有力量_gas_predict_semi.zip"
)

EXPECTED_EVIDENCE = {
    "evidence/AI复现手册_60分保底.md": "e03d304c1def1bca8bcf37cf1f97d5e519491cae4f955ba914e3eb3ccbcc421c",
    "evidence/复赛代码_给学长.zip": "cc9067dd1b7da565dbf3f17677b68f185c8baf70f618dca97a8109e92478d728",
    "source_snapshot/src/build_submission_semi.py": "91d62dac938d084b863b7f1c1f0bbdb32fb93669d0eafc481e84ba6ed15cac8f",
    "source_snapshot/src/build_submission_semi_v15_sepwindow.py": "77a513f13172a41fd693aeb8644b2ddde426086b487efd07422326c9f31be64c",
    "source_snapshot/configs/submission_semi_composed.yaml": "b2275abec284e5a23f5e014f704c3242c641a876a2b4f42b79123c708b2f9b57",
    "source_snapshot/configs/cleaning_v2.yaml": "9a782cfe4fb8afbe6706825787e16c39d96ba6f18afbefea263e06d8eeb3b0f4",
    "source_snapshot/requirements.txt": "1a67e8e7d370c230640fcc56ec312a2d4de43d0e54b46716c3ee48b8e4d78213",
}

DATA_FILES = {
    "data/Semi_gas.csv": (
        "data/raw/train/Semi_gas.csv",
        "b54d7b4e9f4ec41d2b74822b7fabd87e8f7bb0fedfe09f7a8b5374379b20b58c",
    ),
    "data/Semi_gas_holder.csv": (
        "data/raw/train/Semi_gas_holder.csv",
        "241261644742a63f3587b0fd02441989e71ef43edcfdafaabcd35a1ac68569fa",
    ),
    "data/Semi_gas_user.csv": (
        "data/raw/train/Semi_gas_user.csv",
        "6810a0edf71d16a3b01121167d16adedcee0afd81147f44d8384c82161b2288d",
    ),
    "data/Semi_load.csv": (
        "data/raw/train/Semi_load.csv",
        "0d8c14e3abffe91137c18abde00ec19a62a42e80683f7ebda15da8bd955dc309",
    ),
    "test/Semi_test_gas.csv": (
        "data/raw/test/Semi_test_gas.csv",
        "60d1d23b43ebf71c9a6161f8030699ce831908c1dcd29c64d8d19f1fa7121ef2",
    ),
    "test/Semi_test_gas_holder.csv": (
        "data/raw/test/Semi_test_gas_holder.csv",
        "bb06cd2e509645c4e1e22b04df17d8e187a6349c73dcaf4f93c0c91018a8f989",
    ),
    "test/Semi_test_gas_user.csv": (
        "data/raw/test/Semi_test_gas_user.csv",
        "18a5783cf62f385e44988c59536a5a18a3f4a6437cf544416179b71fd553c48c",
    ),
    "test/Semi_test_load.csv": (
        "data/raw/test/Semi_test_load.csv",
        "ca06847286fcbafe7c06f8617638895404ee266810735c8850c615af850b7bf7",
    ),
}

EXPECTED_REFERENCE_SHA256 = "78b0e4eb024e5cc1d058bb8a77a9ac506246304105bbdb070cedaaa91071b12f"
EXPECTED_PACKAGES = {
    "lightgbm": ("lightgbm", "4.7.0"),
    "numpy": ("numpy", "2.1.3"),
    "pandas": ("pandas", "2.2.3"),
    "scikit-learn": ("sklearn", "1.5.2"),
    "PyYAML": ("yaml", "6.0.3"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def environment_manifest() -> dict[str, object]:
    if sys.version_info[:2] != (3, 10):
        raise AssertionError(f"Python 3.10 required, got {platform.python_version()}")
    versions = {
        distribution: str(importlib.import_module(module).__version__)
        for distribution, (module, _) in EXPECTED_PACKAGES.items()
    }
    mismatches = {
        distribution: {"expected": expected, "actual": versions[distribution]}
        for distribution, (_, expected) in EXPECTED_PACKAGES.items()
        if versions[distribution] != expected
    }
    if mismatches:
        raise AssertionError(f"dependency version mismatch: {mismatches}")
    metadata_candidates = {
        distribution: sorted(
            {
                f"{item.version}@{item._path}"
                for item in importlib.metadata.distributions(name=distribution)
            }
        )
        for distribution in EXPECTED_PACKAGES
    }
    warnings = [
        f"multiple metadata records for {name}: {records}"
        for name, records in metadata_candidates.items()
        if len(records) > 1
    ]
    return {
        "python": platform.python_version(),
        "runtime_import_versions": versions,
        "distribution_metadata_candidates": metadata_candidates,
        "warnings": warnings,
    }


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--create", action="store_true", help="create the ignored local workspace after preflight")
    args = parser.parse_args()

    root = args.root.resolve()
    snapshot = root / SNAPSHOT_RELATIVE
    workspace = root / WORKSPACE_RELATIVE
    reference = root / REFERENCE_RELATIVE

    evidence_hashes = {
        relative: checked_hash(snapshot / relative, expected)
        for relative, expected in EXPECTED_EVIDENCE.items()
    }
    data_hashes = {
        target: checked_hash(root / source, expected)
        for target, (source, expected) in DATA_FILES.items()
    }
    reference_hash = checked_hash(reference, EXPECTED_REFERENCE_SHA256)
    environment = environment_manifest()

    result: dict[str, object] = {
        "status": "PREFLIGHT_PASS",
        "root": str(root),
        "workspace": str(workspace),
        "environment": environment,
        "evidence_hashes": evidence_hashes,
        "data_hashes": data_hashes,
        "reference_sha256": reference_hash,
        "created": False,
    }

    if args.create:
        if workspace.exists():
            raise FileExistsError(f"workspace already exists and will not be overwritten: {workspace}")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(snapshot / "source_snapshot", workspace)
        materialization: dict[str, str] = {}
        for target_relative, (source_relative, _) in DATA_FILES.items():
            materialization[target_relative] = link_or_copy(
                root / source_relative, workspace / target_relative
            )
        reference_dir = workspace / "reference_submission"
        reference_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(reference, reference_dir / reference.name)
        result["created"] = True
        result["data_materialization"] = materialization
        result["reference_copy"] = str(reference_dir / reference.name)
        (workspace / "workspace_manifest.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result["status"] = "WORKSPACE_CREATED"

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
