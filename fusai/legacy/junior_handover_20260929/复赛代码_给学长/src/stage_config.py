"""Staging helpers shared by the frozen prelim (v1) and semi (v2) pipelines.

The values in :data:`COMPAT_DEFAULTS` are the **frozen prelim v1 contract**.  They
exist only so that the prelim pipelines stay reproducible byte-for-byte after the
semi adaptation.  Every semi v2 configuration must declare these keys explicitly
and must never rely on the defaults; see ``docs/task_contract_v2.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Centralised compatibility defaults.  Do not add staging constants anywhere else.
COMPAT_DEFAULTS: dict[str, Any] = {
    "data_prefix": "Pre",
    "test_data_prefix": "Pre_test",
    "result_offsets_minutes": [15, 30, 45, 60, 75, 90, 105, 120],
    "expected_origin_rows": 192,
    "sampling_minutes": 15,
    "time_budget_seconds": 30.0 * 60.0,
    "modeling_strategy": "per_horizon",
    "horizon_group_size": 8,
}

SUPPORTED_MODELING_STRATEGIES = ("per_horizon", "horizon_grouped")


def resolve_data_prefix(config: dict[str, Any] | None, partition: str) -> str:
    """Return the train/test CSV prefix, falling back to the frozen v1 default."""
    if partition == "train":
        key, default = "data_prefix", COMPAT_DEFAULTS["data_prefix"]
    elif partition == "test":
        key, default = "test_data_prefix", COMPAT_DEFAULTS["test_data_prefix"]
    else:
        raise ValueError(f"unknown partition: {partition}")
    if not config:
        return str(default)
    return str(config.get(key, default))


def load_contract_config(path: Path | str | None) -> dict[str, Any]:
    """Resolve the submission contract.

    ``path is None`` returns the frozen prelim v1 contract, which keeps the
    original ``validate_contract.py`` behaviour reproducible.
    """
    if path is None:
        return {
            "stage": "prelim_short_forecast",
            "data_prefix": COMPAT_DEFAULTS["data_prefix"],
            "test_data_prefix": COMPAT_DEFAULTS["test_data_prefix"],
            "result_offsets_minutes": list(COMPAT_DEFAULTS["result_offsets_minutes"]),
            "expected_origin_rows": COMPAT_DEFAULTS["expected_origin_rows"],
            "sampling_minutes": COMPAT_DEFAULTS["sampling_minutes"],
            "raw_sampling_minutes": COMPAT_DEFAULTS["sampling_minutes"],
            "is_v1_compatible": True,
            "source_path": None,
        }
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    targets = dict(config.get("targets") or {})
    offsets = targets.get("result_offsets_minutes")
    if offsets is None:
        offsets = targets.get("long_offsets_minutes")
    if offsets is None:
        raise AssertionError(
            f"{config_path}: semi contract config must declare targets.result_offsets_minutes"
        )
    time_block = dict(config.get("time") or {})
    sampling = time_block.get("sampling_minutes")
    if sampling is None:
        raise AssertionError(f"{config_path}: semi contract config must declare time.sampling_minutes")
    raw_sampling = time_block.get("raw_sampling_minutes", sampling)
    return {
        "stage": config.get("stage"),
        "data_prefix": config.get("data_prefix"),
        "test_data_prefix": config.get("test_data_prefix"),
        "result_offsets_minutes": [int(value) for value in offsets],
        "expected_origin_rows": config.get("expected_origin_rows"),
        "sampling_minutes": int(sampling),
        "raw_sampling_minutes": int(raw_sampling),
        "is_v1_compatible": False,
        "source_path": config_path.as_posix(),
    }


def resolve_modeling_strategy(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the semi modelling strategy, group size and wall-clock budget."""
    config = config or {}
    strategy = str(config.get("modeling_strategy", COMPAT_DEFAULTS["modeling_strategy"]))
    if strategy not in SUPPORTED_MODELING_STRATEGIES:
        raise ValueError(
            f"unsupported modeling_strategy: {strategy}; allowed {list(SUPPORTED_MODELING_STRATEGIES)}"
        )
    group_size = int(config.get("horizon_group_size", COMPAT_DEFAULTS["horizon_group_size"]))
    if group_size < 1:
        raise ValueError("horizon_group_size must be >= 1")
    budget = float(config.get("time_budget_seconds", COMPAT_DEFAULTS["time_budget_seconds"]))
    if budget <= 0:
        raise ValueError("time_budget_seconds must be positive")
    return {
        "strategy": strategy,
        "horizon_group_size": group_size,
        "time_budget_seconds": budget,
    }


def build_horizon_groups(horizons: list[int], group_size: int) -> list[list[int]]:
    """Split horizons into contiguous cohorts preserving the given order."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    ordered = [int(value) for value in horizons]
    return [ordered[index : index + group_size] for index in range(0, len(ordered), group_size)]