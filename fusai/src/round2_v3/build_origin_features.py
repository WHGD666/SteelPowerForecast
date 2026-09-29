"""Build quarter-hour origin features with explicit source-time provenance."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import read_csv_strict, sha256, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "round2_v3" / "features_v1.yaml"
BASE_DIR = PROJECT_ROOT / "data" / "prepared" / "round2_v3_causal_base_1"
CENTRAL_MANIFEST = PROJECT_ROOT / "manifests" / "round2_v3" / "origin_features_v1.json"
CENTRAL_REGISTRY = PROJECT_ROOT / "manifests" / "round2_v3" / "feature_registry_v1.csv"


def _observed_aggregates(frame: pd.DataFrame, config: dict) -> dict[str, pd.Series]:
    aggregates: dict[str, pd.Series] = {}
    for name, specification in config["derived_observed_aggregates"].items():
        columns = specification["columns"]
        missing = set(columns) - set(frame.columns)
        if missing:
            raise KeyError(f"aggregate {name} missing columns: {sorted(missing)}")
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        operation = specification["operation"]
        if operation == "sum":
            aggregates[name] = values.sum(axis=1, min_count=1)
        elif operation == "count":
            aggregates[name] = values.notna().sum(axis=1).astype(float)
        else:
            raise ValueError(f"unsupported aggregate operation: {operation}")
    return aggregates


def build_origin_features(
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create causal features and a registry describing each source interval."""
    local = frame.copy()
    local["datetime"] = pd.to_datetime(local["datetime"], errors="raise")
    if local["datetime"].duplicated().any():
        raise ValueError("causal base contains duplicate datetime values")
    local = local.sort_values("datetime").set_index("datetime")
    expected_index = pd.date_range(local.index.min(), local.index.max(), freq="1min")
    if not local.index.equals(expected_index):
        raise ValueError("causal base must already be a complete one-minute grid")

    origins = local.index[
        (local.index.minute % 15 == 0) & (local.index.second == 0)
    ]
    feature_values: dict[str, np.ndarray] = {}
    registry: list[dict[str, object]] = []

    signals: dict[str, pd.Series] = {}
    for column in config["signal_columns"]:
        if column not in local:
            raise KeyError(f"configured signal absent from causal base: {column}")
        signals[column] = pd.to_numeric(local[column], errors="coerce")
    signals.update(_observed_aggregates(local, config))

    def add_feature(
        name: str,
        values: pd.Series | np.ndarray,
        base_signal: str,
        transform: str,
        source_offset_min: int,
        source_offset_max: int,
    ) -> None:
        if name in feature_values:
            raise ValueError(f"duplicate feature name: {name}")
        feature_values[name] = np.asarray(values, dtype=float)
        registry.append(
            {
                "feature_name": name,
                "base_signal": base_signal,
                "transform": transform,
                "source_offset_min_minutes": source_offset_min,
                "source_offset_max_minutes": source_offset_max,
                "causal": source_offset_max <= 0,
            }
        )

    history = config["history"]
    minimum_fraction = float(history["rolling_minimum_observed_fraction"])
    for signal_name, series in signals.items():
        if history["include_current"]:
            add_feature(signal_name, series.reindex(origins), signal_name, "current", 0, 0)
        for lag in history["lag_minutes"]:
            lag = int(lag)
            add_feature(
                f"{signal_name}__lag_{lag}m",
                series.shift(lag).reindex(origins),
                signal_name,
                f"lag_{lag}m",
                -lag,
                -lag,
            )
        for window in history["rolling_windows_minutes"]:
            window = int(window)
            min_periods = max(1, math.ceil(window * minimum_fraction))
            rolling = series.rolling(window=window, min_periods=min_periods)
            for statistic in history["rolling_statistics"]:
                if statistic == "mean":
                    values = rolling.mean().reindex(origins)
                elif statistic == "std":
                    values = rolling.std(ddof=0).reindex(origins)
                else:
                    raise ValueError(f"unsupported rolling statistic: {statistic}")
                add_feature(
                    f"{signal_name}__roll_{window}m_{statistic}",
                    values,
                    signal_name,
                    f"rolling_{window}m_{statistic}",
                    -(window - 1),
                    0,
                )

    for signal_name in config["signal_columns"]:
        missing_column = f"{signal_name}__missing"
        if missing_column not in local:
            raise KeyError(f"missing indicator absent: {missing_column}")
        missing_series = pd.to_numeric(local[missing_column], errors="raise")
        for window in history["missing_fraction_windows_minutes"]:
            window = int(window)
            values = missing_series.rolling(window=window, min_periods=1).mean().reindex(origins)
            add_feature(
                f"{signal_name}__missing_fraction_{window}m",
                values,
                missing_column,
                f"rolling_{window}m_mean",
                -(window - 1),
                0,
            )

    indicator_config = config["current_state_indicators"]
    indicator_columns = [
        column
        for column in local.columns
        if any(column.endswith(suffix) for suffix in indicator_config["include_suffixes"])
        or any(column.startswith(prefix) for prefix in indicator_config["include_prefixes"])
    ]
    for column in indicator_columns:
        add_feature(
            column,
            pd.to_numeric(local[column], errors="raise").reindex(origins),
            column,
            "current_state_indicator",
            0,
            0,
        )

    hour_angle = 2 * np.pi * (origins.hour + origins.minute / 60) / 24
    minute_angle = 2 * np.pi * origins.minute / 60
    day_angle = 2 * np.pi * origins.dayofweek / 7
    calendar_values = {
        "hour_sin": np.sin(hour_angle),
        "hour_cos": np.cos(hour_angle),
        "minute_sin": np.sin(minute_angle),
        "minute_cos": np.cos(minute_angle),
        "dayofweek_sin": np.sin(day_angle),
        "dayofweek_cos": np.cos(day_angle),
        "is_weekend": (origins.dayofweek >= 5).astype(float),
        "month": origins.month.astype(float),
        "day_of_month": origins.day.astype(float),
    }
    for name in config["origin_calendar"]:
        add_feature(name, calendar_values[name], "datetime", "calendar", 0, 0)

    result = pd.DataFrame(feature_values, index=origins)
    result.index.name = "datetime"
    feature_frame = result.reset_index()
    registry_frame = pd.DataFrame.from_records(registry)
    if not registry_frame["causal"].all():
        raise AssertionError("feature registry contains a future-dependent feature")
    return feature_frame, registry_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = PROJECT_ROOT / config["output"]["directory"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    partition_records = []
    reference_registry: pd.DataFrame | None = None
    for partition in ("train", "test"):
        source = BASE_DIR / f"{partition}_causal_base_1min.csv"
        frame, encoding = read_csv_strict(source)
        features, registry = build_origin_features(frame, config)
        if reference_registry is None:
            reference_registry = registry
        else:
            pd.testing.assert_frame_equal(reference_registry, registry)
        output_name = config["output"][f"{partition}_file"]
        output_path = output_dir / output_name
        features.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format=config["output"]["float_format"],
        )
        partition_records.append(
            {
                "partition": partition,
                "source": str(source.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_encoding": encoding,
                "source_sha256": sha256(source),
                "output": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "output_sha256": sha256(output_path),
                "rows": int(len(features)),
                "columns": int(len(features.columns)),
                "datetime_min": features["datetime"].min().isoformat(),
                "datetime_max": features["datetime"].max().isoformat(),
            }
        )

    assert reference_registry is not None
    registry_path = output_dir / "feature_registry.csv"
    reference_registry.to_csv(registry_path, index=False, encoding="utf-8")
    reference_registry.to_csv(CENTRAL_REGISTRY, index=False, encoding="utf-8")
    manifest = {
        "artifact_version": config["feature_version"],
        "config": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "config_sha256": sha256(config_path),
        "base_manifest_sha256": sha256(BASE_DIR / "manifest.json"),
        "feature_count": int(len(reference_registry)),
        "maximum_source_time_offset_minutes": int(
            reference_registry["source_offset_max_minutes"].max()
        ),
        "minimum_source_time_offset_minutes": int(
            reference_registry["source_offset_min_minutes"].min()
        ),
        "all_features_causal": bool(reference_registry["causal"].all()),
        "target_history_included": config["target_history_included"],
        "future_observations_included": config["future_observations_included"],
        "registry": str(registry_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "registry_sha256": sha256(registry_path),
        "partitions": partition_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(CENTRAL_MANIFEST, manifest)
    print(
        "PASS "
        f"features={manifest['feature_count']} max_source_offset="
        f"{manifest['maximum_source_time_offset_minutes']} "
        + " ".join(f"{p['partition']}_rows={p['rows']}" for p in partition_records)
        + f" output={output_dir}"
    )


if __name__ == "__main__":
    main()
