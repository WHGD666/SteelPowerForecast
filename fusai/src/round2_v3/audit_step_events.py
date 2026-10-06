"""Audit legacy low-fuel event days and causal pre-event signals.

This is a diagnostic, not a forecasting model.  Future labels are used only
to describe event outcomes.  Every precursor value is built from observations
available at or before its audit origin.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from src.common.io_utils import read_csv_strict, sha256, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "round2_v3"
    / "diagnostics"
    / "step_event_audit_v1.yaml"
)
BASE_DIR = PROJECT_ROOT / "data" / "prepared" / "round2_v3_causal_base_1"
LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "prepared"
    / "round2_v3_labels_1"
    / "interval_targets_15min.csv"
)
RAW_LOAD_PATH = PROJECT_ROOT / "data" / "raw" / "train" / "Semi_load.csv"

BF_PRODUCTION = [f"blast_furnace_{index}" for index in range(1, 6)]
AIR_HEATERS = [f"air_heater_{index}" for index in range(1, 6)]
BF_USERS = [f"blast_furnace_user{index}" for index in range(1, 5)]
CONVERTER_USERS = ["converter_user1", "converter_user2"]
GENERATOR_FUELS = [
    "generator_use_blast_furnace_gas",
    "generator_use_coke_gas",
    "generator_use_converter_gas",
]
MIXED_GAS = [
    "into_gas_mixed_blast_furnace",
    "into_gas_mixed_coke",
    "into_gas_mixed_converter",
]
TARGET_HISTORY_PREFIXES = (
    "g1_previous_complete_interval__",
    "gall_previous_complete_interval__",
)


def available_in_formal_test(feature_name: str) -> bool:
    """Return whether a precursor avoids unavailable true target history."""
    return not str(feature_name).startswith(TARGET_HISTORY_PREFIXES)


def _quarter_hour_grid(frame: pd.DataFrame, value_columns: Iterable[str]) -> pd.DataFrame:
    """Return the legacy :00/:15/:30/:45 grid with causal limited ffill."""
    columns = list(value_columns)
    missing = {"datetime", *columns} - set(frame.columns)
    if missing:
        raise KeyError(f"quarter-hour grid missing columns: {sorted(missing)}")
    local = frame[["datetime", *columns]].copy()
    local["datetime"] = pd.to_datetime(local["datetime"], errors="raise")
    if local["datetime"].duplicated().any():
        raise ValueError("duplicate datetime values are not allowed")
    local = local.sort_values("datetime")
    local = local.loc[
        (local["datetime"].dt.minute % 15 == 0)
        & (local["datetime"].dt.second == 0)
    ].reset_index(drop=True)
    local[columns] = local[columns].apply(pd.to_numeric, errors="coerce")
    local[columns] = local[columns].ffill(limit=4)
    return local


def fit_fuel_proxy(
    train_grid: pd.DataFrame,
    fuel_columns: list[str],
    target_column: str = "generator_all",
) -> np.ndarray:
    """Fit the exact legacy two-fuel linear proxy on training observations."""
    x_values = train_grid[fuel_columns].to_numpy(dtype=float)
    x = np.column_stack([x_values, np.ones(len(train_grid), dtype=float)])
    y = pd.to_numeric(train_grid[target_column], errors="coerce").to_numpy(float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if valid.sum() < len(fuel_columns) + 2:
        raise ValueError("insufficient finite observations for fuel proxy")
    coefficients, *_ = np.linalg.lstsq(x[valid], y[valid], rcond=None)
    return coefficients


def legacy_fuel_ratio(
    grid: pd.DataFrame,
    fuel_columns: list[str],
    coefficients: np.ndarray,
    short_points: int,
    baseline_points: int,
    baseline_minimum_points: int,
) -> pd.Series:
    """Calculate the legacy smoothed fuel level divided by its trailing baseline."""
    x_values = grid[fuel_columns].to_numpy(dtype=float)
    x = np.column_stack([x_values, np.ones(len(grid), dtype=float)])
    proxy = np.clip(x @ np.asarray(coefficients, dtype=float), 0.0, None)
    proxy[~np.isfinite(x).all(axis=1)] = np.nan
    smoothed = pd.Series(proxy).rolling(short_points, min_periods=1).mean()
    baseline = smoothed.rolling(
        baseline_points, min_periods=baseline_minimum_points
    ).mean()
    ratio = smoothed / baseline
    ratio.index = pd.DatetimeIndex(grid["datetime"])
    ratio.name = "legacy_fuel_ratio"
    return ratio


def event_day_catalog(
    ratio: pd.Series,
    threshold: float,
    prefix: str,
) -> pd.DataFrame:
    """Summarize every calendar date containing a below-threshold ratio."""
    local = ratio.dropna().rename("ratio").to_frame()
    local = local.loc[local["ratio"] < float(threshold)].copy()
    columns = [
        "event_day_id",
        "event_date",
        "first_flag",
        "last_flag",
        "flagged_points",
        "min_ratio",
    ]
    if local.empty:
        return pd.DataFrame(columns=columns)
    local["event_date"] = local.index.normalize()
    records: list[dict[str, object]] = []
    for index, (date, part) in enumerate(local.groupby("event_date"), start=1):
        records.append(
            {
                "event_day_id": f"{prefix}_day_{index:02d}",
                "event_date": pd.Timestamp(date),
                "first_flag": pd.Timestamp(part.index.min()),
                "last_flag": pd.Timestamp(part.index.max()),
                "flagged_points": int(len(part)),
                "min_ratio": float(part["ratio"].min()),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def merge_event_days(event_days: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Merge consecutive event dates into independent episode groups."""
    columns = [
        "episode_id",
        "first_event_date",
        "last_event_date",
        "episode_start",
        "episode_end",
        "event_day_count",
        "flagged_points",
        "min_ratio",
        "member_event_day_ids",
    ]
    if event_days.empty:
        return pd.DataFrame(columns=columns)
    ordered = event_days.sort_values("event_date").reset_index(drop=True).copy()
    day_gap = pd.to_datetime(ordered["event_date"]).diff().dt.days.fillna(99)
    ordered["episode_number"] = (day_gap > 1).cumsum()
    records: list[dict[str, object]] = []
    for index, (_, part) in enumerate(ordered.groupby("episode_number"), start=1):
        records.append(
            {
                "episode_id": f"{prefix}_episode_{index:02d}",
                "first_event_date": pd.Timestamp(part["event_date"].min()),
                "last_event_date": pd.Timestamp(part["event_date"].max()),
                "episode_start": pd.Timestamp(part["first_flag"].min()),
                "episode_end": pd.Timestamp(part["last_flag"].max()),
                "event_day_count": int(len(part)),
                "flagged_points": int(part["flagged_points"].sum()),
                "min_ratio": float(part["min_ratio"].min()),
                "member_event_day_ids": "|".join(part["event_day_id"].astype(str)),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def build_observed_signals(
    quarter_grid: pd.DataFrame,
    interval_targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build small physically motivated signal groups on the observed grid."""
    local = quarter_grid.copy().set_index("datetime")
    numeric = local.apply(pd.to_numeric, errors="coerce")
    # The third blast furnace is absent before commissioning.  Only this
    # physically documented partial-activation column is interpreted as zero.
    numeric["blast_furnace_3"] = numeric["blast_furnace_3"].fillna(0.0)

    def total(columns: list[str]) -> pd.Series:
        return numeric[columns].sum(axis=1, min_count=1)

    signals = pd.DataFrame(index=numeric.index)
    signals["bf_production_total"] = total(BF_PRODUCTION)
    signals["air_heater_total"] = total(AIR_HEATERS)
    signals["bf_user_total"] = total(BF_USERS)
    signals["converter_user_total"] = total(CONVERTER_USERS)
    signals["generator_fuel_total"] = total(GENERATOR_FUELS)
    signals["generator_bf_fuel"] = numeric["generator_use_blast_furnace_gas"]
    signals["generator_coke_fuel"] = numeric["generator_use_coke_gas"]
    signals["generator_converter_fuel"] = numeric[
        "generator_use_converter_gas"
    ]
    signals["mixed_gas_total"] = total(MIXED_GAS)
    signals["holder_2"] = numeric["blast_furnace_gas_holder_2"]

    signals["bf_balance_proxy"] = (
        signals["bf_production_total"]
        - signals["air_heater_total"]
        - signals["bf_user_total"]
        - numeric["into_gas_mixed_blast_furnace"]
    )
    signals["coke_balance_proxy"] = (
        numeric["coke_oven_1"] - numeric["into_gas_mixed_coke"]
    )
    signals["converter_balance_proxy"] = (
        numeric["converter_1"]
        - numeric["converter_user2"]
        - numeric["into_gas_mixed_converter"]
    )
    signals["balance_proxy_total"] = signals[
        ["bf_balance_proxy", "coke_balance_proxy", "converter_balance_proxy"]
    ].sum(axis=1, min_count=1)

    if interval_targets is not None:
        labels = interval_targets.copy()
        labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
        labels = labels.set_index("datetime").sort_index()
        # At origin t the [t, t+14] interval is not complete.  Shift one interval
        # so target-history diagnostics use only [t-15, t-1].
        signals["g1_previous_complete_interval"] = pd.to_numeric(
            labels["generator_1"], errors="coerce"
        ).shift(1).reindex(signals.index)
        signals["gall_previous_complete_interval"] = pd.to_numeric(
            labels["generator_all"], errors="coerce"
        ).shift(1).reindex(signals.index)
    return signals


def build_causal_transforms(
    signals: pd.DataFrame,
    delta_hours: Iterable[int],
    volatility_hours: Iterable[int],
    level_baseline_hours: int,
) -> pd.DataFrame:
    """Create transforms whose maximum source timestamp equals the origin."""
    if not isinstance(signals.index, pd.DatetimeIndex):
        raise TypeError("signals must use a DatetimeIndex")
    if not signals.index.is_monotonic_increasing or signals.index.duplicated().any():
        raise ValueError("signal index must be unique and increasing")
    transformed: dict[str, pd.Series] = {}
    for column in signals.columns:
        series = pd.to_numeric(signals[column], errors="coerce")
        transformed[f"{column}__level"] = series
        for hours in delta_hours:
            points = int(hours) * 4
            recent = series.rolling(points, min_periods=points).mean()
            previous = series.shift(points).rolling(points, min_periods=points).mean()
            transformed[f"{column}__delta_{hours}h"] = recent - previous
            transformed[f"{column}__relative_delta_{hours}h"] = (
                (recent - previous) / previous.abs().clip(lower=1e-9)
            )
        for hours in volatility_hours:
            points = int(hours) * 4
            transformed[f"{column}__std_{hours}h"] = series.rolling(
                points, min_periods=points
            ).std(ddof=0)
        baseline_points = int(level_baseline_hours) * 4
        recent_points = 16
        recent_level = series.rolling(recent_points, min_periods=recent_points).mean()
        historical_level = series.rolling(
            baseline_points, min_periods=min(96, baseline_points)
        ).median()
        transformed[f"{column}__level_vs_{level_baseline_hours}h"] = (
            recent_level / historical_level.abs().clip(lower=1e-9) - 1.0
        )
    result = pd.DataFrame(transformed, index=signals.index)
    return result.replace([np.inf, -np.inf], np.nan)


def confirmation_table(
    episodes: pd.DataFrame,
    interval_targets: pd.DataFrame,
    past_hours: int,
    future_hours: Iterable[int],
) -> pd.DataFrame:
    """Describe future target outcomes after each diagnostic episode starts."""
    labels = interval_targets.copy()
    labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
    labels = labels.set_index("datetime").sort_index()
    g1 = pd.to_numeric(labels["generator_1"], errors="coerce")
    records: list[dict[str, object]] = []
    for row in episodes.itertuples(index=False):
        start = pd.Timestamp(row.episode_start)
        past = g1.loc[
            (g1.index >= start - pd.Timedelta(hours=past_hours))
            & (g1.index < start)
        ]
        record: dict[str, object] = {
            "episode_id": row.episode_id,
            "episode_start": start,
            "past_g1_mean": float(past.mean()),
            "past_g1_median": float(past.median()),
        }
        for hours in future_hours:
            future = g1.loc[
                (g1.index >= start)
                & (g1.index < start + pd.Timedelta(hours=int(hours)))
            ]
            future_mean = float(future.mean())
            future_min = float(future.min())
            record[f"future_{hours}h_g1_mean"] = future_mean
            record[f"future_{hours}h_g1_min"] = future_min
            record[f"future_{hours}h_g1_mean_change"] = (
                future_mean - record["past_g1_mean"]
            )
            record[f"future_{hours}h_g1_min_change"] = (
                future_min - record["past_g1_mean"]
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def add_outcome_classes(confirmation: pd.DataFrame, drop_threshold_mw: float) -> pd.DataFrame:
    """Type legacy fuel episodes by their observed generator_1 response."""
    result = confirmation.copy()
    threshold = float(drop_threshold_mw)
    for hours in (2, 6, 12, 24):
        column = f"future_{hours}h_g1_mean_change"
        result[f"g1_drop_{hours}h"] = pd.to_numeric(
            result[column], errors="coerce"
        ) <= threshold
    result["sustained_drop_6h"] = result["g1_drop_2h"] & result["g1_drop_6h"]
    result["sustained_drop_12h"] = (
        result["sustained_drop_6h"] & result["g1_drop_12h"]
    )
    result["rebound_by_24h"] = result["g1_drop_2h"] & (
        pd.to_numeric(result["future_24h_g1_mean_change"], errors="coerce") > 0
    )
    return result


def _rank_auc(values: pd.Series, labels: pd.Series) -> float:
    valid = values.notna() & labels.notna()
    x = pd.to_numeric(values.loc[valid], errors="coerce")
    y = labels.loc[valid].astype(bool)
    valid_numeric = x.notna()
    x = x.loc[valid_numeric]
    y = y.loc[valid_numeric]
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = x.rank(method="average")
    return float(
        (ranks.loc[y].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _leave_one_out_threshold_accuracy(values: pd.Series, labels: pd.Series) -> float:
    frame = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="coerce"), "label": labels.astype(bool)}
    ).dropna()
    predictions: list[bool] = []
    actual: list[bool] = []
    for index in frame.index:
        train = frame.drop(index=index)
        positive = train.loc[train["label"], "value"]
        negative = train.loc[~train["label"], "value"]
        if positive.empty or negative.empty:
            continue
        positive_median = float(positive.median())
        negative_median = float(negative.median())
        threshold = (positive_median + negative_median) / 2
        predict_positive = (
            frame.at[index, "value"] >= threshold
            if positive_median >= negative_median
            else frame.at[index, "value"] <= threshold
        )
        predictions.append(bool(predict_positive))
        actual.append(bool(frame.at[index, "label"]))
    if not actual:
        return np.nan
    prediction_array = np.asarray(predictions, dtype=bool)
    actual_array = np.asarray(actual, dtype=bool)
    true_positive_rate = prediction_array[actual_array].mean() if actual_array.any() else np.nan
    true_negative_rate = (
        (~prediction_array[~actual_array]).mean() if (~actual_array).any() else np.nan
    )
    return float(np.nanmean([true_positive_rate, true_negative_rate]))


def outcome_discrimination_summary(
    per_event: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_column: str,
) -> pd.DataFrame:
    """Assess whether precursors separate true g1 drops from fuel-only events."""
    if outcome_column not in outcomes:
        raise KeyError(f"unknown event outcome: {outcome_column}")
    merged = per_event.merge(
        outcomes[["episode_id", outcome_column]],
        on="episode_id",
        how="inner",
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for (lead, feature), part in merged.groupby(
        ["lead_hours", "feature"], sort=False
    ):
        labels = part[outcome_column].astype(bool)
        values = pd.to_numeric(part["standardized_difference"], errors="coerce")
        positive = values.loc[labels]
        negative = values.loc[~labels]
        auc = _rank_auc(values, labels)
        loo_accuracy = _leave_one_out_threshold_accuracy(values, labels)
        records.append(
            {
                "outcome": outcome_column,
                "lead_hours": int(lead),
                "feature": feature,
                "available_in_formal_test": available_in_formal_test(feature),
                "event_samples": int(part["episode_id"].nunique()),
                "positive_events": int(labels.sum()),
                "negative_events": int((~labels).sum()),
                "positive_median_z": float(positive.median()),
                "negative_median_z": float(negative.median()),
                "rank_auc": auc,
                "rank_separation": abs(auc - 0.5) * 2 if np.isfinite(auc) else np.nan,
                "loo_threshold_balanced_accuracy": loo_accuracy,
                "screening_score": (
                    abs(auc - 0.5) * 2 * loo_accuracy
                    if np.isfinite(auc) and np.isfinite(loo_accuracy)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["lead_hours", "screening_score"], ascending=[True, False]
    ).reset_index(drop=True)


def validation_origins(episodes: pd.DataFrame, lead_hours: Iterable[int]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in episodes.itertuples(index=False):
        for lead in lead_hours:
            records.append(
                {
                    "episode_id": row.episode_id,
                    "episode_start": pd.Timestamp(row.episode_start),
                    "lead_hours": int(lead),
                    "origin": pd.Timestamp(row.episode_start)
                    - pd.Timedelta(hours=int(lead)),
                    "split_group": row.episode_id,
                }
            )
    return pd.DataFrame.from_records(records)


def matched_precursor_audit(
    features: pd.DataFrame,
    episodes: pd.DataFrame,
    origins: pd.DataFrame,
    max_distance_days: int,
    fallback_distance_days: int,
    exclusion_hours: int,
    maximum_controls: int,
    minimum_controls: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each pre-event origin with nearby same-clock non-event controls."""
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    blocked = pd.Series(False, index=features.index)
    exclusion = pd.Timedelta(hours=int(exclusion_hours))
    for row in episodes.itertuples(index=False):
        blocked |= (features.index >= pd.Timestamp(row.episode_start) - exclusion) & (
            features.index <= pd.Timestamp(row.episode_end) + exclusion
        )

    records: list[dict[str, object]] = []
    for origin_row in origins.itertuples(index=False):
        origin = pd.Timestamp(origin_row.origin)
        if origin not in features.index:
            continue
        same_clock = (
            (features.index.hour == origin.hour)
            & (features.index.minute == origin.minute)
            & ~blocked.to_numpy()
            & (features.index != origin)
        )
        distance = np.abs((features.index - origin) / pd.Timedelta(days=1))
        candidates = features.index[same_clock & (distance <= max_distance_days)]
        if len(candidates) < minimum_controls:
            candidates = features.index[
                same_clock & (distance <= fallback_distance_days)
            ]
        candidates = pd.DatetimeIndex(
            sorted(candidates, key=lambda value: abs(value - origin))[:maximum_controls]
        )
        if len(candidates) < minimum_controls:
            continue

        event_values = features.loc[origin]
        controls = features.loc[candidates]
        control_median = controls.median(axis=0, skipna=True)
        control_mad = controls.sub(control_median, axis=1).abs().median(axis=0, skipna=True)
        control_scale = 1.4826 * control_mad
        control_std = controls.std(axis=0, ddof=0, skipna=True)
        control_scale = control_scale.where(control_scale > 1e-12, control_std)
        standardized = (event_values - control_median) / control_scale
        for feature in features.columns:
            event_value = event_values[feature]
            median_value = control_median[feature]
            if not np.isfinite(event_value) or not np.isfinite(median_value):
                continue
            z_value = standardized[feature]
            records.append(
                {
                    "episode_id": origin_row.episode_id,
                    "episode_start": origin_row.episode_start,
                    "lead_hours": int(origin_row.lead_hours),
                    "origin": origin,
                    "feature": feature,
                    "available_in_formal_test": available_in_formal_test(feature),
                    "event_value": float(event_value),
                    "control_median": float(median_value),
                    "paired_difference": float(event_value - median_value),
                    "control_scale": float(control_scale[feature])
                    if np.isfinite(control_scale[feature])
                    else np.nan,
                    "standardized_difference": float(np.clip(z_value, -10.0, 10.0))
                    if np.isfinite(z_value)
                    else np.nan,
                    "control_count": int(len(candidates)),
                }
            )
    per_event = pd.DataFrame.from_records(records)
    if per_event.empty:
        return per_event, pd.DataFrame()

    summary_records: list[dict[str, object]] = []
    for (lead, feature), part in per_event.groupby(
        ["lead_hours", "feature"], sort=False
    ):
        z = pd.to_numeric(part["standardized_difference"], errors="coerce").dropna()
        differences = pd.to_numeric(part["paired_difference"], errors="coerce").dropna()
        positive_fraction = float((differences > 0).mean()) if len(differences) else np.nan
        consistency = (
            max(positive_fraction, 1.0 - positive_fraction)
            if np.isfinite(positive_fraction)
            else np.nan
        )
        median_z = float(z.median()) if len(z) else np.nan
        summary_records.append(
            {
                "lead_hours": int(lead),
                "feature": feature,
                "available_in_formal_test": available_in_formal_test(feature),
                "event_samples": int(part["episode_id"].nunique()),
                "median_event_value": float(part["event_value"].median()),
                "median_control_value": float(part["control_median"].median()),
                "median_paired_difference": float(differences.median())
                if len(differences)
                else np.nan,
                "median_standardized_difference": median_z,
                "positive_fraction": positive_fraction,
                "direction_consistency": consistency,
                "effect_score": abs(median_z) * consistency
                if np.isfinite(median_z) and np.isfinite(consistency)
                else np.nan,
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    return per_event, summary.sort_values(
        ["lead_hours", "effect_score"], ascending=[True, False]
    ).reset_index(drop=True)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _markdown_report(
    run_id: str,
    train_days: pd.DataFrame,
    train_episodes: pd.DataFrame,
    test_days: pd.DataFrame,
    confirmation: pd.DataFrame,
    summary: pd.DataFrame,
    outcome_summary: pd.DataFrame,
    primary_outcome: str,
) -> str:
    lines = [
        "# 台阶事件与因果前兆审计",
        "",
        f"- run_id: `{run_id}`",
        "- 角色：diagnostic；不是模型分数，也不是提交候选。",
        f"- 训练期 legacy 事件日：{len(train_days)} 天。",
        f"- 连续日期合并后的独立事件段：{len(train_episodes)} 段。",
        f"- 测试期同口径事件日：{len(test_days)} 天。",
        "",
        "## 关键口径",
        "",
        "事件定义严格复现学弟异常日权重口径：用训练期两列机组耗气线性反推 generator_all，",
        "取 16 点滚动均值除以 672 点滚动基线，ratio < 0.8 记为事件日。",
        "该定义使用全训练期拟合，仅用于诊断和建立事件分组；未来正式验证必须在训练折内重新拟合。",
        "",
        "前兆特征只使用审计起点及以前的观测。目标历史使用上一完整 15 分钟区间，",
        "没有把起点后的区间均值混入特征。",
        "",
        "## 训练事件段",
        "",
        "| episode | start | end | event days | min ratio |",
        "|---|---|---|---:|---:|",
    ]
    for row in train_episodes.itertuples(index=False):
        lines.append(
            f"| {row.episode_id} | {row.episode_start} | {row.episode_end} | "
            f"{row.event_day_count} | {row.min_ratio:.4f} |"
        )

    lines.extend(["", "## generator_1 结果确认", ""])
    if confirmation.empty:
        lines.append("没有可用事件结果。")
    else:
        for hours in (2, 6, 12, 24):
            column = f"future_{hours}h_g1_mean_change"
            values = pd.to_numeric(confirmation[column], errors="coerce")
            lines.append(
                f"- 事件开始后 {hours}h 的 g1 均值相对事件前 4h："
                f"中位变化 {values.median():.3f} MW；下降事件占比 {(values < 0).mean():.1%}。"
            )

    lines.extend(
        [
            "",
            "## 匹配对照下的高效应候选",
            "",
            "以下是描述性排序，不是可直接采用的模型特征重要性。每个事件起点与附近、",
            "相同时刻且远离事件段的普通起点匹配。只有跨事件一致且在折内复核后才可进入模型。",
            "",
        ]
    )
    if not confirmation.empty:
        lines.extend(
            [
                "## 事件分型",
                "",
                f"以未来均值相对事件前 4h 下降至少 5 MW 为阈值：",
                f"2h 下降 {int(confirmation['g1_drop_2h'].sum())}/{len(confirmation)}，",
                f"6h 下降 {int(confirmation['g1_drop_6h'].sum())}/{len(confirmation)}，",
                f"12h 下降 {int(confirmation['g1_drop_12h'].sum())}/{len(confirmation)}，",
                f"24h 下降 {int(confirmation['g1_drop_24h'].sum())}/{len(confirmation)}。",
                f"同时满足 2h 和 6h 下降的 `{primary_outcome}` 仅有 "
                f"{int(confirmation[primary_outcome].sum())}/{len(confirmation)}。",
                "",
                "这说明 legacy 低燃料事件只能作为候选母集，不能直接作为 long-g1 正标签。",
                "",
            ]
        )
    for lead in sorted(summary["lead_hours"].unique()) if not summary.empty else []:
        lines.extend(
            [
                f"### 提前 {lead} 小时",
                "",
                "| feature | events | median z | direction consistency |",
                "|---|---:|---:|---:|",
            ]
        )
        part = summary.loc[
            (summary["lead_hours"] == lead)
            & (summary["event_samples"] >= 8)
            & summary["available_in_formal_test"]
        ].head(10)
        for row in part.itertuples(index=False):
            lines.append(
                f"| `{row.feature}` | {row.event_samples} | "
                f"{row.median_standardized_difference:.3f} | "
                f"{row.direction_consistency:.1%} |"
            )
        lines.append("")

    lines.extend(
        [
            f"## `{primary_outcome}` 分型筛选",
            "",
            "该筛选只有 13 个 episode，且在 224 个特征上做了探索，存在严重多重比较风险。",
            "LOO 阈值准确率只是淘汰指标，不能作为泛化证明。",
            "",
        ]
    )
    for lead in (
        sorted(outcome_summary["lead_hours"].unique())
        if not outcome_summary.empty
        else []
    ):
        lines.extend(
            [
                f"### 提前 {lead} 小时",
                "",
                "| feature | positive/negative | rank AUC | LOO balanced accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        part = outcome_summary.loc[
            (outcome_summary["lead_hours"] == lead)
            & outcome_summary["available_in_formal_test"]
        ].head(8)
        for row in part.itertuples(index=False):
            lines.append(
                f"| `{row.feature}` | {row.positive_events}/{row.negative_events} | "
                f"{row.rank_auc:.3f} | {row.loo_threshold_balanced_accuracy:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 使用限制",
            "",
            "- 24 个事件日并非 24 个独立样本；相邻事件日必须按 episode 整体留出。",
            "- 事件定义由 generator_all 的燃料代理触发，不能直接等同于 generator_1 必然下台阶。",
            "- 匹配对照和效应量只用于 Phase A 筛选；不能作为 65 分增益证据。",
            "- 历史真实 g1/gall 在正式测试期不可得，相关特征保留为 diagnostic-only，未进入候选排名。",
            "- 测试期事件表只确认观测到的低燃料状态，不提供未知测试目标，也不能反推未来真值。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    event_config = config["event_definition"]
    precursor_config = config["precursor_audit"]

    train_base_path = BASE_DIR / "train_causal_base_1min.csv"
    test_base_path = BASE_DIR / "test_causal_base_1min.csv"
    required_covariates = sorted(
        set(
            BF_PRODUCTION
            + AIR_HEATERS
            + BF_USERS
            + CONVERTER_USERS
            + GENERATOR_FUELS
            + MIXED_GAS
            + [
                "coke_oven_1",
                "converter_1",
                "blast_furnace_gas_holder_2",
            ]
        )
    )
    train_base, train_encoding = read_csv_strict(
        train_base_path, usecols=["datetime", *required_covariates]
    )
    test_base, test_encoding = read_csv_strict(
        test_base_path, usecols=["datetime", *required_covariates]
    )
    raw_load, load_encoding = read_csv_strict(
        RAW_LOAD_PATH, usecols=["datetime", "generator_all"]
    )
    labels, label_encoding = read_csv_strict(LABEL_PATH)
    labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")

    train_grid = _quarter_hour_grid(train_base, required_covariates)
    test_grid = _quarter_hour_grid(test_base, required_covariates)
    raw_target_grid = _quarter_hour_grid(raw_load, ["generator_all"])
    train_fit_grid = train_grid.merge(
        raw_target_grid, on="datetime", how="left", validate="one_to_one"
    )
    fuel_columns = list(event_config["generator_all_fuel_columns"])
    coefficients = fit_fuel_proxy(train_fit_grid, fuel_columns)

    combined_grid = pd.concat([train_grid, test_grid], ignore_index=True)
    combined_ratio = legacy_fuel_ratio(
        combined_grid,
        fuel_columns,
        coefficients,
        int(event_config["short_rolling_points"]),
        int(event_config["baseline_rolling_points"]),
        int(event_config["baseline_minimum_points"]),
    )
    train_end = pd.Timestamp(train_grid["datetime"].max())
    train_ratio = combined_ratio.loc[combined_ratio.index <= train_end]
    test_ratio = combined_ratio.loc[combined_ratio.index > train_end]
    threshold = float(event_config["threshold"])
    train_days = event_day_catalog(train_ratio, threshold, "train")
    test_days = event_day_catalog(test_ratio, threshold, "test")
    expected_days = int(event_config["expected_train_event_days"])
    if len(train_days) != expected_days:
        raise AssertionError(
            f"legacy event-day replication failed: {len(train_days)} != {expected_days}"
        )
    train_episodes = merge_event_days(train_days, "train")
    test_episodes = merge_event_days(test_days, "test")

    signals = build_observed_signals(train_grid, labels)
    precursor_features = build_causal_transforms(
        signals,
        precursor_config["delta_hours"],
        precursor_config["volatility_hours"],
        int(precursor_config["level_baseline_hours"]),
    )
    origins = validation_origins(train_episodes, precursor_config["lead_hours"])
    per_event, precursor_summary = matched_precursor_audit(
        precursor_features,
        train_episodes,
        origins,
        int(precursor_config["matched_control_max_distance_days"]),
        int(precursor_config["matched_control_fallback_distance_days"]),
        int(precursor_config["event_exclusion_hours"]),
        int(precursor_config["maximum_controls_per_event"]),
        int(precursor_config["minimum_controls_per_event"]),
    )
    confirmation = confirmation_table(
        train_episodes,
        labels,
        int(config["label_confirmation"]["past_reference_hours"]),
        config["label_confirmation"]["future_hours"],
    )
    confirmation = add_outcome_classes(
        confirmation,
        float(config["label_confirmation"]["drop_threshold_mw"]),
    )
    primary_outcome = str(config["label_confirmation"]["primary_outcome"])
    outcome_summary = outcome_discrimination_summary(
        per_event,
        confirmation,
        primary_outcome,
    )

    fingerprint = sha256(config_path)[:10]
    run_id = f"{started_at.strftime('%Y%m%dT%H%M%SZ')}_step_event_audit_{fingerprint}"
    output_dir = PROJECT_ROOT / config["output"]["root"] / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    float_format = config["output"]["float_format"]
    outputs = {
        "train_event_days": train_days,
        "train_event_episodes": train_episodes,
        "test_event_days": test_days,
        "test_event_episodes": test_episodes,
        "event_validation_origins": origins,
        "event_target_confirmation": confirmation,
        "per_event_precursors": per_event,
        "precursor_summary": precursor_summary,
        "outcome_discrimination_summary": outcome_summary,
    }
    output_paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(
            path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d %H:%M:%S",
            float_format=float_format,
        )
        output_paths[name] = path

    report_path = output_dir / "report.md"
    report_path.write_text(
        _markdown_report(
            run_id,
            train_days,
            train_episodes,
            test_days,
            confirmation,
            precursor_summary,
            outcome_summary,
            primary_outcome,
        ),
        encoding="utf-8",
    )
    output_paths["report"] = report_path

    git_commit, git_dirty = _git_state()
    ended_at = datetime.now(timezone.utc)
    manifest = {
        "run_id": run_id,
        "status": "completed",
        "experiment_role": config["experiment_role"],
        "diagnostic_version": config["diagnostic_version"],
        "control_artifact_id": config["control_artifact_id"],
        "hypothesis": config["hypothesis"],
        "command": (
            "python -m src.round2_v3.audit_step_events "
            f"--config {config_path.relative_to(PROJECT_ROOT).as_posix()}"
        ),
        "working_directory": str(PROJECT_ROOT),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "holdout_evaluated": False,
        "submission_eligible": False,
        "test_targets_used": False,
        "future_covariates_used_as_precursors": False,
        "event_definition_uses_full_train_fit_for_diagnostic_only": True,
        "formal_validation_requirement": (
            "refit event definition and all learned thresholds inside each training fold"
        ),
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "input_files": {
            "train_causal_base": {
                "path": train_base_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(train_base_path),
                "encoding": train_encoding,
            },
            "test_causal_base": {
                "path": test_base_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(test_base_path),
                "encoding": test_encoding,
            },
            "raw_train_load": {
                "path": RAW_LOAD_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(RAW_LOAD_PATH),
                "encoding": load_encoding,
            },
            "interval_targets": {
                "path": LABEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(LABEL_PATH),
                "encoding": label_encoding,
            },
        },
        "fuel_proxy_coefficients": coefficients.tolist(),
        "event_counts": {
            "train_event_days": int(len(train_days)),
            "train_independent_episodes": int(len(train_episodes)),
            "test_event_days": int(len(test_days)),
            "test_independent_episodes": int(len(test_episodes)),
        },
        "precursor_feature_count": int(precursor_features.shape[1]),
        "formal_test_available_precursor_feature_count": int(
            sum(available_in_formal_test(name) for name in precursor_features.columns)
        ),
        "validation_origin_count": int(len(origins)),
        "primary_event_outcome": primary_outcome,
        "primary_event_outcome_positive_count": int(
            confirmation[primary_outcome].sum()
        ),
        "outputs": {
            name: {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in output_paths.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        "PASS "
        f"run_id={run_id} train_event_days={len(train_days)} "
        f"train_episodes={len(train_episodes)} test_event_days={len(test_days)} "
        f"features={precursor_features.shape[1]} output={output_dir}"
    )


if __name__ == "__main__":
    main()
