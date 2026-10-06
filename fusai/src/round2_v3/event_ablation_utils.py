"""Evaluation helpers for the long-g1 event-process feature ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.round2_v3.metrics import regression_metrics


def horizon_bucket(minutes: pd.Series) -> pd.Series:
    values = pd.to_numeric(minutes, errors="raise")
    return pd.cut(
        values,
        bins=[0, 120, 360, 720, 1440],
        labels=["h015_120", "h135_360", "h375_720", "h735_1440"],
        include_lowest=True,
    ).astype(str)


def attach_event_context(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    context_hours: int = 24,
) -> pd.DataFrame:
    """Attach evaluation-only event labels without changing model features."""
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="raise")
    result["interval_start"] = pd.to_datetime(
        result["interval_start"], errors="raise"
    )
    result["origin_regime"] = "ordinary"
    result["origin_episode_id"] = ""
    result["target_in_event"] = False
    result["target_episode_id"] = ""
    context = pd.Timedelta(hours=int(context_hours))
    ordered = episodes.copy()
    ordered["episode_start"] = pd.to_datetime(
        ordered["episode_start"], errors="raise"
    )
    ordered["episode_end"] = pd.to_datetime(ordered["episode_end"], errors="raise")
    for row in ordered.itertuples(index=False):
        start = pd.Timestamp(row.episode_start)
        end = pd.Timestamp(row.episode_end)
        pre = (result["datetime"] >= start - context) & (result["datetime"] < start)
        active = (result["datetime"] >= start) & (result["datetime"] <= end)
        recovery = (result["datetime"] > end) & (result["datetime"] <= end + context)
        result.loc[pre, "origin_regime"] = "pre_event"
        result.loc[active, "origin_regime"] = "active_event"
        result.loc[recovery, "origin_regime"] = "recovery"
        result.loc[pre | active | recovery, "origin_episode_id"] = row.episode_id
        target = (result["interval_start"] >= start) & (
            result["interval_start"] <= end
        )
        result.loc[target, "target_in_event"] = True
        result.loc[target, "target_episode_id"] = row.episode_id
    result["horizon_bucket"] = horizon_bucket(result["horizon_minutes"])
    return result


def summarize_ablation(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant",
        "fold_id",
        "actual",
        "prediction",
        "horizon_bucket",
        "origin_regime",
        "target_in_event",
        "target_episode_id",
    }
    if not required <= set(predictions.columns):
        raise KeyError(f"ablation predictions missing: {sorted(required - set(predictions.columns))}")
    rows: list[dict[str, object]] = []

    def add(scope: str, group_columns: list[str], source: pd.DataFrame) -> None:
        for keys, part in source.groupby(group_columns, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            identity = dict(zip(group_columns, keys, strict=True))
            rows.append(
                {
                    "scope": scope,
                    "variant": identity.get("variant", "all"),
                    "fold_id": identity.get("fold_id", "all"),
                    "horizon_bucket": identity.get("horizon_bucket", "all"),
                    "origin_regime": identity.get("origin_regime", "all"),
                    "target_in_event": identity.get("target_in_event", "all"),
                    "episode_id": identity.get("target_episode_id", "all"),
                    **regression_metrics(part["actual"], part["prediction"]),
                }
            )

    add("overall", ["variant"], predictions)
    add("fold", ["variant", "fold_id"], predictions)
    add("horizon_bucket", ["variant", "horizon_bucket"], predictions)
    add("origin_regime", ["variant", "origin_regime"], predictions)
    add("target_event", ["variant", "target_in_event"], predictions)
    event_rows = predictions.loc[predictions["target_in_event"]].copy()
    if not event_rows.empty:
        add("episode", ["variant", "target_episode_id"], event_rows)
    return pd.DataFrame.from_records(rows)


def gate_results(
    metrics: pd.DataFrame,
    control_variant: str,
    candidate_variants: list[str],
    gate: dict,
    recent_folds: list[str],
) -> pd.DataFrame:
    """Evaluate pre-registered promotion gates against the feature-off control."""

    def accuracy(
        variant: str,
        scope: str,
        *,
        fold_id: str = "all",
        horizon: str = "all",
        regime: str = "all",
    ) -> float:
        selected = metrics.loc[
            (metrics["variant"] == variant)
            & (metrics["scope"] == scope)
            & (metrics["fold_id"].astype(str) == str(fold_id))
            & (metrics["horizon_bucket"].astype(str) == str(horizon))
            & (metrics["origin_regime"].astype(str) == str(regime))
        ]
        return np.nan if selected.empty else float(selected.iloc[0]["accuracy_1_minus_mape"])

    control_overall = accuracy(control_variant, "overall")
    control_ordinary = accuracy(control_variant, "origin_regime", regime="ordinary")
    records: list[dict[str, object]] = []
    for variant in candidate_variants:
        overall_gain = (accuracy(variant, "overall") - control_overall) * 100
        ordinary_loss = (
            control_ordinary - accuracy(variant, "origin_regime", regime="ordinary")
        ) * 100
        recent_deltas = [
            (accuracy(variant, "fold", fold_id=fold) - accuracy(control_variant, "fold", fold_id=fold))
            * 100
            for fold in recent_folds
        ]
        recent_deltas = [value for value in recent_deltas if np.isfinite(value)]
        bucket_deltas = {}
        for bucket in ("h015_120", "h135_360", "h375_720", "h735_1440"):
            bucket_deltas[bucket] = (
                accuracy(variant, "horizon_bucket", horizon=bucket)
                - accuracy(control_variant, "horizon_bucket", horizon=bucket)
            ) * 100

        event_control = metrics.loc[
            (metrics["scope"] == "episode")
            & (metrics["variant"] == control_variant),
            ["episode_id", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "control_accuracy"})
        event_candidate = metrics.loc[
            (metrics["scope"] == "episode") & (metrics["variant"] == variant),
            ["episode_id", "accuracy_1_minus_mape"],
        ].rename(columns={"accuracy_1_minus_mape": "candidate_accuracy"})
        episode_comparison = event_control.merge(
            event_candidate, on="episode_id", how="inner", validate="one_to_one"
        )
        episode_fraction = (
            float(
                (
                    episode_comparison["candidate_accuracy"]
                    > episode_comparison["control_accuracy"]
                ).mean()
            )
            if not episode_comparison.empty
            else np.nan
        )
        recent_mean = float(np.mean(recent_deltas)) if recent_deltas else np.nan
        near_far_minimum = float(
            np.nanmin([bucket_deltas["h015_120"], bucket_deltas["h735_1440"]])
        )
        pass_overall = overall_gain >= float(
            gate["minimum_long_g1_accuracy_gain_pct"]
        )
        pass_ordinary = ordinary_loss <= float(
            gate["maximum_ordinary_origin_accuracy_loss_pct"]
        )
        pass_recent = (not gate["require_recent_fold_non_degradation"]) or (
            np.isfinite(recent_mean) and recent_mean >= 0
        )
        pass_episodes = (not gate["require_majority_episode_improvement"]) or (
            np.isfinite(episode_fraction) and episode_fraction > 0.5
        )
        pass_near_far = (not gate["require_near_far_safety_check"]) or (
            near_far_minimum
            >= -float(gate["maximum_ordinary_origin_accuracy_loss_pct"])
        )
        records.append(
            {
                "variant": variant,
                "overall_accuracy_gain_pct": overall_gain,
                "ordinary_accuracy_loss_pct": ordinary_loss,
                "recent_fold_mean_gain_pct": recent_mean,
                "episode_improvement_fraction": episode_fraction,
                "evaluated_episode_count": int(len(episode_comparison)),
                "near_gain_pct": bucket_deltas["h015_120"],
                "mid1_gain_pct": bucket_deltas["h135_360"],
                "mid2_gain_pct": bucket_deltas["h375_720"],
                "far_gain_pct": bucket_deltas["h735_1440"],
                "pass_overall": pass_overall,
                "pass_ordinary": pass_ordinary,
                "pass_recent": pass_recent,
                "pass_episodes": pass_episodes,
                "pass_near_far": pass_near_far,
                "promotion_gate_passed": bool(
                    pass_overall
                    and pass_ordinary
                    and pass_recent
                    and pass_episodes
                    and pass_near_far
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        "overall_accuracy_gain_pct", ascending=False
    ).reset_index(drop=True)
