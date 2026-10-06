"""Causal helpers for the fold-local holder-2 boundary diagnostic."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


LEVEL = "holder_2__level"
DELTAS = ("holder_2__delta_1h", "holder_2__delta_4h", "holder_2__delta_12h")
BALANCE = "balance_proxy_total__level"
REQUIRED_COLUMNS = (LEVEL, *DELTAS, BALANCE)


def _finite_numeric(series: pd.Series, name: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"{name} has no finite values")
    return values


def _iqr(values: np.ndarray, floor: float) -> float:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return float(max(q75 - q25, floor))


def fit_boundary_contract(
    history: pd.DataFrame,
    *,
    reference_quantiles: Iterable[float],
    zone_quantiles: Iterable[float],
    scale_floor: float,
) -> dict[str, float]:
    """Fit all thresholds and robust scales using training history only."""
    missing = set(REQUIRED_COLUMNS) - set(history.columns)
    if missing:
        raise KeyError(f"boundary history missing columns: {sorted(missing)}")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    ref_low_q, ref_high_q = [float(value) for value in reference_quantiles]
    zone_low_q, zone_high_q = [float(value) for value in zone_quantiles]
    if not (0 <= ref_low_q < zone_low_q < zone_high_q < ref_high_q <= 1):
        raise ValueError("boundary quantiles must be strictly ordered")

    level = _finite_numeric(history[LEVEL], LEVEL)
    ref_low, zone_low, zone_high, ref_high = np.quantile(
        level, [ref_low_q, zone_low_q, zone_high_q, ref_high_q]
    )
    span = float(ref_high - ref_low)
    if not np.isfinite(span) or span <= scale_floor:
        raise ValueError("holder reference span is degenerate")
    contract: dict[str, float] = {
        "reference_low": float(ref_low),
        "zone_low": float(zone_low),
        "zone_high": float(zone_high),
        "reference_high": float(ref_high),
        "reference_span": span,
        "level_median": float(np.median(level)),
    }
    for column in DELTAS:
        values = _finite_numeric(history[column], column)
        contract[f"{column}_scale"] = _iqr(values, scale_floor)
    balance = _finite_numeric(history[BALANCE], BALANCE)
    contract["balance_median"] = float(np.median(balance))
    contract["balance_scale"] = _iqr(balance, scale_floor)
    return contract


def build_boundary_features(
    frame: pd.DataFrame,
    contract: dict[str, float],
    *,
    maximum_dwell_steps: int,
) -> pd.DataFrame:
    """Build causal boundary features; row t depends only on rows up to t."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise KeyError(f"boundary frame missing columns: {sorted(missing)}")
    if maximum_dwell_steps <= 0:
        raise ValueError("maximum_dwell_steps must be positive")
    result = pd.DataFrame(index=frame.index)
    level = pd.to_numeric(frame[LEVEL], errors="coerce").astype(float)
    level = level.fillna(float(contract["level_median"]))
    span = float(contract["reference_span"])
    position = (level - float(contract["reference_low"])) / span
    result["holder_position"] = position
    result["distance_to_low"] = position
    result["distance_to_high"] = 1.0 - position
    result["low_zone"] = (level <= float(contract["zone_low"])).astype(int)
    result["high_zone"] = (level >= float(contract["zone_high"])).astype(int)

    high_dwell: list[int] = []
    low_dwell: list[int] = []
    high_count = low_count = 0
    for is_high, is_low in zip(result["high_zone"], result["low_zone"], strict=True):
        high_count = min(high_count + 1, maximum_dwell_steps) if is_high else 0
        low_count = min(low_count + 1, maximum_dwell_steps) if is_low else 0
        high_dwell.append(high_count)
        low_dwell.append(low_count)
    result["high_dwell_fraction"] = np.asarray(high_dwell, dtype=float) / maximum_dwell_steps
    result["low_dwell_fraction"] = np.asarray(low_dwell, dtype=float) / maximum_dwell_steps
    result["signed_dwell"] = result["high_dwell_fraction"] - result["low_dwell_fraction"]

    velocity_parts: list[np.ndarray] = []
    for column in DELTAS:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scaled = np.clip(values / float(contract[f"{column}_scale"]), -3.0, 3.0) / 3.0
        name = column.replace("holder_2__delta_", "velocity_")
        result[name] = scaled
        velocity_parts.append(scaled)
    result["velocity_mean"] = np.mean(np.column_stack(velocity_parts), axis=1)

    balance = pd.to_numeric(frame[BALANCE], errors="coerce").fillna(
        float(contract["balance_median"])
    )
    result["balance_state"] = np.clip(
        (balance.to_numpy(dtype=float) - float(contract["balance_median"]))
        / float(contract["balance_scale"]),
        -3.0,
        3.0,
    ) / 3.0
    centered_position = 2.0 * np.clip(position.to_numpy(dtype=float), 0.0, 1.0) - 1.0
    result["boundary_pressure_score"] = (
        0.45 * centered_position
        + 0.25 * result["velocity_mean"].to_numpy(dtype=float)
        + 0.20 * result["signed_dwell"].to_numpy(dtype=float)
        + 0.10 * result["balance_state"].to_numpy(dtype=float)
    )
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("boundary feature construction produced non-finite values")
    return result


def spearman_rank_correlation(left: object, right: object) -> float:
    """Return Spearman correlation without an optional scipy dependency."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x_rank = pd.Series(x[valid]).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y[valid]).rank(method="average").to_numpy(dtype=float)
    if np.isclose(x_rank.std(), 0.0) or np.isclose(y_rank.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def origin_residual_outcomes(
    control_oof: pd.DataFrame,
    *,
    near_bucket: str,
    far_bucket: str,
) -> pd.DataFrame:
    """Aggregate per-cell control residuals into one row per (fold, origin)."""
    required = {"fold_id", "datetime", "horizon_bucket", "actual", "prediction"}
    missing = required - set(control_oof.columns)
    if missing:
        raise KeyError(f"control OOF missing columns: {sorted(missing)}")
    frame = control_oof.copy()
    frame["actual"] = pd.to_numeric(frame["actual"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    valid = np.isfinite(frame["actual"]) & np.isfinite(frame["prediction"]) & (frame["actual"] != 0)
    frame = frame.loc[valid].copy()
    frame["signed_percentage_residual"] = (frame["actual"] - frame["prediction"]) / frame["actual"].abs()
    keys = ["fold_id", "datetime"]
    residual = (
        frame.groupby(keys + ["horizon_bucket"], sort=False)["signed_percentage_residual"]
        .mean()
        .unstack("horizon_bucket")
    )
    actual = (
        frame.groupby(keys + ["horizon_bucket"], sort=False)["actual"]
        .mean()
        .unstack("horizon_bucket")
    )
    for bucket in (near_bucket, far_bucket):
        if bucket not in residual or bucket not in actual:
            raise AssertionError(f"control OOF lacks horizon bucket {bucket}")
    result = pd.DataFrame(index=residual.index)
    result["near_signed_percentage_residual"] = residual[near_bucket]
    result["far_signed_percentage_residual"] = residual[far_bucket]
    result["near_actual_mean"] = actual[near_bucket]
    result["far_actual_mean"] = actual[far_bucket]
    result["far_minus_near_actual_ratio"] = (
        (result["far_actual_mean"] - result["near_actual_mean"])
        / result["near_actual_mean"].abs().clip(lower=1e-6)
    )
    return result.reset_index()


def check_fresh_block_layout(
    blocks: Iterable[dict[str, object]],
    dev_blocks: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Assert fresh pseudo-test blocks are complete, contiguous and uncontaminated."""
    validated: list[dict[str, object]] = []
    seen: set[pd.Timestamp] = set()
    for block in blocks:
        start = pd.Timestamp(block["validation_start"])
        end = pd.Timestamp(block["validation_end"])
        origins = pd.date_range(start, end, freq="15min")
        if len(origins) != 960:
            raise ValueError(f"{block['fold_id']} is not a complete ten-day block")
        if any(origin in seen for origin in origins):
            raise ValueError(f"{block['fold_id']} overlaps another fresh block")
        seen.update(origins)
        validated.append({**block, "start": start, "end": end, "origins": origins})
    dev_seen: set[pd.Timestamp] = set()
    for block in dev_blocks:
        dev_seen.update(pd.date_range(block["validation_start"], block["validation_end"], freq="15min"))
    contaminated = seen & dev_seen
    if contaminated:
        example = min(contaminated).isoformat()
        raise ValueError(f"fresh blocks overlap development blocks, e.g. {example}")
    return validated


def summarize_pressure_relationships(
    per_origin: pd.DataFrame,
    *,
    block_ids: Iterable[str],
) -> pd.DataFrame:
    """Summarize the preregistered pressure-to-residual relationship."""
    rows: list[dict[str, object]] = []
    scopes = [("pooled", per_origin)] + [
        (str(block), per_origin.loc[per_origin["fold_id"] == block]) for block in block_ids
    ]
    for scope, part in scopes:
        high = part.loc[part["score_group"] == "high"]
        low = part.loc[part["score_group"] == "low"]
        rows.append(
            {
                "scope": scope,
                "origin_count": len(part),
                "pressure_residual_spearman": spearman_rank_correlation(
                    part["boundary_pressure_score"], part["far_signed_percentage_residual"]
                ),
                "pressure_target_change_spearman": spearman_rank_correlation(
                    part["boundary_pressure_score"], part["far_minus_near_actual_ratio"]
                ),
                "high_score_fraction": float((part["score_group"] == "high").mean()),
                "low_score_fraction": float((part["score_group"] == "low").mean()),
                "boundary_zone_fraction": float(
                    ((part["high_zone"] == 1) | (part["low_zone"] == 1)).mean()
                ),
                "high_minus_low_far_residual_pct": float(
                    (high["far_signed_percentage_residual"].mean()
                    - low["far_signed_percentage_residual"].mean())
                    * 100.0
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def evaluate_inverse_gate(
    summary: pd.DataFrame,
    gate: dict[str, object],
) -> pd.DataFrame:
    """Apply the preregistered INVERSE boundary-pressure gate to a fresh-block summary.

    The hypothesis under test is the sign-flipped relationship observed post hoc on
    development blocks: high boundary pressure must predict positive far residuals
    of the control (control over-predicts). Every threshold is an upper bound.
    """
    pooled = summary.loc[summary["scope"] == "pooled"].iloc[0]
    blocks = summary.loc[summary["scope"] != "pooled"].copy()
    negative_rho_fraction = float((blocks["pressure_residual_spearman"] < 0).mean())
    negative_spread_fraction = float((blocks["high_minus_low_far_residual_pct"] < 0).mean())
    negative_change_fraction = float((blocks["pressure_target_change_spearman"] < 0).mean())
    min_extreme_fraction = float(
        blocks[["high_score_fraction", "low_score_fraction"]].to_numpy(dtype=float).min()
    )
    pass_rho = float(pooled["pressure_residual_spearman"]) <= float(
        gate["maximum_pooled_far_residual_spearman"]
    )
    pass_block_direction = negative_rho_fraction >= 1.0 - float(
        gate["maximum_positive_block_fraction"]
    )
    pass_spread = float(pooled["high_minus_low_far_residual_pct"]) <= float(
        gate["maximum_pooled_high_low_far_residual_spread_pct"]
    )
    pass_spread_blocks = negative_spread_fraction >= 1.0 - float(
        gate["maximum_positive_spread_block_fraction"]
    )
    pass_change = float(pooled["pressure_target_change_spearman"]) <= float(
        gate["maximum_pooled_target_change_spearman"]
    )
    pass_change_blocks = negative_change_fraction >= 1.0 - float(
        gate["maximum_positive_target_change_block_fraction"]
    )
    pass_coverage = min_extreme_fraction >= float(
        gate["minimum_extreme_group_fraction_per_block"]
    )
    passed = bool(
        pass_rho
        and pass_block_direction
        and pass_spread
        and pass_spread_blocks
        and pass_change
        and pass_change_blocks
        and pass_coverage
    )
    return pd.DataFrame(
        [
            {
                "pooled_far_residual_spearman": pooled["pressure_residual_spearman"],
                "negative_rho_block_fraction": negative_rho_fraction,
                "pooled_high_low_far_residual_spread_pct": pooled["high_minus_low_far_residual_pct"],
                "negative_spread_block_fraction": negative_spread_fraction,
                "pooled_target_change_spearman": pooled["pressure_target_change_spearman"],
                "negative_change_block_fraction": negative_change_fraction,
                "minimum_extreme_group_fraction": min_extreme_fraction,
                "pass_pooled_rho_inverse": pass_rho,
                "pass_block_direction_inverse": pass_block_direction,
                "pass_pooled_spread_inverse": pass_spread,
                "pass_spread_blocks_inverse": pass_spread_blocks,
                "pass_mechanism_target_change": pass_change,
                "pass_mechanism_blocks": pass_change_blocks,
                "pass_coverage": pass_coverage,
                "inverse_diagnostic_gate_passed": passed,
                "model_experiment_allowed": bool(passed and gate["model_experiment_allowed_on_pass"]),
                "platform_submission_eligible": False,
            }
        ]
    )
