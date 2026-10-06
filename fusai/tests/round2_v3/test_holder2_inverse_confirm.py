import numpy as np
import pandas as pd
import pytest

from src.round2_v3.holder_boundary_utils import (
    check_fresh_block_layout,
    evaluate_inverse_gate,
    origin_residual_outcomes,
)


def _summary_row(scope: str, rho: float, spread: float, change: float, hi: float, lo: float) -> dict:
    return {
        "scope": scope,
        "origin_count": 960,
        "pressure_residual_spearman": rho,
        "pressure_target_change_spearman": change,
        "high_score_fraction": hi,
        "low_score_fraction": lo,
        "boundary_zone_fraction": 0.2,
        "high_minus_low_far_residual_pct": spread,
    }


def _gate_config() -> dict:
    return {
        "maximum_pooled_far_residual_spearman": -0.08,
        "maximum_positive_block_fraction": 0.25,
        "maximum_pooled_high_low_far_residual_spread_pct": -1.0,
        "maximum_positive_spread_block_fraction": 0.25,
        "maximum_pooled_target_change_spearman": -0.05,
        "maximum_positive_target_change_block_fraction": 0.25,
        "minimum_extreme_group_fraction_per_block": 0.05,
        "model_experiment_allowed_on_pass": True,
        "platform_submission_on_pass": False,
    }


def test_inverse_gate_passes_on_consistently_negative_relationship() -> None:
    rows = [_summary_row("pooled", -0.20, -4.0, -0.30, 0.15, 0.40)]
    for index in range(4):
        rows.append(
            _summary_row(f"fresh_{index}", -0.10 - 0.05 * index, -2.0 - index, -0.10, 0.10, 0.35)
        )
    gate = evaluate_inverse_gate(pd.DataFrame.from_records(rows), _gate_config()).iloc[0]
    assert bool(gate["inverse_diagnostic_gate_passed"]) is True
    assert bool(gate["model_experiment_allowed"]) is True
    assert bool(gate["pass_pooled_rho_inverse"]) is True
    assert bool(gate["pass_mechanism_target_change"]) is True


def test_inverse_gate_rejects_each_violation() -> None:
    base = [_summary_row("pooled", -0.20, -4.0, -0.30, 0.15, 0.40)] + [
        _summary_row(f"fresh_{index}", -0.15, -3.0, -0.10, 0.10, 0.35) for index in range(4)
    ]
    assert bool(evaluate_inverse_gate(pd.DataFrame.from_records(base), _gate_config()).iloc[0]["inverse_diagnostic_gate_passed"]) is True

    wrong_rho = [dict(base[0], pressure_residual_spearman=+0.05)] + base[1:]
    assert bool(evaluate_inverse_gate(pd.DataFrame.from_records(wrong_rho), _gate_config()).iloc[0]["inverse_diagnostic_gate_passed"]) is False

    wrong_spread = [dict(base[0], high_minus_low_far_residual_pct=+0.5)] + base[1:]
    assert bool(evaluate_inverse_gate(pd.DataFrame.from_records(wrong_spread), _gate_config()).iloc[0]["inverse_diagnostic_gate_passed"]) is False

    weak_mechanism = [dict(base[0], pressure_target_change_spearman=-0.01)] + base[1:]
    assert bool(evaluate_inverse_gate(pd.DataFrame.from_records(weak_mechanism), _gate_config()).iloc[0]["inverse_diagnostic_gate_passed"]) is False

    mixed_blocks = [
        _summary_row("pooled", -0.20, -4.0, -0.30, 0.15, 0.40),
        _summary_row("fresh_0", +0.20, +3.0, +0.10, 0.10, 0.35),
        _summary_row("fresh_1", +0.15, +2.0, +0.10, 0.10, 0.35),
        _summary_row("fresh_2", -0.15, -3.0, -0.10, 0.10, 0.35),
        _summary_row("fresh_3", -0.15, -3.0, -0.10, 0.10, 0.35),
    ]
    gate = evaluate_inverse_gate(pd.DataFrame.from_records(mixed_blocks), _gate_config()).iloc[0]
    assert bool(gate["pass_block_direction_inverse"]) is False
    assert bool(gate["inverse_diagnostic_gate_passed"]) is False

    low_coverage = [
        _summary_row("pooled", -0.20, -4.0, -0.30, 0.01, 0.40),
        _summary_row("fresh_0", -0.15, -3.0, -0.10, 0.01, 0.35),
        _summary_row("fresh_1", -0.15, -3.0, -0.10, 0.01, 0.35),
        _summary_row("fresh_2", -0.15, -3.0, -0.10, 0.01, 0.35),
        _summary_row("fresh_3", -0.15, -3.0, -0.10, 0.01, 0.35),
    ]
    gate = evaluate_inverse_gate(pd.DataFrame.from_records(low_coverage), _gate_config()).iloc[0]
    assert bool(gate["pass_coverage"]) is False
    assert bool(gate["inverse_diagnostic_gate_passed"]) is False


def test_origin_residual_outcomes_matches_hand_computation() -> None:
    oof = pd.DataFrame(
        {
            "fold_id": ["f"] * 4,
            "datetime": [pd.Timestamp("2025-06-11 00:00:00")] * 4,
            "horizon_minutes": [15, 30, 735, 750],
            "horizon_bucket": ["h015_120", "h015_120", "h735_1440", "h735_1440"],
            "actual": [100.0, 200.0, 100.0, 150.0],
            "prediction": [90.0, 180.0, 120.0, 150.0],
        }
    )
    outcomes = origin_residual_outcomes(oof, near_bucket="h015_120", far_bucket="h735_1440")
    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    # near: (100-90)/100=+0.10, (200-180)/200=+0.10 -> mean +0.10
    assert np.isclose(row["near_signed_percentage_residual"], 0.10)
    # far: (100-120)/100=-0.20, (150-150)/150=0.0 -> mean -0.10
    assert np.isclose(row["far_signed_percentage_residual"], -0.10)
    assert np.isclose(row["near_actual_mean"], 150.0)
    assert np.isclose(row["far_actual_mean"], 125.0)
    assert np.isclose(row["far_minus_near_actual_ratio"], (125.0 - 150.0) / 150.0)


def test_origin_residual_outcomes_drops_missing_and_zero_actuals() -> None:
    oof = pd.DataFrame(
        {
            "fold_id": ["f"] * 4,
            "datetime": [pd.Timestamp("2025-06-11 00:00:00")] * 4,
            "horizon_minutes": [15, 30, 735, 750],
            "horizon_bucket": ["h015_120", "h015_120", "h735_1440", "h735_1440"],
            "actual": [np.nan, 200.0, 0.0, 100.0],
            "prediction": [90.0, 180.0, 120.0, 110.0],
        }
    )
    outcomes = origin_residual_outcomes(oof, near_bucket="h015_120", far_bucket="h735_1440")
    row = outcomes.iloc[0]
    # NaN actual dropped from near; zero actual dropped from far
    assert np.isclose(row["near_signed_percentage_residual"], 0.10)
    assert np.isclose(row["far_signed_percentage_residual"], -0.10)


def test_origin_residual_outcomes_requires_both_buckets() -> None:
    oof = pd.DataFrame(
        {
            "fold_id": ["f"],
            "datetime": [pd.Timestamp("2025-06-11 00:00:00")],
            "horizon_minutes": [735],
            "horizon_bucket": ["h735_1440"],
            "actual": [100.0],
            "prediction": [110.0],
        }
    )
    with pytest.raises(AssertionError, match="lacks horizon bucket"):
        origin_residual_outcomes(oof, near_bucket="h015_120", far_bucket="h735_1440")


def _block(fold_id: str, start: str, end: str) -> dict:
    return {"fold_id": fold_id, "validation_start": start, "validation_end": end}


def test_fresh_block_layout_accepts_valid_blocks() -> None:
    blocks = [
        _block("fresh_01", "2025-06-11 00:00:00", "2025-06-20 23:45:00"),
        _block("fresh_02", "2025-06-21 00:00:00", "2025-06-30 23:45:00"),
    ]
    dev = [_block("dev", "2025-07-01 00:00:00", "2025-07-10 23:45:00")]
    validated = check_fresh_block_layout(blocks, dev)
    assert len(validated) == 2
    assert len(validated[0]["origins"]) == 960


def test_fresh_block_layout_rejects_incomplete_block() -> None:
    with pytest.raises(ValueError, match="not a complete ten-day block"):
        check_fresh_block_layout(
            [_block("short", "2025-06-11 00:00:00", "2025-06-20 23:30:00")], []
        )


def test_fresh_block_layout_rejects_fresh_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps another fresh block"):
        check_fresh_block_layout(
            [
                _block("a", "2025-06-11 00:00:00", "2025-06-20 23:45:00"),
                _block("b", "2025-06-15 00:00:00", "2025-06-24 23:45:00"),
            ],
            [],
        )


def test_fresh_block_layout_rejects_dev_overlap() -> None:
    with pytest.raises(ValueError, match="development blocks"):
        check_fresh_block_layout(
            [_block("a", "2025-07-05 00:00:00", "2025-07-14 23:45:00")],
            [_block("dev", "2025-07-01 00:00:00", "2025-07-10 23:45:00")],
        )
