import numpy as np
import pandas as pd

from src.round2_v3.holder_boundary_utils import (
    build_boundary_features,
    fit_boundary_contract,
    spearman_rank_correlation,
    summarize_pressure_relationships,
)


def _frame(levels: list[float]) -> pd.DataFrame:
    level = pd.Series(levels, dtype=float)
    return pd.DataFrame(
        {
            "holder_2__level": level,
            "holder_2__delta_1h": level.diff(1),
            "holder_2__delta_4h": level.diff(2),
            "holder_2__delta_12h": level.diff(3),
            "balance_proxy_total__level": np.linspace(-5, 5, len(level)),
        }
    )


def test_boundary_contract_uses_declared_training_values() -> None:
    history = _frame(list(range(100)))
    contract = fit_boundary_contract(
        history,
        reference_quantiles=[0.05, 0.95],
        zone_quantiles=[0.10, 0.90],
        scale_floor=1e-6,
    )
    assert contract["reference_low"] == np.quantile(np.arange(100), 0.05)
    assert contract["reference_high"] == np.quantile(np.arange(100), 0.95)
    assert contract["reference_low"] < contract["zone_low"]
    assert contract["zone_high"] < contract["reference_high"]


def test_boundary_features_are_causal_under_future_perturbation() -> None:
    history = _frame(list(range(100)))
    contract = fit_boundary_contract(
        history.iloc[:80],
        reference_quantiles=[0.05, 0.95],
        zone_quantiles=[0.10, 0.90],
        scale_floor=1e-6,
    )
    original = build_boundary_features(history, contract, maximum_dwell_steps=8)
    changed = history.copy()
    changed.loc[90:, "holder_2__level"] = -9999
    changed.loc[90:, "holder_2__delta_1h"] = -9999
    perturbed = build_boundary_features(changed, contract, maximum_dwell_steps=8)
    pd.testing.assert_frame_equal(original.iloc[:90], perturbed.iloc[:90])


def test_pressure_score_orders_high_rising_above_low_falling() -> None:
    history = _frame(list(range(100)))
    contract = fit_boundary_contract(
        history,
        reference_quantiles=[0.05, 0.95],
        zone_quantiles=[0.10, 0.90],
        scale_floor=1e-6,
    )
    features = build_boundary_features(history, contract, maximum_dwell_steps=8)
    assert features.iloc[-1]["boundary_pressure_score"] > features.iloc[5]["boundary_pressure_score"]
    assert features.iloc[-1]["high_dwell_fraction"] == 1.0


def test_spearman_and_summary_preserve_expected_direction() -> None:
    assert np.isclose(spearman_rank_correlation([1, 2, 3], [10, 20, 30]), 1.0)
    frame = pd.DataFrame(
        {
            "fold_id": ["a"] * 4 + ["b"] * 4,
            "boundary_pressure_score": [-2, -1, 1, 2] * 2,
            "far_signed_percentage_residual": [-0.2, -0.1, 0.1, 0.2] * 2,
            "far_minus_near_actual_ratio": [-0.1, -0.05, 0.05, 0.1] * 2,
            "score_group": ["low", "low", "high", "high"] * 2,
            "high_zone": [0, 0, 1, 1] * 2,
            "low_zone": [1, 1, 0, 0] * 2,
        }
    )
    summary = summarize_pressure_relationships(frame, block_ids=["a", "b"])
    assert len(summary) == 3
    assert (summary["pressure_residual_spearman"] > 0).all()
    assert (summary["high_minus_low_far_residual_pct"] > 0).all()
