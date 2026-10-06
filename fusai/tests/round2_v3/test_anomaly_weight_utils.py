from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.anomaly_weight_utils import (
    anomaly_sample_weights,
    build_v16_legacy_features,
    build_v28_features,
    causal_low_fuel_ratio,
    v28_g1_fuel_weight,
)


def test_anomaly_weights_are_strict_and_nan_safe() -> None:
    ratio = np.array([np.nan, 0.79, 0.80, 0.81])
    result = anomaly_sample_weights(ratio, threshold=0.8, multiplier=3.0)
    np.testing.assert_array_equal(result, np.array([1.0, 3.0, 1.0, 1.0]))


def test_anomaly_weight_validation() -> None:
    with pytest.raises(ValueError):
        anomaly_sample_weights([0.5], threshold=0.8, multiplier=0.5)
    with pytest.raises(ValueError):
        anomaly_sample_weights([0.5], threshold=0.0, multiplier=3.0)


def test_v28_weight_preserves_deployed_minute_boundary() -> None:
    assert v28_g1_fuel_weight(15) == 0.8
    assert v28_g1_fuel_weight(30) == 0.8
    assert v28_g1_fuel_weight(45) == 0.6
    assert v28_g1_fuel_weight(1440) == 0.6
    with pytest.raises(ValueError):
        v28_g1_fuel_weight(32)


def test_low_fuel_ratio_has_no_future_dependency() -> None:
    rows = 800
    frame = pd.DataFrame(
        {
            "generator_use_blast_furnace_gas": np.linspace(100.0, 120.0, rows),
            "generator_use_converter_gas": np.linspace(20.0, 25.0, rows),
        }
    )
    coefficients = np.array([1.0, 1.0, 0.0])
    original = causal_low_fuel_ratio(frame, coefficients)
    changed = frame.copy()
    changed.loc[600:, "generator_use_blast_furnace_gas"] = 9999.0
    perturbed = causal_low_fuel_ratio(changed, coefficients)
    np.testing.assert_allclose(
        original.iloc[:600], perturbed.iloc[:600], equal_nan=True, rtol=0, atol=0
    )


def test_legacy_and_restored_feature_boundary() -> None:
    rows = 100
    data = {
        "datetime": pd.date_range("2025-09-01", periods=rows, freq="15min"),
        "generator_1": np.linspace(100, 120, rows),
        "generator_all": np.linspace(200, 230, rows),
        "blast_furnace_1": np.ones(rows),
        "blast_furnace_2": np.ones(rows) * 2,
        "blast_furnace_3": np.ones(rows) * 3,
        "blast_furnace_4": np.ones(rows) * 4,
        "blast_furnace_5": np.ones(rows) * 5,
        "air_heater_1": np.ones(rows),
        "air_heater_2": np.ones(rows),
        "air_heater_3": np.ones(rows),
        "air_heater_4": np.ones(rows),
        "air_heater_5": np.ones(rows),
        "blast_furnace_user1": np.ones(rows),
        "blast_furnace_user2": np.ones(rows),
        "blast_furnace_user3": np.ones(rows),
        "blast_furnace_user4": np.ones(rows),
        "into_gas_mixed_blast_furnace": np.ones(rows),
        "coke_oven_1": np.ones(rows),
        "into_gas_mixed_coke": np.ones(rows),
        "converter_1": np.ones(rows),
        "converter_user1": np.ones(rows),
        "converter_user2": np.ones(rows),
        "into_gas_mixed_converter": np.ones(rows),
        "blast_furnace_gas_holder_2": np.arange(rows, dtype=float),
        "generator_use_blast_furnace_gas": np.ones(rows),
        "generator_use_coke_gas": np.ones(rows),
        "generator_use_converter_gas": np.ones(rows),
    }
    frame = pd.DataFrame(data)
    legacy, legacy_columns = build_v16_legacy_features(frame)
    restored, restored_columns = build_v28_features(frame)
    assert "blast_furnace_3" not in legacy_columns
    assert "air_heater_3" not in legacy_columns
    assert "blast_furnace_3" in restored_columns
    assert "air_heater_3" in restored_columns
    assert legacy.shape[0] == restored.shape[0] == rows
    assert np.isfinite(legacy.iloc[-1].to_numpy(dtype=float)).all()
    assert np.isfinite(restored.iloc[-1].to_numpy(dtype=float)).all()
