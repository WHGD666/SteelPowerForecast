import numpy as np
import pandas as pd

from src.round2_v3.run_short_g1_fuel_trend import _load_control, damped_fuel_trend_correction


def test_damped_fuel_trend_factor_and_cap() -> None:
    current = np.asarray([[20.0, 30.0, 40.0], [100.0, 100.0, 100.0]])
    lagged = np.asarray([[16.0, 26.0, 36.0], [50.0, 50.0, 50.0]])
    coefficients = np.asarray([1.0, 0.0, 0.0, 99.0])
    correction_30, raw_30 = damped_fuel_trend_correction(
        current,
        lagged,
        coefficients,
        30,
        fuel_weight=0.8,
        full_extrapolation_minutes=60,
        maximum_factor=1.0,
        cap_mw=5.0,
    )
    assert np.allclose(raw_30, [1.6, 20.0])
    assert np.allclose(correction_30, [1.6, 5.0])
    correction_120, raw_120 = damped_fuel_trend_correction(
        current,
        lagged,
        coefficients,
        120,
        fuel_weight=0.8,
        full_extrapolation_minutes=60,
        maximum_factor=1.0,
        cap_mw=5.0,
    )
    assert np.allclose(raw_120, [3.2, 40.0])
    assert np.allclose(correction_120, [3.2, 5.0])


def test_load_control_normalizes_variant_name(tmp_path) -> None:
    path = tmp_path / "control.csv"
    pd.DataFrame(
        {
            "variant": ["v16_legacy_columns"],
            "fold_id": ["fold_1"],
            "datetime": ["2025-09-01 00:00:00"],
            "interval_start": ["2025-09-01 00:00:00"],
            "horizon_minutes": [15],
            "actual_interval_mean": [100.0],
            "prediction": [101.0],
        }
    ).to_csv(path, index=False)
    loaded = _load_control(path, "v16_legacy_columns", ["fold_1"])
    assert loaded["variant"].tolist() == ["v16_control"]
