from __future__ import annotations

import numpy as np
import pandas as pd

from src.round2_v3.mape_fuel_utils import (
    fit_mape_median_fuel_proxy,
    predict_mape_median_fuel_proxy,
)


def test_mape_median_fuel_proxy_recovers_linear_relation() -> None:
    x1 = np.linspace(10.0, 100.0, 100)
    x2 = np.linspace(5.0, 20.0, 100) ** 1.1
    frame = pd.DataFrame(
        {
            "fuel_1": x1,
            "fuel_2": x2,
            "target": 25.0 + 0.8 * x1 + 1.2 * x2,
        }
    )
    model = fit_mape_median_fuel_proxy(
        frame,
        target="target",
        fuel_columns=["fuel_1", "fuel_2"],
        denominator_floor=20.0,
    )
    prediction = predict_mape_median_fuel_proxy(
        model, frame=frame, fuel_columns=["fuel_1", "fuel_2"]
    )
    np.testing.assert_allclose(prediction, frame["target"], atol=1e-6)

