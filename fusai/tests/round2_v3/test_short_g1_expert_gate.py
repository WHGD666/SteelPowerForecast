import numpy as np
import pandas as pd

from src.round2_v3.run_short_g1_expert_gate import build_origin_gate_table, soft_gate_predictions


def test_soft_gate_endpoints_and_midpoint() -> None:
    control = np.asarray([10.0, 20.0, 30.0])
    candidate = np.asarray([20.0, 10.0, 40.0])
    probability = np.asarray([0.0, 0.5, 1.0])
    assert np.allclose(soft_gate_predictions(control, candidate, probability), [10.0, 15.0, 40.0])


def test_origin_gate_label_uses_all_horizons() -> None:
    rows = []
    for variant, predictions in (("control", [110.0, 110.0]), ("candidate", [101.0, 99.0])):
        for horizon, prediction in zip([15, 30], predictions, strict=True):
            rows.append({"variant": variant, "fold_id": "fold", "datetime": "2025-09-01", "interval_start": "2025-09-01", "horizon_minutes": horizon, "actual": 100.0, "prediction": prediction})
    table = build_origin_gate_table(pd.DataFrame(rows), "control", "candidate", [15, 30])
    assert table["expert_win"].tolist() == [1]
    assert table["disagreement_mean"].tolist() == [-10.0]
