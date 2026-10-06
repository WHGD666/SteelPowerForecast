import numpy as np
import pandas as pd

from src.round2_v3.run_test_similarity_calibration import (
    normalized_similarity_weights,
    weighted_accuracy_gain_pct,
)


def test_normalized_similarity_weights_are_bounded_and_mean_one() -> None:
    weights = normalized_similarity_weights(
        np.array([0.01, 0.2, 0.5, 0.99]),
        probability_clip=(0.05, 0.95),
        weight_clip=(0.25, 4.0),
    )
    assert np.isclose(weights.mean(), 1.0)
    assert np.all(weights > 0)
    assert weights[-1] > weights[0]


def test_weighted_accuracy_gain_matches_manual_mape_difference() -> None:
    keys = {
        "fold_id": ["f", "f"],
        "datetime": pd.to_datetime(["2025-09-01 00:00", "2025-09-01 00:15"]),
        "interval_start": pd.to_datetime(["2025-09-01 00:00", "2025-09-01 00:15"]),
        "horizon_minutes": [15, 15],
        "actual": [100.0, 100.0],
        "weight": [1.0, 3.0],
    }
    control = pd.DataFrame({**keys, "variant": "control", "prediction": [90.0, 80.0]})
    candidate = pd.DataFrame({**keys, "variant": "candidate", "prediction": [80.0, 90.0]})
    frame = pd.concat([control, candidate], ignore_index=True)
    gain = weighted_accuracy_gain_pct(frame, control_variant="control", candidate_variant="candidate", actual_column="actual", weight_column="weight", actual_floor=1e-6)
    # control weighted MAPE=17.5%, candidate=12.5%
    assert np.isclose(gain, 5.0)

