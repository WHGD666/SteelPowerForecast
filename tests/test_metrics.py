import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_baseline", ROOT / "src" / "run_baseline.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_metrics_exact_values() -> None:
    actual = np.array([10.0, 20.0])
    predicted = np.array([8.0, 22.0])
    result = MODULE.calculate_metrics(actual, predicted)
    assert np.isclose(result["mape"], 0.15)
    assert np.isclose(result["score_1_mape"], 0.85)
    assert np.isclose(result["mae"], 2.0)
    assert np.isclose(result["rmse"], 2.0)
    assert result["zero_actual_count"] == 0


def test_metrics_exclude_zero_actual_and_count_it() -> None:
    result = MODULE.calculate_metrics(np.array([0.0, 10.0]), np.array([5.0, 8.0]))
    assert np.isclose(result["mape"], 0.2)
    assert result["zero_actual_count"] == 1
