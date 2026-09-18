import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_submission", ROOT / "src" / "build_submission.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_persistence_submission_schema_and_values() -> None:
    prepared = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-05-01", periods=3, freq="15min"),
            "blast_furnace_1": [1.0, 2.0, 3.0],
            "feat_current_generator_1": [10.0, 11.0, 12.0],
            "feat_current_generator_all": [20.0, 21.0, 22.0],
            "feat_missing_generator_1": [0, 1, 0],
            "feat_missing_generator_all": [0, 0, 1],
        }
    )
    targets = ["generator_1", "generator_all"]
    horizons = [15, 30]
    submission_input = MODULE.build_submission_input(prepared, targets)
    predictions = MODULE.build_persistence_predictions(submission_input, targets, horizons)

    assert "feat_current_generator_1" not in submission_input
    assert "feat_current_generator_all" not in submission_input
    assert submission_input["generator_1"].tolist() == [10.0, 11.0, 12.0]
    assert submission_input["generator_all"].tolist() == [20.0, 21.0, 22.0]
    assert predictions.columns.tolist() == [
        "datetime",
        "generator_1_t+15_pred",
        "generator_1_t+30_pred",
        "generator_all_t+15_pred",
        "generator_all_t+30_pred",
    ]
    assert predictions["generator_1_t+30_pred"].tolist() == [10.0, 11.0, 12.0]
    assert predictions["generator_all_t+15_pred"].tolist() == [20.0, 21.0, 22.0]


def test_submission_validator_accepts_complete_contract() -> None:
    datetimes = pd.Series(pd.date_range("2025-05-01", periods=3, freq="15min"), name="datetime")
    prepared = pd.DataFrame(
        {
            "datetime": datetimes,
            "blast_furnace_1": [1.0, 2.0, 3.0],
            "feat_current_generator_1": [10.0, 11.0, 12.0],
            "feat_current_generator_all": [20.0, 21.0, 22.0],
            "feat_missing_generator_1": [0, 1, 0],
            "feat_missing_generator_all": [0, 0, 1],
        }
    )
    targets = ["generator_1", "generator_all"]
    horizons = [15, 30]
    submission_input = MODULE.build_submission_input(prepared, targets)
    predictions = MODULE.build_persistence_predictions(submission_input, targets, horizons)
    result = MODULE.validate_submission(
        submission_input, predictions, datetimes, targets, horizons
    )
    assert result["rows"] == 3
    assert result["missing_cells"] == 0
