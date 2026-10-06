from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from src.round2_v3.build_component_ladder import (
    _validate_weight,
    blend_generator_1,
    block_change_report,
    write_deterministic_zip,
)


def _frame(g1: list[float], gall: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": ["2025-10-01 00:00:00", "2025-10-01 00:15:00"],
            "generator_1_t+15_pred": g1,
            "generator_all_t+15_pred": gall,
        }
    )


def test_blend_changes_only_generator_1() -> None:
    base = _frame([10.0, 20.0], [100.0, 200.0])
    alternate = _frame([14.0, 12.0], [999.0, 999.0])
    candidate = blend_generator_1(base, alternate, 0.25)
    assert candidate["generator_1_t+15_pred"].tolist() == [11.0, 18.0]
    assert candidate["generator_all_t+15_pred"].tolist() == [100.0, 200.0]
    report = block_change_report(candidate, base, alternate, 0.25)
    assert report["generator_all_exactly_frozen"] is True
    assert report["generator_1_cells_changed_vs_base"] == 2


def test_zero_and_one_weights_preserve_endpoints() -> None:
    base = _frame([10.0, 20.0], [100.0, 200.0])
    alternate = _frame([14.0, 12.0], [100.0, 200.0])
    assert blend_generator_1(base, alternate, 0.0).equals(base)
    one = blend_generator_1(base, alternate, 1.0)
    assert np.array_equal(
        one[["generator_1_t+15_pred"]].to_numpy(),
        alternate[["generator_1_t+15_pred"]].to_numpy(),
    )


def test_weight_must_be_frozen_choice() -> None:
    assert _validate_weight(0.5, [0.0, 0.25, 0.5, 1.0]) == 0.5
    with pytest.raises(ValueError, match="frozen allowed_weights"):
        _validate_weight(0.33, [0.0, 0.25, 0.5, 1.0])


def test_deterministic_zip_is_byte_stable(tmp_path) -> None:
    members = [("s_result.csv", b"a,b\n1,2\n"), ("l_result.csv", b"a,b\n3,4\n")]
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    timestamp = (2026, 10, 2, 0, 0, 0)
    write_deterministic_zip(first, members, timestamp)
    write_deterministic_zip(second, members, timestamp)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first, "r") as archive:
        assert archive.namelist() == ["s_result.csv", "l_result.csv"]
        assert archive.testzip() is None
