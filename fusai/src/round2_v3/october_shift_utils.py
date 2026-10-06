"""Pure per-column distribution-shift metrics for the October localization audit."""

from __future__ import annotations

import numpy as np


def column_shift_metrics(
    train: np.ndarray,
    reference: np.ndarray,
    test: np.ndarray,
    *,
    soft_low_quantile: float = 0.005,
    soft_high_quantile: float = 0.995,
) -> dict[str, float]:
    """Compare a test-period column against full-train support and a reference window.

    `train` provides the support the models were trained on; `reference` (September)
    provides the recent baseline for the robust median shift; `test` is October.
    All inputs may contain NaN.  Every returned value is a float (NaN when undefined).
    """
    train = np.asarray(train, dtype=float)
    reference = np.asarray(reference, dtype=float)
    test = np.asarray(test, dtype=float)
    train_finite = train[np.isfinite(train)]
    reference_finite = reference[np.isfinite(reference)]
    test_finite = test[np.isfinite(test)]
    result: dict[str, float] = {
        "train_n": float(train_finite.size),
        "reference_n": float(reference_finite.size),
        "test_n": float(test_finite.size),
        "test_missing_rate": float(1.0 - test_finite.size / test.size) if test.size else float("nan"),
        "train_median": float("nan"),
        "reference_median": float("nan"),
        "test_median": float("nan"),
        "reference_iqr": float("nan"),
        "median_shift_in_reference_iqr": float("nan"),
        "soft_support_low": float("nan"),
        "soft_support_high": float("nan"),
        "out_of_support_rate": float("nan"),
        "out_of_support_below_rate": float("nan"),
        "out_of_support_above_rate": float("nan"),
        "hard_out_of_range_rate": float("nan"),
        "hard_below_rate": float("nan"),
        "hard_above_rate": float("nan"),
    }
    if not train_finite.size or not test_finite.size:
        return result
    low, high = np.quantile(train_finite, [soft_low_quantile, soft_high_quantile])
    below = float((test_finite < low).mean())
    above = float((test_finite > high).mean())
    hard_min, hard_max = float(train_finite.min()), float(train_finite.max())
    result.update(
        {
            "train_median": float(np.median(train_finite)),
            "soft_support_low": float(low),
            "soft_support_high": float(high),
            "out_of_support_rate": below + above,
            "out_of_support_below_rate": below,
            "out_of_support_above_rate": above,
            "hard_out_of_range_rate": float((test_finite < hard_min).mean() + (test_finite > hard_max).mean()),
            "hard_below_rate": float((test_finite < hard_min).mean()),
            "hard_above_rate": float((test_finite > hard_max).mean()),
        }
    )
    if test_finite.size:
        result["test_median"] = float(np.median(test_finite))
    if reference_finite.size:
        q25, q75 = np.quantile(reference_finite, [0.25, 0.75])
        iqr = float(q75 - q25)
        result["reference_median"] = float(np.median(reference_finite))
        result["reference_iqr"] = iqr
        if iqr > 0:
            result["median_shift_in_reference_iqr"] = float(
                (result["test_median"] - result["reference_median"]) / iqr
            )
    return result
