"""Pure preprocessing and model helpers for direct multi-step GRU experiments."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


class DirectMultiStepGRU(nn.Module):
    """Encode one historical sequence and directly emit every forecast horizon."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_size: int,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0 or output_size <= 0:
            raise ValueError("GRU dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("GRU input must have shape batch x time x features")
        _, hidden = self.gru(values)
        return self.head(self.norm(hidden[-1]))


def set_reproducible_seed(seed: int, *, deterministic_cudnn: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic_cudnn)
    torch.backends.cudnn.benchmark = not bool(deterministic_cudnn)


def add_step_calendar(
    frame: pd.DataFrame,
    datetimes: pd.DatetimeIndex,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Append causal calendar values for every historical input step."""
    if len(frame) != len(datetimes):
        raise ValueError("frame and datetimes have different row counts")
    values = frame.to_numpy(dtype=float)
    names = list(frame.columns)
    additions: list[np.ndarray] = []
    if config["add_step_hour_cyclic"]:
        angle = 2 * np.pi * (datetimes.hour + datetimes.minute / 60.0) / 24.0
        additions.extend([np.sin(angle), np.cos(angle)])
        names.extend(["step_hour_sin", "step_hour_cos"])
    if config["add_step_dayofweek_cyclic"]:
        angle = 2 * np.pi * datetimes.dayofweek / 7.0
        additions.extend([np.sin(angle), np.cos(angle)])
        names.extend(["step_dayofweek_sin", "step_dayofweek_cos"])
    if additions:
        values = np.column_stack([values, *additions])
    return values, names


def fit_transform_time_features(
    all_values: np.ndarray,
    training_row_mask: np.ndarray,
    *,
    scale_floor: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit median/IQR on training history rows and transform the complete grid."""
    values = np.asarray(all_values, dtype=float)
    mask = np.asarray(training_row_mask, dtype=bool)
    if values.ndim != 2 or mask.ndim != 1 or len(values) != len(mask):
        raise ValueError("feature matrix/training mask shapes are inconsistent")
    if not mask.any() or scale_floor <= 0:
        raise ValueError("training mask must be non-empty and scale floor positive")
    training = values[mask]
    median = np.nanmedian(training, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("training history contains an all-missing feature")
    filled_training = np.where(np.isfinite(training), training, median)
    filled_all = np.where(np.isfinite(values), values, median)
    q25, q75 = np.percentile(filled_training, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, float(scale_floor))
    transformed = (filled_all - median) / scale
    if not np.isfinite(transformed).all():
        raise AssertionError("transformed sequence features contain non-finite values")
    return transformed.astype(np.float32), {"median": median, "scale": scale}


def build_sequence_windows(
    transformed_grid: np.ndarray,
    origin_positions: np.ndarray,
    *,
    window_steps: int,
) -> np.ndarray:
    """Create windows ending at and including each origin position."""
    values = np.asarray(transformed_grid, dtype=np.float32)
    positions = np.asarray(origin_positions, dtype=int)
    if values.ndim != 2 or positions.ndim != 1:
        raise ValueError("grid must be 2D and positions must be 1D")
    if window_steps <= 0 or len(positions) == 0:
        raise ValueError("window_steps and origin positions must be non-empty")
    if positions.min() < window_steps - 1 or positions.max() >= len(values):
        raise ValueError("origin position lacks a complete history window")
    offsets = np.arange(-(window_steps - 1), 1, dtype=int)
    return values[positions[:, None] + offsets[None, :]]


def fit_log_target_normalization(
    targets: np.ndarray,
    *,
    scale_floor: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Log-transform positive trajectories and standardize each horizon."""
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("training trajectories must be finite and strictly positive")
    logged = np.log(values)
    mean = logged.mean(axis=0)
    scale = np.maximum(logged.std(axis=0), float(scale_floor))
    return ((logged - mean) / scale).astype(np.float32), {"mean": mean, "scale": scale}


def inverse_log_target_normalization(
    standardized: np.ndarray,
    fitted: dict[str, np.ndarray],
    *,
    prediction_min: float,
    prediction_max: float,
) -> np.ndarray:
    """Invert per-horizon log normalization and enforce physical bounds."""
    values = np.asarray(standardized, dtype=float)
    mean = np.asarray(fitted["mean"], dtype=float)
    scale = np.asarray(fitted["scale"], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(mean) or mean.shape != scale.shape:
        raise ValueError("standardized outputs and target normalization differ")
    if not 0 < prediction_min < prediction_max:
        raise ValueError("prediction bounds are invalid")
    logged = values * scale + mean
    predictions = np.exp(np.clip(logged, -20.0, np.log(prediction_max * 10.0)))
    return np.clip(predictions, prediction_min, prediction_max)
