"""Pure helpers for causal historical-analog trajectory prediction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def build_future_trajectory_matrix(
    interval_targets: pd.DataFrame,
    origins: Iterable[pd.Timestamp | str],
    horizons_minutes: Iterable[int],
    target: str,
) -> np.ndarray:
    """Return origin x horizon official interval-mean labels."""
    if "datetime" not in interval_targets or target not in interval_targets:
        raise KeyError(f"label frame missing datetime/{target}")
    labels = interval_targets[["datetime", target]].copy()
    labels["datetime"] = pd.to_datetime(labels["datetime"], errors="raise")
    if labels["datetime"].duplicated().any():
        raise ValueError("label timestamps must be unique")
    series = pd.to_numeric(labels.set_index("datetime")[target], errors="coerce")
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origins), errors="raise"))
    horizons = np.asarray([int(value) for value in horizons_minutes], dtype=int)
    starts = origin_index.to_numpy(dtype="datetime64[ns]")[:, None] + (
        horizons[None, :] - 15
    ).astype("timedelta64[m]")
    flat = pd.DatetimeIndex(starts.reshape(-1))
    return series.reindex(flat).to_numpy(dtype=float).reshape(len(origin_index), len(horizons))


def robust_standardize(
    library: np.ndarray,
    query: np.ndarray,
    *,
    scale_floor: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit median/IQR preprocessing on the library only and apply to queries."""
    lib = np.asarray(library, dtype=float)
    qry = np.asarray(query, dtype=float)
    if lib.ndim != 2 or qry.ndim != 2 or lib.shape[1] != qry.shape[1]:
        raise ValueError("library/query feature matrices must be 2D with equal width")
    if scale_floor <= 0:
        raise ValueError("scale_floor must be positive")
    median = np.nanmedian(lib, axis=0)
    if not np.isfinite(median).all():
        raise ValueError("library contains an all-missing feature")
    lib_filled = np.where(np.isfinite(lib), lib, median)
    qry_filled = np.where(np.isfinite(qry), qry, median)
    q25, q75 = np.percentile(lib_filled, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, float(scale_floor))
    return (
        (lib_filled - median) / scale,
        (qry_filled - median) / scale,
        {"median": median, "scale": scale},
    )


def select_diverse_neighbors(
    distances: np.ndarray,
    source_days: np.ndarray,
    *,
    neighbor_count: int,
    maximum_per_day: int,
) -> np.ndarray:
    """Select nearest indices subject to a source-day diversity cap."""
    values = np.asarray(distances, dtype=float)
    days = np.asarray(source_days)
    if values.ndim != 1 or len(values) != len(days) or not np.isfinite(values).all():
        raise ValueError("distances/source_days must be aligned and finite")
    if neighbor_count <= 0 or maximum_per_day <= 0:
        raise ValueError("neighbor limits must be positive")
    chosen: list[int] = []
    counts: dict[object, int] = {}
    for index in np.argsort(values, kind="stable"):
        day = days[index]
        count = counts.get(day, 0)
        if count >= maximum_per_day:
            continue
        chosen.append(int(index))
        counts[day] = count + 1
        if len(chosen) == neighbor_count:
            break
    if len(chosen) != neighbor_count:
        raise ValueError(
            f"only {len(chosen)} diverse neighbors available; need {neighbor_count}"
        )
    return np.asarray(chosen, dtype=int)


def predict_analog_trajectories(
    library_features: np.ndarray,
    query_features: np.ndarray,
    library_trajectories: np.ndarray,
    source_days: np.ndarray,
    *,
    neighbor_count: int,
    maximum_per_day: int,
    distance_epsilon: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predict complete future trajectories with inverse-distance neighbors."""
    library = np.asarray(library_features, dtype=float)
    query = np.asarray(query_features, dtype=float)
    trajectories = np.asarray(library_trajectories, dtype=float)
    if library.ndim != 2 or query.ndim != 2 or trajectories.ndim != 2:
        raise ValueError("analog inputs must be 2D")
    if library.shape[0] != trajectories.shape[0] or library.shape[1] != query.shape[1]:
        raise ValueError("analog input shapes are inconsistent")
    if not np.isfinite(library).all() or not np.isfinite(query).all():
        raise ValueError("standardized analog features must be finite")
    if not np.isfinite(trajectories).all():
        raise ValueError("library trajectories must be complete and finite")
    if distance_epsilon <= 0:
        raise ValueError("distance_epsilon must be positive")

    predictions = np.empty((len(query), trajectories.shape[1]), dtype=float)
    diagnostics: list[dict[str, float | int]] = []
    for row_index, row in enumerate(query):
        distances = np.sqrt(np.mean(np.square(library - row), axis=1))
        selected = select_diverse_neighbors(
            distances,
            source_days,
            neighbor_count=neighbor_count,
            maximum_per_day=maximum_per_day,
        )
        selected_distances = distances[selected]
        weights = 1.0 / np.maximum(selected_distances, distance_epsilon)
        weights /= weights.sum()
        predictions[row_index] = weights @ trajectories[selected]
        diagnostics.append(
            {
                "query_index": row_index,
                "nearest_distance": float(selected_distances.min()),
                "mean_neighbor_distance": float(selected_distances.mean()),
                "maximum_neighbor_distance": float(selected_distances.max()),
                "effective_neighbor_count": float(1.0 / np.square(weights).sum()),
                "unique_source_days": int(len(np.unique(source_days[selected]))),
            }
        )
    return predictions, pd.DataFrame.from_records(diagnostics)
