"""Common calibration error metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(measured: ArrayLike, predicted: ArrayLike) -> float:
    error = np.asarray(measured, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(error**2)))


def relative_residual_error(
    measured: ArrayLike,
    predicted: ArrayLike,
    *,
    axis: int | None = None,
) -> np.ndarray:
    measured_array = np.asarray(measured, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    numerator = np.linalg.norm(measured_array - predicted_array, axis=axis)
    denominator = np.linalg.norm(measured_array, axis=axis)
    return np.divide(
        100.0 * numerator,
        denominator,
        out=np.full_like(numerator, np.inf, dtype=float),
        where=denominator > 0.0,
    )


def condition_number(matrix: ArrayLike) -> float:
    return float(np.linalg.cond(np.asarray(matrix, dtype=float)))
