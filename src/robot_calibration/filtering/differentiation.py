"""Numerical differentiation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def central_difference(time: ArrayLike, signal: ArrayLike) -> NDArray[np.float64]:
    """Estimate time derivative using central differences."""

    t = np.asarray(time, dtype=float)
    y = np.asarray(signal, dtype=float)
    if t.ndim != 1:
        raise ValueError("time must be a 1D array")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("signal must be a 1D or 2D array")
    if y.shape[0] != t.shape[0]:
        raise ValueError("signal must have one row per timestamp")
    if t.shape[0] < 3:
        raise ValueError("at least three samples are required")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("time must be strictly increasing")

    derivative = np.empty_like(y)
    derivative[1:-1] = (y[2:] - y[:-2]) / (t[2:, None] - t[:-2, None])
    derivative[0] = (y[1] - y[0]) / (t[1] - t[0])
    derivative[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return derivative
