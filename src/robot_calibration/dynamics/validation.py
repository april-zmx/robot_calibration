"""Validation helpers for dynamic calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def predict_torque(matrix: ArrayLike, parameters: ArrayLike) -> NDArray[np.float64]:
    """Predict stacked torques from an observation matrix."""

    observation = np.asarray(matrix, dtype=float)
    params = np.asarray(parameters, dtype=float)
    if observation.ndim != 2 or params.ndim != 1:
        raise ValueError("matrix must be 2D and parameters must be 1D")
    if observation.shape[1] != params.shape[0]:
        raise ValueError("parameter length must match matrix columns")
    return observation @ params
