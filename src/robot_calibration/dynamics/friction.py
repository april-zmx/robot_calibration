"""Friction models and regressors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def linear_friction_regressor(velocities: ArrayLike) -> NDArray[np.float64]:
    """Return block-diagonal rows for Fv * qd + Fc * sign(qd) + F0."""

    qd = np.asarray(velocities, dtype=float)
    if qd.ndim != 1:
        raise ValueError("velocities must be a 1D array")
    n_joints = qd.shape[0]
    regressor = np.zeros((n_joints, 3 * n_joints), dtype=float)
    for joint in range(n_joints):
        start = 3 * joint
        regressor[joint, start : start + 3] = [
            qd[joint],
            np.sign(qd[joint]),
            1.0,
        ]
    return regressor
