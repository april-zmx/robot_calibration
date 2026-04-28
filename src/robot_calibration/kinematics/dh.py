"""Denavit-Hartenberg forward kinematics."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_calibration.models import DHParameter


def dh_transform(a: float, alpha: float, d: float, theta: float) -> NDArray[np.float64]:
    """Return a standard DH homogeneous transform."""

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(
    parameters: Sequence[DHParameter],
    joint_positions: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the end-effector transform for a serial DH chain."""

    q = np.asarray(joint_positions, dtype=float)
    if q.shape != (len(parameters),):
        raise ValueError("joint_positions must match number of DH rows")

    transform = np.eye(4)
    for row, joint_position in zip(parameters, q):
        transform = transform @ dh_transform(
            row.a,
            row.alpha,
            row.d,
            row.theta + float(joint_position),
        )
    return transform


def with_parameter_offsets(
    parameters: Sequence[DHParameter],
    offsets: ArrayLike,
    *,
    estimate: tuple[str, ...],
) -> list[DHParameter]:
    """Apply per-link offsets for selected DH fields."""

    delta = np.asarray(offsets, dtype=float)
    expected = len(parameters) * len(estimate)
    if delta.shape != (expected,):
        raise ValueError(f"offsets must have shape ({expected},)")

    updated: list[DHParameter] = []
    index = 0
    for row in parameters:
        values = {}
        for field in estimate:
            values[field] = getattr(row, field) + float(delta[index])
            index += 1
        updated.append(replace(row, **values))
    return updated
