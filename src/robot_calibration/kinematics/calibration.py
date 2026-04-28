"""Kinematic calibration routines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from robot_calibration.kinematics.dh import forward_kinematics, with_parameter_offsets
from robot_calibration.models import CalibrationResult, DHParameter


def _residuals(
    offsets: np.ndarray,
    parameters: Sequence[DHParameter],
    joint_positions: np.ndarray,
    observations: np.ndarray,
    estimate: tuple[str, ...],
) -> np.ndarray:
    calibrated = with_parameter_offsets(parameters, offsets, estimate=estimate)
    predictions = np.array(
        [forward_kinematics(calibrated, q)[:3, 3] for q in joint_positions]
    )
    return (predictions - observations).ravel()


def calibrate_dh_offsets(
    parameters: Sequence[DHParameter],
    joint_positions: ArrayLike,
    observed_positions: ArrayLike,
    *,
    estimate: tuple[str, ...] = ("a", "d", "theta"),
) -> CalibrationResult:
    """Estimate offsets for selected DH fields from end-effector positions."""

    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for calibrate_dh_offsets; install robot-calibration dependencies"
        ) from exc

    q = np.asarray(joint_positions, dtype=float)
    observations = np.asarray(observed_positions, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(parameters):
        raise ValueError("joint_positions must have shape (n_samples, n_joints)")
    if observations.shape != (q.shape[0], 3):
        raise ValueError("observed_positions must have shape (n_samples, 3)")
    invalid = set(estimate) - {"a", "alpha", "d", "theta"}
    if invalid:
        raise ValueError(f"unsupported DH fields: {', '.join(sorted(invalid))}")

    initial = np.zeros(len(parameters) * len(estimate), dtype=float)
    solution = least_squares(
        _residuals,
        initial,
        args=(parameters, q, observations, tuple(estimate)),
    )
    calibrated = with_parameter_offsets(parameters, solution.x, estimate=tuple(estimate))
    return CalibrationResult(
        success=bool(solution.success),
        parameters=calibrated,
        residuals=solution.fun,
        cost=float(solution.cost),
        message=str(solution.message),
    )
