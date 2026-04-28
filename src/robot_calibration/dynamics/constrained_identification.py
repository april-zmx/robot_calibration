"""Constrained least-squares identification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_calibration.dynamics.validation import predict_torque


@dataclass(frozen=True)
class BoundedIdentificationResult:
    """Result from bounded ordinary least-squares identification."""

    parameters: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: float
    success: bool
    message: str
    active_lower_bounds: NDArray[np.int64]
    active_upper_bounds: NDArray[np.int64]


def _bounds_array(
    value: ArrayLike | None,
    *,
    size: int,
    fill: float,
    name: str,
) -> NDArray[np.float64]:
    if value is None:
        return np.full(size, fill, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def identify_bounded_ols(
    matrix: ArrayLike,
    torque: ArrayLike,
    *,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    nonnegative_indices: Sequence[int] | None = None,
) -> BoundedIdentificationResult:
    """Estimate parameters with lower and upper bound constraints."""

    observation = np.asarray(matrix, dtype=float)
    tau = np.asarray(torque, dtype=float).reshape(-1)
    if observation.ndim != 2:
        raise ValueError("matrix must be 2D")
    if observation.shape[0] != tau.shape[0]:
        raise ValueError("torque length must match matrix rows")

    n_parameters = observation.shape[1]
    lower = _bounds_array(
        lower_bounds,
        size=n_parameters,
        fill=-np.inf,
        name="lower_bounds",
    )
    upper = _bounds_array(
        upper_bounds,
        size=n_parameters,
        fill=np.inf,
        name="upper_bounds",
    )
    if nonnegative_indices is not None:
        for index in nonnegative_indices:
            if index < 0 or index >= n_parameters:
                raise ValueError("nonnegative index out of range")
            lower[index] = max(lower[index], 0.0)
    if np.any(lower > upper):
        raise ValueError("lower_bounds must be less than or equal to upper_bounds")

    try:
        from scipy.optimize import lsq_linear
    except Exception:
        unconstrained, *_ = np.linalg.lstsq(observation, tau, rcond=None)
        clipped = np.clip(unconstrained, lower, upper)
        residuals = tau - predict_torque(observation, clipped)
        tolerance = 1e-9
        active_lower = np.flatnonzero(np.isfinite(lower) & (clipped <= lower + tolerance))
        active_upper = np.flatnonzero(np.isfinite(upper) & (clipped >= upper - tolerance))
        return BoundedIdentificationResult(
            parameters=clipped,
            residuals=residuals,
            cost=float(0.5 * residuals @ residuals),
            success=True,
            message="solved with numpy fallback",
            active_lower_bounds=active_lower.astype(int),
            active_upper_bounds=active_upper.astype(int),
        )

    solution = lsq_linear(observation, tau, bounds=(lower, upper))
    residuals = tau - predict_torque(observation, solution.x)
    tolerance = 1e-9
    active_lower = np.flatnonzero(np.isfinite(lower) & (solution.x <= lower + tolerance))
    active_upper = np.flatnonzero(np.isfinite(upper) & (solution.x >= upper - tolerance))
    return BoundedIdentificationResult(
        parameters=solution.x,
        residuals=residuals,
        cost=float(solution.cost),
        success=bool(solution.success),
        message=str(solution.message),
        active_lower_bounds=active_lower.astype(int),
        active_upper_bounds=active_upper.astype(int),
    )
