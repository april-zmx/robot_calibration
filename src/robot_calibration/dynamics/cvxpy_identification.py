"""CVXPY-backed constrained least-squares identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_calibration.dynamics.validation import predict_torque


@dataclass(frozen=True)
class CvxpyIdentificationResult:
    """Result from CVXPY constrained least-squares identification."""

    parameters: NDArray[np.float64]
    residuals: NDArray[np.float64]
    objective_value: float
    status: str
    success: bool


@dataclass(frozen=True)
class PhysicalParameterBlock:
    """Pinocchio-style physical parameter block.

    Layout:
    ``[m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, Im?]``.
    """

    start: int
    has_motor_inertia: bool = False

    @property
    def width(self) -> int:
        return 11 if self.has_motor_inertia else 10


def _load_cvxpy():
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cvxpy is required for identify_cvxpy_constrained_ols; install robot-calibration with the cvxpy extra"
        ) from exc
    return cp


def _index_array(indices: Sequence[int] | None, *, size: int, name: str) -> NDArray[np.int64]:
    if indices is None:
        return np.array([], dtype=int)
    array = np.asarray(indices, dtype=int)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    if np.any((array < 0) | (array >= size)):
        raise ValueError(f"{name} contains index out of range")
    return array


def identify_cvxpy_constrained_ols(
    matrix: ArrayLike,
    torque: ArrayLike,
    *,
    nonnegative_indices: Sequence[int] | None = None,
    mass_indices: Sequence[int] | None = None,
    mass_upper_bounds: ArrayLike | None = None,
    physical_parameter_blocks: Sequence[PhysicalParameterBlock] | None = None,
    solver: str | None = None,
) -> CvxpyIdentificationResult:
    """Estimate parameters using CVXPY least squares with simple constraints."""

    cp = _load_cvxpy()
    observation = np.asarray(matrix, dtype=float)
    tau = np.asarray(torque, dtype=float).reshape(-1)
    if observation.ndim != 2:
        raise ValueError("matrix must be 2D")
    if observation.shape[0] != tau.shape[0]:
        raise ValueError("torque length must match matrix rows")

    n_parameters = observation.shape[1]
    nonnegative = _index_array(
        nonnegative_indices,
        size=n_parameters,
        name="nonnegative_indices",
    )
    masses = _index_array(mass_indices, size=n_parameters, name="mass_indices")
    if mass_upper_bounds is not None:
        upper = np.asarray(mass_upper_bounds, dtype=float)
        if upper.shape != (masses.size,):
            raise ValueError("mass_upper_bounds must match mass_indices length")
    else:
        upper = None

    params = cp.Variable(n_parameters)
    constraints = []
    if nonnegative.size:
        constraints.append(params[nonnegative] >= 0.0)
    if masses.size:
        constraints.append(params[masses] >= 0.0)
        if upper is not None:
            constraints.append(params[masses] <= upper)
    if physical_parameter_blocks is not None:
        for block in physical_parameter_blocks:
            if block.start < 0 or block.start + block.width > n_parameters:
                raise ValueError("physical parameter block is out of range")
            m = params[block.start]
            h = params[block.start + 1 : block.start + 4]
            ixx = params[block.start + 4]
            ixy = params[block.start + 5]
            iyy = params[block.start + 6]
            ixz = params[block.start + 7]
            iyz = params[block.start + 8]
            izz = params[block.start + 9]
            inertia = cp.bmat(
                [
                    [ixx, ixy, ixz],
                    [ixy, iyy, iyz],
                    [ixz, iyz, izz],
                ]
            )
            pseudo_inertia = cp.bmat(
                [
                    [0.5 * cp.trace(inertia) * np.eye(3) - inertia, h[:, None]],
                    [h[None, :], cp.reshape(m, (1, 1), order="C")],
                ]
            )
            constraints.append(pseudo_inertia >> 0)
            constraints.append(m >= 0.0)
            if block.has_motor_inertia:
                constraints.append(params[block.start + 10] >= 0.0)

    objective = cp.Minimize(cp.sum_squares(observation @ params - tau))
    problem = cp.Problem(objective, constraints)
    if solver is None:
        problem.solve()
    else:
        problem.solve(solver=solver)

    if params.value is None:
        parameters = np.full(n_parameters, np.nan, dtype=float)
    else:
        parameters = np.asarray(params.value, dtype=float).reshape(-1)
    residuals = tau - predict_torque(observation, parameters)
    return CvxpyIdentificationResult(
        parameters=parameters,
        residuals=residuals,
        objective_value=float(problem.value) if problem.value is not None else np.nan,
        status=str(problem.status),
        success=problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE},
    )
