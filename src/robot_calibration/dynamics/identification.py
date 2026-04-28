"""Least-squares dynamic parameter identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class IdentificationResult:
    parameters: NDArray[np.float64]
    residuals: NDArray[np.float64]
    rank: int
    singular_values: NDArray[np.float64]
    covariance: NDArray[np.float64]
    standard_deviation: NDArray[np.float64]
    relative_standard_deviation: NDArray[np.float64]


def identify_ols(matrix: ArrayLike, torque: ArrayLike) -> IdentificationResult:
    """Estimate parameters with ordinary least squares."""

    observation = np.asarray(matrix, dtype=float)
    tau = np.asarray(torque, dtype=float).reshape(-1)
    if observation.ndim != 2:
        raise ValueError("matrix must be 2D")
    if observation.shape[0] != tau.shape[0]:
        raise ValueError("torque length must match matrix rows")

    parameters, residuals, rank, singular_values = np.linalg.lstsq(
        observation,
        tau,
        rcond=None,
    )
    prediction_error = tau - observation @ parameters
    dof = max(observation.shape[0] - observation.shape[1], 1)
    variance = float(prediction_error @ prediction_error / dof)
    normal = observation.T @ observation
    covariance = variance * np.linalg.pinv(normal)
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    relative = np.divide(
        100.0 * standard_deviation,
        np.abs(parameters),
        out=np.full_like(standard_deviation, np.inf),
        where=np.abs(parameters) > 0.0,
    )
    return IdentificationResult(
        parameters=parameters,
        residuals=prediction_error,
        rank=int(rank),
        singular_values=singular_values,
        covariance=covariance,
        standard_deviation=standard_deviation,
        relative_standard_deviation=relative,
    )
