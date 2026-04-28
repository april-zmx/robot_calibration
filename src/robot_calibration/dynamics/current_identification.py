"""Current-driven dynamic parameter identification workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_calibration.data import CalibrationDataset
from robot_calibration.dynamics.base_parameters import BaseParameterMapping
from robot_calibration.dynamics.constrained_identification import (
    BoundedIdentificationResult,
    identify_bounded_ols,
)
from robot_calibration.dynamics.cvxpy_identification import (
    CvxpyIdentificationResult,
    PhysicalParameterBlock,
    identify_cvxpy_constrained_ols,
)
from robot_calibration.dynamics.friction import linear_friction_regressor
from robot_calibration.dynamics.identification import IdentificationResult, identify_ols
from robot_calibration.dynamics.observation import Regressor
from robot_calibration.dynamics.validation import predict_torque


@dataclass(frozen=True)
class CurrentDrivenIdentificationResult:
    """Result of current-driven dynamics identification."""

    identification: IdentificationResult | BoundedIdentificationResult | CvxpyIdentificationResult
    dynamic_parameters: NDArray[np.float64]
    friction_parameters: NDArray[np.float64] | None
    observation_matrix: NDArray[np.float64]
    measured_torque: NDArray[np.float64]
    predicted_torque: NDArray[np.float64]


def estimate_current_torque(
    currents: ArrayLike,
    drive_gains: ArrayLike,
) -> NDArray[np.float64]:
    """Convert motor current samples to joint torques with per-joint gains."""

    current_array = np.asarray(currents, dtype=float)
    gains = np.asarray(drive_gains, dtype=float)
    if current_array.ndim != 2:
        raise ValueError("currents must be a 2D array")
    if gains.shape != (current_array.shape[1],):
        raise ValueError("drive_gains must have one value per joint")
    return current_array * gains


def _base_regressor(
    regressor_output: NDArray[np.float64],
    base_mapping: BaseParameterMapping | None,
) -> NDArray[np.float64]:
    if base_mapping is None:
        return regressor_output
    return regressor_output @ base_mapping.permutation[:, : base_mapping.rank]


def build_current_driven_observation_matrix(
    dataset: CalibrationDataset,
    regressor: Regressor,
    *,
    drive_gains: ArrayLike,
    base_mapping: BaseParameterMapping | None = None,
    include_friction: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Build ``[Y_base, Y_friction]`` and torque vector from current data."""

    if dataset.velocities is None or dataset.accelerations is None:
        raise ValueError("velocities and accelerations are required")
    if dataset.currents is None:
        raise ValueError("currents are required")

    torque = estimate_current_torque(dataset.currents, drive_gains)
    rows = []
    dynamic_width = None
    for q, qd, qdd in zip(dataset.positions, dataset.velocities, dataset.accelerations):
        yi = np.asarray(regressor(q, qd, qdd), dtype=float)
        if yi.ndim != 2 or yi.shape[0] != dataset.n_joints:
            raise ValueError("regressor output must have one row per joint")
        yi_base = _base_regressor(yi, base_mapping)
        if dynamic_width is None:
            dynamic_width = yi_base.shape[1]
        if include_friction:
            yi_base = np.concatenate([yi_base, linear_friction_regressor(qd)], axis=1)
        rows.append(yi_base)

    if dynamic_width is None:
        raise ValueError("dataset must contain at least one sample")

    return np.vstack(rows), torque.reshape(-1), int(dynamic_width)


def identify_current_driven_dynamics(
    dataset: CalibrationDataset,
    regressor: Regressor,
    *,
    drive_gains: ArrayLike,
    base_mapping: BaseParameterMapping | None = None,
    include_friction: bool = True,
    method: str = "ols",
    nonnegative_indices: list[int] | None = None,
    physical_consistency: bool = False,
    physical_block_width: int = 10,
    physical_has_motor_inertia: bool = False,
) -> CurrentDrivenIdentificationResult:
    """Estimate dynamic and optional friction parameters from motor currents."""

    observation_matrix, measured_torque, dynamic_width = (
        build_current_driven_observation_matrix(
            dataset,
            regressor,
            drive_gains=drive_gains,
            base_mapping=base_mapping,
            include_friction=include_friction,
        )
    )
    if physical_consistency and method != "cvxpy":
        raise ValueError("physical_consistency requires method='cvxpy'")
    if method == "ols":
        identification = identify_ols(observation_matrix, measured_torque)
    elif method == "bounded":
        identification = identify_bounded_ols(
            observation_matrix,
            measured_torque,
            nonnegative_indices=nonnegative_indices,
        )
    elif method == "cvxpy":
        physical_blocks = None
        if physical_consistency:
            if physical_block_width <= 0:
                raise ValueError("physical_block_width must be positive")
            physical_blocks = [
                PhysicalParameterBlock(
                    start=start,
                    has_motor_inertia=physical_has_motor_inertia,
                )
                for start in range(0, dynamic_width, physical_block_width)
            ]
        identification = identify_cvxpy_constrained_ols(
            observation_matrix,
            measured_torque,
            nonnegative_indices=nonnegative_indices,
            physical_parameter_blocks=physical_blocks,
        )
    else:
        raise ValueError("method must be 'ols', 'bounded', or 'cvxpy'")
    parameters = identification.parameters
    friction_parameters = parameters[dynamic_width:] if include_friction else None
    predicted = predict_torque(observation_matrix, parameters)
    return CurrentDrivenIdentificationResult(
        identification=identification,
        dynamic_parameters=parameters[:dynamic_width],
        friction_parameters=friction_parameters,
        observation_matrix=observation_matrix,
        measured_torque=measured_torque,
        predicted_torque=predicted,
    )
