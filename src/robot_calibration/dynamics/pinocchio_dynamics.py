"""Notebook-style Pinocchio dynamics workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_calibration.data import CalibrationDataset
from robot_calibration.dynamics.base_parameters import (
    BaseParameterMapping,
    extract_base_parameters,
)
from robot_calibration.dynamics.current_identification import estimate_current_torque
from robot_calibration.dynamics.friction import linear_friction_regressor
from robot_calibration.dynamics.pinocchio_adapter import (
    PinocchioRegressor,
    _load_pinocchio,
    _load_robot_wrapper,
)
from robot_calibration.dynamics.validation import predict_torque


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass(frozen=True)
class BaseDynamicsParameters:
    """QR-derived minimum dynamic-parameter set."""

    mapping: BaseParameterMapping
    full_parameters: NDArray[np.float64]
    base_parameters: NDArray[np.float64]
    dependent_parameters: NDArray[np.float64]
    observation_matrix: NDArray[np.float64]
    torque_samples: NDArray[np.float64]

    @property
    def num_base_params(self) -> int:
        return self.mapping.num_base_params

    @property
    def num_dep_params(self) -> int:
        return self.mapping.num_dep_params

    @property
    def permutation_matrix(self) -> NDArray[np.float64]:
        return self.mapping.permutation_matrix

    @property
    def beta(self) -> NDArray[np.float64]:
        return self.mapping.beta

    @property
    def base_projection(self) -> NDArray[np.float64]:
        return self.mapping.base_projection()


@dataclass(frozen=True)
class BaseDynamicsIdentificationResult:
    """Notebook-style base-parameter identification result."""

    base_parameters: NDArray[np.float64]
    full_parameters: NDArray[np.float64] | None
    friction_parameters: NDArray[np.float64]
    estimated_standard_deviation: NDArray[np.float64]
    estimated_relative_standard_deviation: NDArray[np.float64]
    observation_matrix: NDArray[np.float64]
    measured_torque: NDArray[np.float64]
    predicted_torque: NDArray[np.float64]
    objective_value: float
    status: str
    success: bool

    @property
    def est_std(self) -> NDArray[np.float64]:
        return self.estimated_standard_deviation

    @property
    def est_rel_std(self) -> NDArray[np.float64]:
        return self.estimated_relative_standard_deviation

    @property
    def pi_base(self) -> NDArray[np.float64]:
        return self.base_parameters

    @property
    def pi_full(self) -> NDArray[np.float64] | None:
        return self.full_parameters

    @property
    def pi_frictions(self) -> NDArray[np.float64]:
        return self.friction_parameters


@dataclass(frozen=True)
class PinocchioDynamicsModel:
    """Pinocchio-backed dynamics helpers matching the notebook workflow."""

    robot: Any
    include_motor_dynamics: bool = False
    link_parameter_count: int = 10
    pinocchio: Any | None = "auto"
    _regressor: PinocchioRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.link_parameter_count <= 0:
            raise ValueError("link_parameter_count must be positive")
        object.__setattr__(
            self,
            "_regressor",
            PinocchioRegressor(
                self.robot,
                include_motor_dynamics=self.include_motor_dynamics,
                link_parameter_count=self.link_parameter_count,
                pinocchio=self.pinocchio,
            ),
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        *,
        include_motor_dynamics: bool = False,
        link_parameter_count: int = 10,
        model_only: bool = True,
        verbose: bool = False,
        pinocchio: Any | None = "auto",
        robot_wrapper_cls: Any | None = "auto",
    ) -> "PinocchioDynamicsModel":
        pin = _load_pinocchio() if pinocchio == "auto" else pinocchio
        if pin is None:
            raise ModuleNotFoundError(
                "pinocchio is required for PinocchioDynamicsModel.from_urdf"
            )
        if model_only:
            model = pin.buildModelFromUrdf(str(urdf_path))
            robot = SimpleNamespace(model=model, data=model.createData())
        else:
            wrapper_cls = (
                _load_robot_wrapper()
                if robot_wrapper_cls == "auto"
                else robot_wrapper_cls
            )
            if wrapper_cls is None:
                raise ModuleNotFoundError(
                    "pinocchio.robot_wrapper is required when model_only=False"
                )
            model, collision_model, visual_model = pin.buildModelsFromUrdf(
                str(urdf_path),
                verbose=verbose,
            )
            robot = wrapper_cls(model, collision_model, visual_model)
        return cls(
            robot,
            include_motor_dynamics=include_motor_dynamics,
            link_parameter_count=link_parameter_count,
            pinocchio=pin,
        )

    @property
    def nq(self) -> int:
        return int(self.robot.model.nq)

    @property
    def parameter_block_width(self) -> int:
        return self.link_parameter_count + (1 if self.include_motor_dynamics else 0)

    def regressor(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        qdd: ArrayLike,
    ) -> NDArray[np.float64]:
        return self._regressor(q, qd, qdd)

    def friction_regressor(self, qd: ArrayLike) -> NDArray[np.float64]:
        return linear_friction_regressor(qd)

    def dynamic_parameter_vector(
        self,
        *,
        include_friction_model: bool = False,
    ) -> NDArray[np.float64]:
        parameters = []
        for joint in range(1, self.nq + 1):
            block = np.asarray(
                self.robot.model.inertias[joint].toDynamicParameters(),
                dtype=float,
            ).reshape(-1)
            if block.shape != (self.link_parameter_count,):
                raise ValueError(
                    "dynamic parameter block size does not match link_parameter_count"
                )
            parameters.append(block)
            if self.include_motor_dynamics:
                parameters.append(np.zeros(1, dtype=float))
        if include_friction_model:
            parameters.append(np.zeros(self.nq * 3, dtype=float))
        return np.concatenate(parameters, axis=0)

    def random_state_samples(
        self,
        sample_count: int,
        *,
        rng: np.random.Generator | int | None = None,
        acceleration_limit_scale: float = 2.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        if sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if acceleration_limit_scale <= 0.0:
            raise ValueError("acceleration_limit_scale must be positive")
        generator = _as_rng(rng)
        q_min = np.asarray(self.robot.model.lowerPositionLimit, dtype=float).reshape(-1)
        q_max = np.asarray(self.robot.model.upperPositionLimit, dtype=float).reshape(-1)
        qd_max = np.asarray(self.robot.model.velocityLimit, dtype=float).reshape(-1)
        if q_min.shape != (self.nq,) or q_max.shape != (self.nq,) or qd_max.shape != (self.nq,):
            raise ValueError("robot limits must have one entry per joint")
        qdd_max = acceleration_limit_scale * qd_max
        q = np.empty((sample_count, self.nq), dtype=float)
        qd = np.empty((sample_count, self.nq), dtype=float)
        qdd = np.empty((sample_count, self.nq), dtype=float)
        # Match the notebook's RNG consumption order exactly: q, qd, qdd per sample.
        for sample in range(sample_count):
            q[sample] = generator.uniform(q_min, q_max)
            qd[sample] = generator.uniform(-qd_max, qd_max)
            qdd[sample] = generator.uniform(-qdd_max, qdd_max)
        return q, qd, qdd

    def extract_base_parameters(
        self,
        sample_count: int,
        *,
        rng: np.random.Generator | int | None = None,
        acceleration_limit_scale: float = 2.0,
        tolerance: float | None = None,
    ) -> BaseDynamicsParameters:
        q, qd, qdd = self.random_state_samples(
            sample_count,
            rng=rng,
            acceleration_limit_scale=acceleration_limit_scale,
        )
        full_parameters = self.dynamic_parameter_vector()
        observation_rows = []
        torque_rows = []
        for sample_q, sample_qd, sample_qdd in zip(q, qd, qdd):
            y = self.regressor(sample_q, sample_qd, sample_qdd)
            observation_rows.append(y)
            torque_rows.append(y @ full_parameters)
        observation_matrix = np.vstack(observation_rows)
        torque_samples = np.asarray(torque_rows, dtype=float).reshape(-1)
        mapping = extract_base_parameters(observation_matrix, tolerance=tolerance)
        sorted_parameters = mapping.sort_parameters(full_parameters)
        dependent_parameters = sorted_parameters[mapping.rank :]
        return BaseDynamicsParameters(
            mapping=mapping,
            full_parameters=full_parameters,
            base_parameters=mapping.base_parameter_vector(full_parameters),
            dependent_parameters=dependent_parameters,
            observation_matrix=observation_matrix,
            torque_samples=torque_samples,
        )


class CurrentDrivenBaseDynamicsIdentifier:
    """Current-driven base-parameter identification following the notebook."""

    def __init__(
        self,
        dynamics_model: PinocchioDynamicsModel,
        base_parameters: BaseDynamicsParameters,
        *,
        drive_gains: ArrayLike,
    ) -> None:
        self.dynamics_model = dynamics_model
        self.base_parameters = base_parameters
        self.drive_gains = np.asarray(drive_gains, dtype=float).reshape(-1)
        if self.drive_gains.shape != (self.dynamics_model.nq,):
            raise ValueError("drive_gains must have one value per joint")

    def build_observation_matrix(
        self,
        dataset: CalibrationDataset,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if dataset.velocities is None or dataset.accelerations is None:
            raise ValueError("velocities and accelerations are required")
        if dataset.currents is None:
            raise ValueError("currents are required")

        e1 = self.base_parameters.base_projection
        rows = []
        for q, qd, qdd in zip(
            dataset.positions,
            dataset.velocities,
            dataset.accelerations,
        ):
            y = self.dynamics_model.regressor(q, qd, qdd) @ e1
            y_friction = self.dynamics_model.friction_regressor(qd)
            rows.append(np.concatenate([y, y_friction], axis=1))
        torque = estimate_current_torque(dataset.currents, self.drive_gains)
        return np.vstack(rows), torque.reshape(-1)

    def identify(
        self,
        dataset: CalibrationDataset,
        *,
        method: str = "least_square",
        solver: str = "CLARABEL",
        mass_error_range: float = 0.1,
    ) -> BaseDynamicsIdentificationResult:
        observation_matrix, measured_torque = self.build_observation_matrix(dataset)
        if method == "least_square":
            parameters = self._solve_least_square(observation_matrix, measured_torque)
            base_parameters = parameters[: self.base_parameters.num_base_params]
            friction_parameters = parameters[self.base_parameters.num_base_params :]
            full_parameters = None
            objective_value = float(
                np.linalg.norm(measured_torque - observation_matrix @ parameters)
            )
            status = "solved"
            success = True
        elif method == "sdp":
            (
                base_parameters,
                full_parameters,
                friction_parameters,
                objective_value,
                status,
                success,
            ) = self._solve_sdp(
                observation_matrix,
                measured_torque,
                solver=solver,
                mass_error_range=mass_error_range,
            )
            parameters = np.concatenate([base_parameters, friction_parameters], axis=0)
        else:
            raise ValueError("method must be 'least_square' or 'sdp'")

        predicted_torque = predict_torque(observation_matrix, parameters)
        estimated_standard_deviation, estimated_relative_standard_deviation = (
            self._estimate_statistics(
                observation_matrix,
                measured_torque,
                parameters,
            )
        )
        return BaseDynamicsIdentificationResult(
            base_parameters=base_parameters,
            full_parameters=full_parameters,
            friction_parameters=friction_parameters,
            estimated_standard_deviation=estimated_standard_deviation,
            estimated_relative_standard_deviation=estimated_relative_standard_deviation,
            observation_matrix=observation_matrix,
            measured_torque=measured_torque,
            predicted_torque=predicted_torque,
            objective_value=objective_value,
            status=status,
            success=success,
        )

    def _solve_least_square(
        self,
        observation_matrix: NDArray[np.float64],
        measured_torque: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        normal = observation_matrix.T @ observation_matrix
        rhs = observation_matrix.T @ measured_torque
        return np.linalg.solve(normal, rhs)

    def _solve_sdp(
        self,
        observation_matrix: NDArray[np.float64],
        measured_torque: NDArray[np.float64],
        *,
        solver: str,
        mass_error_range: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        str,
        bool,
    ]:
        if self.dynamics_model.link_parameter_count != 10:
            raise ValueError(
                "sdp identification requires 10 inertial parameters per link"
            )
        try:
            import cvxpy as cp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "cvxpy is required for sdp identification"
            ) from exc

        nb = self.base_parameters.num_base_params
        nd = self.base_parameters.num_dep_params
        full_dynamic_size = self.base_parameters.full_parameters.size
        friction_width = self.dynamics_model.nq * 3

        pi_base = cp.Variable(nb)
        pi_dep = None if nd == 0 else cp.Variable(nd)
        pi_friction = cp.Variable(friction_width)

        sorted_parameters = (
            pi_base if nd == 0 else cp.hstack([pi_base, pi_dep])
        )
        reconstruction = self.base_parameters.mapping.reconstruction_matrix()
        pi_dynamic = reconstruction @ sorted_parameters

        constraints = []
        masses = np.array(
            [inertia.mass for inertia in self.dynamics_model.robot.model.inertias[1:]],
            dtype=float,
        )
        mass_upper_bounds = masses * (1.0 + mass_error_range)
        mass_indices = np.arange(
            0,
            full_dynamic_size,
            self.dynamics_model.parameter_block_width,
        )
        constraints.append(pi_dynamic[mass_indices] >= 0.0)
        constraints.append(pi_dynamic[mass_indices] <= mass_upper_bounds)

        for block_start in range(
            0,
            full_dynamic_size,
            self.dynamics_model.parameter_block_width,
        ):
            inertia_matrix = cp.bmat(
                [
                    [
                        pi_dynamic[block_start + 4],
                        pi_dynamic[block_start + 5],
                        pi_dynamic[block_start + 7],
                    ],
                    [
                        pi_dynamic[block_start + 5],
                        pi_dynamic[block_start + 6],
                        pi_dynamic[block_start + 8],
                    ],
                    [
                        pi_dynamic[block_start + 7],
                        pi_dynamic[block_start + 8],
                        pi_dynamic[block_start + 9],
                    ],
                ]
            )
            first_moment = pi_dynamic[block_start + 1 : block_start + 4]
            pseudo_inertia = cp.bmat(
                [
                    [
                        0.5 * cp.trace(inertia_matrix) * np.eye(3) - inertia_matrix,
                        first_moment[:, None],
                    ],
                    [
                        first_moment[None, :],
                        cp.reshape(pi_dynamic[block_start], (1, 1), order="C"),
                    ],
                ]
            )
            constraints.append(pseudo_inertia >> 0)
            if self.dynamics_model.include_motor_dynamics:
                constraints.append(
                    pi_dynamic[block_start + self.dynamics_model.parameter_block_width - 1]
                    >= 0.0
                )

        constraints.append(pi_friction[0::3] >= 0.0)
        constraints.append(pi_friction[1::3] >= 0.0)

        delta = measured_torque - observation_matrix @ cp.hstack([pi_base, pi_friction])
        problem = cp.Problem(cp.Minimize(cp.pnorm(delta)), constraints)
        problem.solve(solver=solver, verbose=False, max_iter=2000)

        base_value = np.asarray(pi_base.value, dtype=float).reshape(-1)
        dep_value = (
            np.zeros(0, dtype=float)
            if pi_dep is None
            else np.asarray(pi_dep.value, dtype=float).reshape(-1)
        )
        friction_value = np.asarray(pi_friction.value, dtype=float).reshape(-1)
        full_value = reconstruction @ np.concatenate([base_value, dep_value], axis=0)
        success = problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        return (
            base_value,
            full_value,
            friction_value,
            float(problem.value) if problem.value is not None else np.nan,
            str(problem.status),
            success,
        )

    def _estimate_statistics(
        self,
        observation_matrix: NDArray[np.float64],
        measured_torque: NDArray[np.float64],
        parameters: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        residual = measured_torque - observation_matrix @ parameters
        dof = observation_matrix.shape[0] - observation_matrix.shape[1]
        if dof <= 0:
            variance = 0.0
        else:
            variance = float(residual @ residual / dof)
        covariance = variance * np.linalg.pinv(observation_matrix.T @ observation_matrix)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        relative = np.divide(
            100.0 * standard_deviation,
            np.abs(parameters),
            out=np.full_like(standard_deviation, np.inf),
            where=np.abs(parameters) > 0.0,
        )
        return standard_deviation, relative
