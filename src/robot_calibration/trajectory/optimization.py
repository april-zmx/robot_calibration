"""Excitation trajectory optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robot_calibration.dynamics.base_parameters import BaseParameterMapping

from robot_calibration.trajectory.fourier import (
    ExcitationTrajectoryGenerator,
    MixedExcitationTrajectory,
)

try:
    from pymoo.core.problem import Problem
except ModuleNotFoundError:
    class Problem:  # type: ignore[no-redef]
        """Lightweight fallback so metric evaluation works without pymoo."""

        def __init__(
            self,
            *,
            n_var: int,
            n_obj: int,
            n_ieq_constr: int = 0,
            xl: ArrayLike | None,
            xu: ArrayLike | None,
            vtype=float,
        ) -> None:
            self.n_var = int(n_var)
            self.n_obj = int(n_obj)
            self.n_ieq_constr = int(n_ieq_constr)
            self.xl = None if xl is None else np.asarray(xl, dtype=float)
            self.xu = None if xu is None else np.asarray(xu, dtype=float)
            self.vtype = vtype

Regressor = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    NDArray[np.float64],
]
FrictionRegressor = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class ObservationMatrixMetrics:
    """Conditioning metrics for one excitation trajectory."""

    condition_number: float
    objective_value: float
    observation_rank: int
    target_rank: int


def _coefficient_bounds(
    coefficient_bounds: tuple[float, float] | ArrayLike | None,
    n_var: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    if coefficient_bounds is None:
        return None, None
    bounds = np.asarray(coefficient_bounds, dtype=float)
    if bounds.shape == (2,):
        lower = np.full(n_var, bounds[0], dtype=float)
        upper = np.full(n_var, bounds[1], dtype=float)
    elif bounds.shape == (n_var, 2):
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    else:
        raise ValueError("coefficient_bounds must have shape (2,) or (n_var, 2)")
    if np.any(lower > upper):
        raise ValueError("coefficient lower bounds must be <= upper bounds")
    return lower, upper


def _unpack_coefficients(
    vector: ArrayLike,
    n_joints: int,
    n_harmonics: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(vector, dtype=float)
    expected_size = 2 * n_joints * n_harmonics
    if values.shape != (expected_size,):
        raise ValueError("coefficient vector has incompatible size")
    split = n_joints * n_harmonics
    sine = values[:split].reshape(n_joints, n_harmonics)
    cosine = values[split:].reshape(n_joints, n_harmonics)
    return sine, cosine


def _joint_limit_array(
    limits: ArrayLike | None,
    n_joints: int,
    *,
    name: str,
    symmetric: bool = False,
) -> NDArray[np.float64] | None:
    if limits is None:
        return None
    values = np.asarray(limits, dtype=float)
    if symmetric and values.shape == (n_joints,):
        values = np.column_stack((-values, values))
    if values.shape != (n_joints, 2):
        raise ValueError(f"{name} must have shape (n_joints, 2)")
    if np.any(values[:, 0] > values[:, 1]):
        raise ValueError(f"{name} lower limits must be <= upper limits")
    return values


def _condition_number_and_rank(
    matrix: ArrayLike,
    rcond: float,
) -> tuple[float, int]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("observation matrix must be 2D")
    if values.size == 0:
        return np.inf, 0
    condition_number = float(np.linalg.cond(values))
    rank = int(np.linalg.matrix_rank(values, tol=rcond))
    return condition_number, rank


class ExcitationConditionNumberProblem(Problem):
    """Pymoo problem minimizing the trajectory observation matrix condition number."""

    def __init__(
        self,
        q0: ArrayLike,
        n_harmonics: int,
        fundamental_frequency: float,
        duration: float,
        sample_count: int,
        coefficient_bounds: tuple[float, float] | ArrayLike | None,
        regressor: Regressor,
        *,
        base_parameter_mapping: BaseParameterMapping | None = None,
        friction_regressor: FrictionRegressor | None = None,
        joint_position_limits: ArrayLike | None = None,
        joint_velocity_limits: ArrayLike | None = None,
        joint_acceleration_limits: ArrayLike | None = None,
        singular_penalty: float = 1.0e12,
        condition_rcond: float = 1.0e-10,
    ) -> None:
        self.q0 = np.asarray(q0, dtype=float)
        if self.q0.ndim != 1:
            raise ValueError("q0 must be 1D")
        if n_harmonics < 1:
            raise ValueError("n_harmonics must be at least 1")
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if fundamental_frequency <= 0.0:
            raise ValueError("fundamental_frequency must be positive")
        self.n_joints = self.q0.size
        self.n_harmonics = int(n_harmonics)
        self.fundamental_frequency = float(fundamental_frequency)
        self.duration = float(duration)
        self.sample_count = int(sample_count)
        self.regressor = regressor
        self.base_parameter_mapping = base_parameter_mapping
        self.friction_regressor = friction_regressor
        self.singular_penalty = float(singular_penalty)
        self.condition_rcond = float(condition_rcond)
        self.joint_position_limits = _joint_limit_array(
            joint_position_limits,
            self.n_joints,
            name="joint_position_limits",
        )
        self.joint_velocity_limits = _joint_limit_array(
            joint_velocity_limits,
            self.n_joints,
            name="joint_velocity_limits",
            symmetric=True,
        )
        self.joint_acceleration_limits = _joint_limit_array(
            joint_acceleration_limits,
            self.n_joints,
            name="joint_acceleration_limits",
            symmetric=True,
        )

        n_var = 2 * self.n_joints * self.n_harmonics
        lower, upper = _coefficient_bounds(coefficient_bounds, n_var)
        n_ieq_constr = 0
        for limits in (
            self.joint_position_limits,
            self.joint_velocity_limits,
            self.joint_acceleration_limits,
        ):
            if limits is not None:
                n_ieq_constr += 2 * self.n_joints
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_ieq_constr=n_ieq_constr,
            xl=lower,
            xu=upper,
            vtype=float,
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        candidates = np.atleast_2d(np.asarray(x, dtype=float))
        out["F"] = np.array(
            [[self.evaluate_objective(candidate)] for candidate in candidates],
            dtype=float,
        )
        if self.n_ieq_constr:
            out["G"] = np.array(
                [self.evaluate_limit_constraints(candidate) for candidate in candidates],
                dtype=float,
            )

    def evaluate_condition_number(self, vector: ArrayLike) -> float:
        return self.evaluate_observation_metrics(vector).condition_number

    def evaluate_objective(self, vector: ArrayLike) -> float:
        return self.evaluate_observation_metrics(vector).objective_value

    def evaluate_observation_metrics(self, vector: ArrayLike) -> ObservationMatrixMetrics:
        matrix = self.observation_matrix_from_vector(vector)
        value, rank = _condition_number_and_rank(
            matrix,
            self.condition_rcond,
        )
        target_rank = min(matrix.shape)
        objective_value = value
        if rank < target_rank:
            finite_value = value if np.isfinite(value) else 0.0
            objective_value = max(self.singular_penalty, float(finite_value)) + float(
                target_rank - rank
            )
        elif not np.isfinite(value):
            objective_value = self.singular_penalty
        return ObservationMatrixMetrics(
            condition_number=value,
            objective_value=objective_value,
            observation_rank=rank,
            target_rank=target_rank,
        )

    def observation_matrix_from_vector(self, vector: ArrayLike) -> NDArray[np.float64]:
        sine, cosine = _unpack_coefficients(
            vector,
            self.n_joints,
            self.n_harmonics,
        )
        generator = ExcitationTrajectoryGenerator(
            q0=self.q0,
            sine_coefficients=sine,
            cosine_coefficients=cosine,
            fundamental_frequency=self.fundamental_frequency,
            duration=self.duration,
        )
        trajectory = generator.build()
        _, q, qd, qdd = trajectory.sample_uniform(self.sample_count)
        rows = []
        for i in range(self.sample_count):
            yi = np.asarray(self.regressor(q[:, i], qd[:, i], qdd[:, i]), dtype=float)
            if yi.ndim != 2 or yi.shape[0] != self.n_joints:
                raise ValueError("regressor output must have one row per joint")
            if self.base_parameter_mapping is not None:
                projection = self.base_parameter_mapping.base_projection()
                if projection.shape[0] != yi.shape[1]:
                    raise ValueError(
                        "base_parameter_mapping is incompatible with regressor columns"
                    )
                yi = yi @ projection
            if self.friction_regressor is not None:
                yf = np.asarray(self.friction_regressor(qd[:, i]), dtype=float)
                if yf.ndim != 2 or yf.shape[0] != self.n_joints:
                    raise ValueError(
                        "friction_regressor output must have one row per joint"
                    )
                yi = np.concatenate([yi, yf], axis=1)
            rows.append(yi)
        return np.vstack(rows)

    def evaluate_limit_constraints(self, vector: ArrayLike) -> NDArray[np.float64]:
        _, q, qd, qdd = self.trajectory_from_vector(vector).sample_uniform(
            self.sample_count
        )
        constraints = []
        for samples, limits in (
            (q, self.joint_position_limits),
            (qd, self.joint_velocity_limits),
            (qdd, self.joint_acceleration_limits),
        ):
            if limits is None:
                continue
            lower = limits[:, 0:1]
            upper = limits[:, 1:2]
            constraints.extend(np.max(samples - upper, axis=1))
            constraints.extend(np.max(lower - samples, axis=1))
        return np.asarray(constraints, dtype=float)

    def coefficients_from_vector(
        self,
        vector: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return _unpack_coefficients(vector, self.n_joints, self.n_harmonics)

    def trajectory_from_vector(self, vector: ArrayLike) -> MixedExcitationTrajectory:
        sine, cosine = self.coefficients_from_vector(vector)
        return ExcitationTrajectoryGenerator(
            q0=self.q0,
            sine_coefficients=sine,
            cosine_coefficients=cosine,
            fundamental_frequency=self.fundamental_frequency,
            duration=self.duration,
        ).build()


@dataclass(frozen=True)
class ExcitationOptimizationResult:
    """Result of condition-number-based excitation optimization."""

    trajectory: MixedExcitationTrajectory
    sine_coefficients: NDArray[np.float64]
    cosine_coefficients: NDArray[np.float64]
    condition_number: float
    objective_value: float
    observation_rank: int
    target_rank: int
    decision_vector: NDArray[np.float64]
    raw_result: object
    used_initial_guess: bool = False


class ConditionNumberExcitationOptimizer:
    """Optimize Fourier excitation coefficients with pymoo."""

    def __init__(
        self,
        q0: ArrayLike,
        n_harmonics: int,
        fundamental_frequency: float,
        duration: float,
        sample_count: int,
        coefficient_bounds: tuple[float, float] | ArrayLike | None,
        regressor: Regressor,
        *,
        base_parameter_mapping: BaseParameterMapping | None = None,
        friction_regressor: FrictionRegressor | None = None,
        joint_position_limits: ArrayLike | None = None,
        joint_velocity_limits: ArrayLike | None = None,
        joint_acceleration_limits: ArrayLike | None = None,
    ) -> None:
        self.problem = ExcitationConditionNumberProblem(
            q0=q0,
            n_harmonics=n_harmonics,
            fundamental_frequency=fundamental_frequency,
            duration=duration,
            sample_count=sample_count,
            coefficient_bounds=coefficient_bounds,
            regressor=regressor,
            base_parameter_mapping=base_parameter_mapping,
            friction_regressor=friction_regressor,
            joint_position_limits=joint_position_limits,
            joint_velocity_limits=joint_velocity_limits,
            joint_acceleration_limits=joint_acceleration_limits,
        )

    def optimize(
        self,
        *,
        max_evaluations: int = 1_000_000,
        seed: int | None = None,
        algorithm=None,
        verbose: bool = False,
    ) -> ExcitationOptimizationResult:
        if max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1")
        try:
            from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
            from pymoo.optimize import minimize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pymoo is required for trajectory optimization"
            ) from exc
        selected_algorithm = (
            PatternSearch(x0=self._default_initial_guess(seed))
            if algorithm is None
            else algorithm
        )
        raw = minimize(
            self.problem,
            selected_algorithm,
            termination=("n_eval", max_evaluations),
            seed=seed,
            verbose=verbose,
        )
        decision_vector, used_initial_guess = self._decision_vector_from_result(raw)
        sine, cosine = self.problem.coefficients_from_vector(decision_vector)
        trajectory = self.problem.trajectory_from_vector(decision_vector)
        metrics = self.problem.evaluate_observation_metrics(decision_vector)
        return ExcitationOptimizationResult(
            trajectory=trajectory,
            sine_coefficients=sine,
            cosine_coefficients=cosine,
            condition_number=metrics.condition_number,
            objective_value=metrics.objective_value,
            observation_rank=metrics.observation_rank,
            target_rank=metrics.target_rank,
            decision_vector=decision_vector,
            raw_result=raw,
            used_initial_guess=used_initial_guess,
        )

    def _default_initial_guess(self, seed: int | None = None) -> NDArray[np.float64]:
        zero = np.zeros(self.problem.n_var, dtype=float)
        if self.problem.n_ieq_constr:
            constraints = self.problem.evaluate_limit_constraints(zero)
            if np.all(constraints <= 0.0):
                return zero

        rng = np.random.default_rng(seed)
        if self.problem.xl is None or self.problem.xu is None:
            return rng.random(self.problem.n_var)
        lower = np.asarray(self.problem.xl, dtype=float)
        upper = np.asarray(self.problem.xu, dtype=float)
        if lower.shape != (self.problem.n_var,) or upper.shape != (self.problem.n_var,):
            raise ValueError("problem bounds must match decision-variable count")
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            return rng.uniform(lower, upper)
        return rng.random(self.problem.n_var)

    def _decision_vector_from_result(self, raw_result) -> tuple[NDArray[np.float64], bool]:
        decision_vector = np.asarray(raw_result.X, dtype=float)
        if (
            decision_vector.shape == (self.problem.n_var,)
            and np.all(np.isfinite(decision_vector))
        ):
            return decision_vector, bool(np.allclose(decision_vector, 0.0))

        zero = np.zeros(self.problem.n_var, dtype=float)
        if self.problem.n_ieq_constr == 0:
            return zero, True
        constraints = self.problem.evaluate_limit_constraints(zero)
        if np.all(constraints <= 0.0):
            return zero, True

        raise RuntimeError(
            "pymoo did not return a feasible excitation trajectory; "
            "increase max_evaluations or relax joint/velocity/acceleration limits"
        )
