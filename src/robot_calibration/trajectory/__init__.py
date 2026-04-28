"""Excitation trajectory generation."""

from robot_calibration.trajectory.fourier import (
    ExcitationTrajectoryGenerator,
    FifthOrderBoundaryPolynomial,
    FourierExcitationTrajectory,
    MixedExcitationTrajectory,
    fifth_order_boundary_coefficients,
    fourier_series,
    mixed_trajectory,
)
from robot_calibration.trajectory.optimization import (
    ConditionNumberExcitationOptimizer,
    ExcitationConditionNumberProblem,
    ExcitationOptimizationResult,
    ObservationMatrixMetrics,
)

__all__ = [
    "ExcitationTrajectoryGenerator",
    "ConditionNumberExcitationOptimizer",
    "ExcitationConditionNumberProblem",
    "ExcitationOptimizationResult",
    "FifthOrderBoundaryPolynomial",
    "FourierExcitationTrajectory",
    "MixedExcitationTrajectory",
    "ObservationMatrixMetrics",
    "fifth_order_boundary_coefficients",
    "fourier_series",
    "mixed_trajectory",
]
