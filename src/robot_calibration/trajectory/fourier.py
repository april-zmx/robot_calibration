"""Fourier and polynomial calibration trajectories."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _coefficient_arrays(
    sine_coefficients: ArrayLike,
    cosine_coefficients: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(sine_coefficients, dtype=float)
    b = np.asarray(cosine_coefficients, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("sine and cosine coefficients must be matching 2D arrays")
    return a, b


def fourier_series(
    time: ArrayLike,
    q0: ArrayLike,
    sine_coefficients: ArrayLike,
    cosine_coefficients: ArrayLike,
    fundamental_frequency: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate truncated Fourier position, velocity, and acceleration."""

    t = np.asarray(time, dtype=float)
    offset = np.asarray(q0, dtype=float)
    a, b = _coefficient_arrays(sine_coefficients, cosine_coefficients)
    if t.ndim != 1:
        raise ValueError("time must be 1D")
    if offset.shape != (a.shape[0],):
        raise ValueError("q0 must have one entry per joint")
    if fundamental_frequency <= 0.0:
        raise ValueError("fundamental_frequency must be positive")

    q = np.repeat(offset[:, None], t.size, axis=1)
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for harmonic in range(1, a.shape[1] + 1):
        w = fundamental_frequency * harmonic
        sin_wt = np.sin(w * t)
        cos_wt = np.cos(w * t)
        ai = a[:, harmonic - 1 : harmonic]
        bi = b[:, harmonic - 1 : harmonic]
        q = q + ai / w * sin_wt - bi / w * cos_wt
        qd = qd + ai * cos_wt + bi * sin_wt
        qdd = qdd - ai * w * sin_wt + bi * w * cos_wt
    return q, qd, qdd


class FourierExcitationTrajectory:
    """Truncated Fourier excitation trajectory."""

    def __init__(
        self,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
    ) -> None:
        self.q0 = np.asarray(q0, dtype=float)
        self.sine_coefficients, self.cosine_coefficients = _coefficient_arrays(
            sine_coefficients,
            cosine_coefficients,
        )
        if self.q0.shape != (self.sine_coefficients.shape[0],):
            raise ValueError("q0 must have one entry per joint")
        if fundamental_frequency <= 0.0:
            raise ValueError("fundamental_frequency must be positive")
        self.fundamental_frequency = float(fundamental_frequency)

    @property
    def n_joints(self) -> int:
        return self.sine_coefficients.shape[0]

    @property
    def n_harmonics(self) -> int:
        return self.sine_coefficients.shape[1]

    def sample(
        self,
        time: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return fourier_series(
            time,
            self.q0,
            self.sine_coefficients,
            self.cosine_coefficients,
            self.fundamental_frequency,
        )


class FifthOrderBoundaryPolynomial:
    """Fifth-order polynomial used to enforce trajectory boundary conditions."""

    def __init__(self, coefficients: ArrayLike) -> None:
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 2 or coeffs.shape[1] != 6:
            raise ValueError("coefficients must have shape (n_joints, 6)")
        self.coefficients = coeffs

    @classmethod
    def from_fourier_boundary(
        cls,
        duration: float,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
    ) -> "FifthOrderBoundaryPolynomial":
        return cls(
            fifth_order_boundary_coefficients(
                duration,
                sine_coefficients,
                cosine_coefficients,
                fundamental_frequency,
                q0,
            )
        )

    @property
    def n_joints(self) -> int:
        return self.coefficients.shape[0]

    def sample(
        self,
        time: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        t = np.asarray(time, dtype=float)
        if t.ndim != 1:
            raise ValueError("time must be 1D")
        powers = np.vstack([t**i for i in range(6)])
        q = self.coefficients @ powers
        qd = (
            self.coefficients[:, 1:2]
            + 2.0 * self.coefficients[:, 2:3] * t
            + 3.0 * self.coefficients[:, 3:4] * t**2
            + 4.0 * self.coefficients[:, 4:5] * t**3
            + 5.0 * self.coefficients[:, 5:6] * t**4
        )
        qdd = (
            2.0 * self.coefficients[:, 2:3]
            + 6.0 * self.coefficients[:, 3:4] * t
            + 12.0 * self.coefficients[:, 4:5] * t**2
            + 20.0 * self.coefficients[:, 5:6] * t**3
        )
        return q, qd, qdd


def fifth_order_boundary_coefficients(
    duration: float,
    sine_coefficients: ArrayLike,
    cosine_coefficients: ArrayLike,
    fundamental_frequency: float,
    q0: ArrayLike,
) -> NDArray[np.float64]:
    """Return polynomial coefficients that cancel Fourier boundary motion."""

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    a, b = _coefficient_arrays(sine_coefficients, cosine_coefficients)
    q_start, qd_start, qdd_start = fourier_series(
        np.array([0.0]),
        np.zeros(a.shape[0]),
        a,
        b,
        fundamental_frequency,
    )
    q_end, qd_end, qdd_end = fourier_series(
        np.array([duration]),
        np.zeros(a.shape[0]),
        a,
        b,
        fundamental_frequency,
    )
    target = np.asarray(q0, dtype=float)
    if target.shape != (a.shape[0],):
        raise ValueError("q0 must have one entry per joint")

    coeffs = np.zeros((a.shape[0], 6), dtype=float)
    coeffs[:, 0] = target - q_start[:, 0]
    coeffs[:, 1] = -qd_start[:, 0]
    coeffs[:, 2] = -0.5 * qdd_start[:, 0]

    system = np.array(
        [
            [duration**3, duration**4, duration**5],
            [3 * duration**2, 4 * duration**3, 5 * duration**4],
            [6 * duration, 12 * duration**2, 20 * duration**3],
        ],
        dtype=float,
    )
    for joint in range(a.shape[0]):
        known_end = (
            coeffs[joint, 0]
            + coeffs[joint, 1] * duration
            + coeffs[joint, 2] * duration**2
        )
        known_velocity = coeffs[joint, 1] + 2 * coeffs[joint, 2] * duration
        known_acceleration = 2 * coeffs[joint, 2]
        rhs = np.array(
            [
                target[joint] - q_end[joint, 0] - known_end,
                -qd_end[joint, 0] - known_velocity,
                -qdd_end[joint, 0] - known_acceleration,
            ],
            dtype=float,
        )
        coeffs[joint, 3:] = np.linalg.solve(system, rhs)
    return coeffs


def mixed_trajectory(
    time: ArrayLike,
    polynomial_coefficients: ArrayLike,
    sine_coefficients: ArrayLike,
    cosine_coefficients: ArrayLike,
    fundamental_frequency: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate Fourier plus fifth-order polynomial trajectory."""

    t = np.asarray(time, dtype=float)
    coeffs = np.asarray(polynomial_coefficients, dtype=float)
    a, b = _coefficient_arrays(sine_coefficients, cosine_coefficients)
    if coeffs.shape != (a.shape[0], 6):
        raise ValueError("polynomial_coefficients must have shape (n_joints, 6)")

    qh, qhd, qhdd = fourier_series(t, np.zeros(a.shape[0]), a, b, fundamental_frequency)
    powers = np.vstack([t**i for i in range(6)])
    q_poly = coeffs @ powers
    qd_poly = (
        coeffs[:, 1:2]
        + 2 * coeffs[:, 2:3] * t
        + 3 * coeffs[:, 3:4] * t**2
        + 4 * coeffs[:, 4:5] * t**3
        + 5 * coeffs[:, 5:6] * t**4
    )
    qdd_poly = (
        2 * coeffs[:, 2:3]
        + 6 * coeffs[:, 3:4] * t
        + 12 * coeffs[:, 4:5] * t**2
        + 20 * coeffs[:, 5:6] * t**3
    )
    return qh + q_poly, qhd + qd_poly, qhdd + qdd_poly


class MixedExcitationTrajectory:
    """Fourier excitation trajectory with fifth-order boundary correction."""

    def __init__(
        self,
        fourier: FourierExcitationTrajectory,
        boundary_polynomial: FifthOrderBoundaryPolynomial,
        duration: float | None = None,
    ) -> None:
        if fourier.n_joints != boundary_polynomial.n_joints:
            raise ValueError("fourier and boundary polynomial joint counts must match")
        if duration is not None and duration <= 0.0:
            raise ValueError("duration must be positive")
        self.fourier = fourier
        self.boundary_polynomial = boundary_polynomial
        self.duration = None if duration is None else float(duration)

    @classmethod
    def from_fourier_boundary(
        cls,
        duration: float,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
    ) -> "MixedExcitationTrajectory":
        fourier = FourierExcitationTrajectory(
            np.zeros_like(np.asarray(q0, dtype=float)),
            sine_coefficients,
            cosine_coefficients,
            fundamental_frequency,
        )
        boundary = FifthOrderBoundaryPolynomial.from_fourier_boundary(
            duration,
            q0,
            sine_coefficients,
            cosine_coefficients,
            fundamental_frequency,
        )
        return cls(fourier, boundary, duration=duration)

    @property
    def n_joints(self) -> int:
        return self.fourier.n_joints

    def sample(
        self,
        time: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        q_fourier, qd_fourier, qdd_fourier = self.fourier.sample(time)
        q_boundary, qd_boundary, qdd_boundary = self.boundary_polynomial.sample(time)
        return (
            q_fourier + q_boundary,
            qd_fourier + qd_boundary,
            qdd_fourier + qdd_boundary,
        )

    def sample_uniform(
        self,
        sample_count: int,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        if self.duration is None:
            raise ValueError("duration is required for uniform sampling")
        if sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        time = np.linspace(0.0, self.duration, sample_count)
        q, qd, qdd = self.sample(time)
        return time, q, qd, qdd


class ExcitationTrajectoryGenerator:
    """Factory for boundary-corrected Fourier excitation trajectories."""

    def __init__(
        self,
        q0: ArrayLike,
        sine_coefficients: ArrayLike,
        cosine_coefficients: ArrayLike,
        fundamental_frequency: float,
        duration: float,
    ) -> None:
        if duration <= 0.0:
            raise ValueError("duration must be positive")
        self.q0 = np.asarray(q0, dtype=float)
        self.sine_coefficients, self.cosine_coefficients = _coefficient_arrays(
            sine_coefficients,
            cosine_coefficients,
        )
        if self.q0.shape != (self.sine_coefficients.shape[0],):
            raise ValueError("q0 must have one entry per joint")
        if fundamental_frequency <= 0.0:
            raise ValueError("fundamental_frequency must be positive")
        self.fundamental_frequency = float(fundamental_frequency)
        self.duration = float(duration)

    def build(self) -> MixedExcitationTrajectory:
        return MixedExcitationTrajectory.from_fourier_boundary(
            self.duration,
            self.q0,
            self.sine_coefficients,
            self.cosine_coefficients,
            self.fundamental_frequency,
        )
