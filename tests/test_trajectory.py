import numpy as np
import pytest

from robot_calibration.trajectory import (
    ExcitationTrajectoryGenerator,
    FifthOrderBoundaryPolynomial,
    FourierExcitationTrajectory,
    MixedExcitationTrajectory,
    fifth_order_boundary_coefficients,
    fourier_series,
    mixed_trajectory,
)


def test_fourier_series_shapes_match_time_and_joints():
    time = np.linspace(0.0, 1.0, 5)
    a = np.ones((2, 3))
    b = np.zeros((2, 3))

    q, qd, qdd = fourier_series(time, np.zeros(2), a, b, 2.0 * np.pi)

    assert q.shape == (2, 5)
    assert qd.shape == (2, 5)
    assert qdd.shape == (2, 5)


def test_fourier_excitation_trajectory_matches_function_helper():
    time = np.linspace(0.0, 1.0, 6)
    q0 = np.array([0.2, -0.4])
    a = np.array([[0.1, -0.2], [0.3, 0.05]])
    b = np.array([[0.05, 0.02], [-0.1, 0.04]])
    omega = 2.0 * np.pi
    trajectory = FourierExcitationTrajectory(q0, a, b, omega)

    q, qd, qdd = trajectory.sample(time)
    expected_q, expected_qd, expected_qdd = fourier_series(time, q0, a, b, omega)

    np.testing.assert_allclose(q, expected_q)
    np.testing.assert_allclose(qd, expected_qd)
    np.testing.assert_allclose(qdd, expected_qdd)


def test_boundary_polynomial_samples_position_velocity_and_acceleration():
    coeffs = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    time = np.array([0.0, 0.5])
    polynomial = FifthOrderBoundaryPolynomial(coeffs)

    q, qd, qdd = polynomial.sample(time)

    expected_q = coeffs @ np.vstack([time**i for i in range(6)])
    expected_qd = (
        coeffs[:, 1:2]
        + 2.0 * coeffs[:, 2:3] * time
        + 3.0 * coeffs[:, 3:4] * time**2
        + 4.0 * coeffs[:, 4:5] * time**3
        + 5.0 * coeffs[:, 5:6] * time**4
    )
    expected_qdd = (
        2.0 * coeffs[:, 2:3]
        + 6.0 * coeffs[:, 3:4] * time
        + 12.0 * coeffs[:, 4:5] * time**2
        + 20.0 * coeffs[:, 5:6] * time**3
    )
    np.testing.assert_allclose(q, expected_q)
    np.testing.assert_allclose(qd, expected_qd)
    np.testing.assert_allclose(qdd, expected_qdd)


def test_mixed_trajectory_starts_and_ends_at_requested_position():
    time = np.linspace(0.0, 2.0, 50)
    a = np.array([[0.2, -0.1]])
    b = np.array([[0.1, 0.05]])
    q0 = np.array([0.5])
    coeffs = fifth_order_boundary_coefficients(2.0, a, b, np.pi, q0)

    q, qd, qdd = mixed_trajectory(time, coeffs, a, b, np.pi)

    np.testing.assert_allclose(q[:, 0], q0, atol=1e-12)
    np.testing.assert_allclose(q[:, -1], q0, atol=1e-12)
    np.testing.assert_allclose(qd[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qd[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[:, -1], 0.0, atol=1e-12)


def test_excitation_generator_builds_uniformly_sampled_mixed_trajectory():
    q0 = np.array([0.1, -0.2])
    a = np.array([[0.2, -0.1], [0.05, 0.15]])
    b = np.array([[0.1, 0.05], [-0.1, 0.02]])
    generator = ExcitationTrajectoryGenerator(
        q0=q0,
        sine_coefficients=a,
        cosine_coefficients=b,
        fundamental_frequency=np.pi,
        duration=2.0,
    )

    trajectory = generator.build()
    time, q, qd, qdd = trajectory.sample_uniform(sample_count=21)

    assert time.shape == (21,)
    assert q.shape == (2, 21)
    assert qd.shape == (2, 21)
    assert qdd.shape == (2, 21)
    np.testing.assert_allclose(time[[0, -1]], [0.0, 2.0])
    np.testing.assert_allclose(q[:, 0], q0, atol=1e-12)
    np.testing.assert_allclose(q[:, -1], q0, atol=1e-12)


def test_fourier_excitation_trajectory_rejects_invalid_time_shape():
    trajectory = FourierExcitationTrajectory(
        np.zeros(1),
        np.ones((1, 2)),
        np.zeros((1, 2)),
        np.pi,
    )

    with pytest.raises(ValueError, match="time must be 1D"):
        trajectory.sample(np.zeros((2, 2)))


def test_mixed_excitation_trajectory_rejects_uniform_sampling_without_duration():
    fourier = FourierExcitationTrajectory(
        np.zeros(1),
        np.ones((1, 2)),
        np.zeros((1, 2)),
        np.pi,
    )
    boundary = FifthOrderBoundaryPolynomial(np.zeros((1, 6)))
    trajectory = MixedExcitationTrajectory(fourier, boundary)

    with pytest.raises(ValueError, match="duration is required"):
        trajectory.sample_uniform(10)


def test_mixed_excitation_trajectory_enforces_rest_boundaries():
    duration = 2.0
    time = np.linspace(0.0, duration, 50)
    q0 = np.array([0.5])
    a = np.array([[0.2, -0.1]])
    b = np.array([[0.1, 0.05]])
    trajectory = MixedExcitationTrajectory.from_fourier_boundary(
        duration,
        q0,
        a,
        b,
        np.pi,
    )

    q, qd, qdd = trajectory.sample(time)

    np.testing.assert_allclose(q[:, 0], q0, atol=1e-12)
    np.testing.assert_allclose(q[:, -1], q0, atol=1e-12)
    np.testing.assert_allclose(qd[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qd[:, -1], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qdd[:, -1], 0.0, atol=1e-12)
