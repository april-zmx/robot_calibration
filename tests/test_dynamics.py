import numpy as np

from robot_calibration.data import CalibrationDataset
from robot_calibration.dynamics import (
    assemble_observation_matrix,
    extract_base_parameters,
    estimate_current_torque,
    identify_current_driven_dynamics,
    identify_ols,
    linear_friction_regressor,
    predict_torque,
)


def test_linear_friction_regressor_has_joint_blocks():
    regressor = linear_friction_regressor(np.array([2.0, -3.0]))

    np.testing.assert_allclose(
        regressor,
        [[2.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -3.0, -1.0, 1.0]],
    )


def test_assemble_observation_matrix_stacks_regressors():
    dataset = CalibrationDataset(
        time=[0.0, 0.1],
        positions=np.array([[1.0], [2.0]]),
        velocities=np.array([[0.5], [0.6]]),
        accelerations=np.array([[0.1], [0.2]]),
        torques=np.array([[3.0], [4.0]]),
    )

    def regressor(q, qd, qdd):
        return np.array([[q[0], qd[0], qdd[0]]])

    observation = assemble_observation_matrix(dataset, regressor)

    np.testing.assert_allclose(observation.matrix, [[1.0, 0.5, 0.1], [2.0, 0.6, 0.2]])
    np.testing.assert_allclose(observation.torque, [3.0, 4.0])


def test_ols_recovers_known_parameters():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    expected = np.array([2.0, -3.0])
    torque = matrix @ expected

    result = identify_ols(matrix, torque)

    np.testing.assert_allclose(result.parameters, expected, atol=1e-12)
    assert result.rank == 2


def test_base_parameter_extraction_reproduces_column_space():
    matrix = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.0], [2.0, 1.0, 3.0]])

    base = extract_base_parameters(matrix, tolerance=1e-10)
    reconstructed_dependent = (
        matrix @ base.permutation[:, : base.rank] @ base.beta
    )
    actual_dependent = matrix @ base.permutation[:, base.rank :]

    np.testing.assert_allclose(reconstructed_dependent, actual_dependent, atol=1e-10)


def test_predict_torque_multiplies_observation_matrix():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    parameters = np.array([0.5, 2.0])

    np.testing.assert_allclose(predict_torque(matrix, parameters), [4.5, 9.5])


def test_estimate_current_torque_applies_joint_drive_gains():
    currents = np.array([[1.0, 2.0], [3.0, 4.0]])
    gains = np.array([10.0, 20.0])

    np.testing.assert_allclose(
        estimate_current_torque(currents, gains),
        [[10.0, 40.0], [30.0, 80.0]],
    )


def test_identify_current_driven_dynamics_recovers_base_and_friction_parameters():
    q = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.4, 0.3],
            [0.6, -0.2],
            [0.8, 0.5],
            [1.0, -0.4],
        ]
    )
    qd = np.array(
        [
            [0.5, -0.7],
            [-0.8, 0.4],
            [1.2, -1.1],
            [-1.5, 0.9],
            [1.7, -1.3],
            [-2.0, 1.5],
        ]
    )
    qdd = np.array(
        [
            [1.0, 0.2],
            [1.5, -0.3],
            [2.0, 0.5],
            [2.5, -0.7],
            [3.0, 0.9],
            [3.5, -1.1],
        ]
    )

    dynamic_parameters = np.array([1.5, -2.0])
    friction_parameters = np.array([0.2, 0.4, 0.1, 0.3, 0.5, -0.2])
    drive_gains = np.array([10.0, 20.0])

    def regressor(_q, _qd, sample_qdd):
        return np.array([[sample_qdd[0], 0.0], [0.0, sample_qdd[1]]])

    torque_rows = []
    for sample_q, sample_qd, sample_qdd in zip(q, qd, qdd):
        y = regressor(sample_q, sample_qd, sample_qdd)
        y_friction = linear_friction_regressor(sample_qd)
        torque_rows.append(y @ dynamic_parameters + y_friction @ friction_parameters)
    torques = np.asarray(torque_rows)
    dataset = CalibrationDataset(
        time=np.arange(q.shape[0], dtype=float),
        positions=q,
        velocities=qd,
        accelerations=qdd,
        currents=torques / drive_gains,
    )

    result = identify_current_driven_dynamics(
        dataset,
        regressor,
        drive_gains=drive_gains,
        include_friction=True,
    )

    np.testing.assert_allclose(result.dynamic_parameters, dynamic_parameters, atol=1e-10)
    np.testing.assert_allclose(result.friction_parameters, friction_parameters, atol=1e-10)
    np.testing.assert_allclose(result.predicted_torque, torques.reshape(-1), atol=1e-10)


def test_identify_current_driven_dynamics_supports_bounded_method():
    dataset = CalibrationDataset(
        time=np.array([0.0, 1.0]),
        positions=np.zeros((2, 1)),
        velocities=np.zeros((2, 1)),
        accelerations=np.array([[1.0], [2.0]]),
        currents=np.array([[-1.0], [-2.0]]),
    )

    def regressor(_q, _qd, qdd):
        return np.array([[qdd[0]]])

    result = identify_current_driven_dynamics(
        dataset,
        regressor,
        drive_gains=np.array([1.0]),
        include_friction=False,
        method="bounded",
        nonnegative_indices=[0],
    )

    np.testing.assert_allclose(result.dynamic_parameters, [0.0], atol=1e-10)
    assert result.identification.success


def test_identify_current_driven_dynamics_supports_cvxpy_method():
    pytest = __import__("pytest")
    pytest.importorskip("cvxpy")
    dataset = CalibrationDataset(
        time=np.array([0.0, 1.0]),
        positions=np.zeros((2, 1)),
        velocities=np.zeros((2, 1)),
        accelerations=np.array([[1.0], [2.0]]),
        currents=np.array([[-1.0], [-2.0]]),
    )

    def regressor(_q, _qd, qdd):
        return np.array([[qdd[0]]])

    result = identify_current_driven_dynamics(
        dataset,
        regressor,
        drive_gains=np.array([1.0]),
        include_friction=False,
        method="cvxpy",
        nonnegative_indices=[0],
    )

    np.testing.assert_allclose(result.dynamic_parameters, [0.0], atol=1e-6)
    assert result.identification.success


def test_identify_current_driven_dynamics_supports_physical_parameter_blocks():
    pytest = __import__("pytest")
    pytest.importorskip("cvxpy")
    target = np.array([[-1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.2, -0.5]])
    dataset = CalibrationDataset(
        time=np.array([0.0]),
        positions=np.zeros((1, 11)),
        velocities=np.zeros((1, 11)),
        accelerations=np.zeros((1, 11)),
        currents=target,
    )

    def regressor(_q, _qd, _qdd):
        return np.eye(11)

    result = identify_current_driven_dynamics(
        dataset,
        regressor,
        drive_gains=np.ones(11),
        include_friction=False,
        method="cvxpy",
        physical_consistency=True,
        physical_block_width=11,
        physical_has_motor_inertia=True,
    )

    assert result.identification.success
    assert result.dynamic_parameters[0] >= -1e-7
    assert result.dynamic_parameters[10] >= -1e-7
