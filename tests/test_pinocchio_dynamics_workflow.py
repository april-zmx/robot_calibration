import types

import numpy as np

from robot_calibration.data import CalibrationDataset
from robot_calibration.dynamics import (
    CurrentDrivenBaseDynamicsIdentifier,
    PinocchioDynamicsModel,
)


class FakeInertia:
    def __init__(self, parameters):
        self._parameters = np.asarray(parameters, dtype=float)
        self.mass = float(self._parameters[0])

    def toDynamicParameters(self):
        return self._parameters.copy()


class FakeModel:
    def __init__(self):
        self.nq = 2
        self.inertias = [
            FakeInertia([0.0, 0.0]),
            FakeInertia([1.0, 2.0]),
            FakeInertia([3.0, 4.0]),
        ]
        self.lowerPositionLimit = np.array([-1.0, -1.0])
        self.upperPositionLimit = np.array([1.0, 1.0])
        self.velocityLimit = np.array([2.0, 2.0])

    def createData(self):
        return object()


class FakeRobot:
    def __init__(self):
        self.model = FakeModel()
        self.data = object()


def build_fake_pinocchio():
    def compute_joint_torque_regressor(_model, _data, _q, _qd, qdd):
        return np.array(
            [
                [qdd[0], qdd[0], 0.0, 0.0],
                [0.0, 0.0, qdd[1], -qdd[1]],
            ]
        )

    return types.SimpleNamespace(
        computeJointTorqueRegressor=compute_joint_torque_regressor
    )


def build_dynamics_model():
    return PinocchioDynamicsModel(
        FakeRobot(),
        include_motor_dynamics=True,
        link_parameter_count=2,
        pinocchio=build_fake_pinocchio(),
    )


def test_pinocchio_dynamics_model_returns_interleaved_dynamic_parameters():
    dynamics = build_dynamics_model()

    np.testing.assert_allclose(
        dynamics.dynamic_parameter_vector(),
        [1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
    )


def test_pinocchio_dynamics_model_extracts_base_parameters_from_random_samples():
    dynamics = build_dynamics_model()

    result = dynamics.extract_base_parameters(
        sample_count=32,
        rng=np.random.default_rng(0),
        acceleration_limit_scale=1.0,
    )

    full_parameters = dynamics.dynamic_parameter_vector()
    sorted_parameters = result.mapping.permutation.T @ full_parameters
    expected_base = (
        sorted_parameters[: result.mapping.rank]
        + result.mapping.beta @ sorted_parameters[result.mapping.rank :]
    )

    assert result.mapping.rank == 2
    np.testing.assert_allclose(result.base_parameters, expected_base)
    np.testing.assert_allclose(
        result.observation_matrix @ full_parameters,
        result.torque_samples,
    )


def test_current_driven_base_dynamics_identifier_recovers_base_and_friction_parameters():
    dynamics = build_dynamics_model()
    base_parameters = dynamics.extract_base_parameters(
        sample_count=32,
        rng=np.random.default_rng(0),
        acceleration_limit_scale=1.0,
    )
    drive_gains = np.array([10.0, 20.0])
    friction_parameters = np.array([0.2, 0.4, 0.1, 0.3, 0.5, -0.2])
    identifier = CurrentDrivenBaseDynamicsIdentifier(
        dynamics,
        base_parameters,
        drive_gains=drive_gains,
    )

    q = np.zeros((6, 2))
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

    torques = []
    for sample_q, sample_qd, sample_qdd in zip(q, qd, qdd):
        y = dynamics.regressor(sample_q, sample_qd, sample_qdd)
        y_friction = dynamics.friction_regressor(sample_qd)
        torques.append(
            y @ dynamics.dynamic_parameter_vector()
            + y_friction @ friction_parameters
        )
    torques = np.asarray(torques)
    dataset = CalibrationDataset(
        time=np.arange(q.shape[0], dtype=float),
        positions=q,
        velocities=qd,
        accelerations=qdd,
        currents=torques / drive_gains,
    )

    observation_matrix, measured_torque = identifier.build_observation_matrix(dataset)
    result = identifier.identify(dataset, method="least_square")

    np.testing.assert_allclose(result.observation_matrix, observation_matrix)
    np.testing.assert_allclose(result.measured_torque, measured_torque)
    np.testing.assert_allclose(
        result.base_parameters,
        base_parameters.base_parameters,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.friction_parameters,
        friction_parameters,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.predicted_torque,
        result.measured_torque,
        atol=1e-10,
    )
    assert result.full_parameters is None
