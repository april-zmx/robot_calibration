import types

import numpy as np
import pytest

from robot_calibration.dynamics.pinocchio_adapter import (
    PinocchioRegressor,
    build_pinocchio_model_regressor_from_urdf,
    build_pinocchio_regressor_from_urdf,
)


class FakeModel:
    nq = 2


class FakeRobot:
    def __init__(self):
        self.model = FakeModel()
        self.data = object()


def test_pinocchio_regressor_delegates_to_compute_joint_torque_regressor():
    calls = []

    def compute_joint_torque_regressor(model, data, q, qd, qdd):
        calls.append((model, data, q.copy(), qd.copy(), qdd.copy()))
        return np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    fake_pin = types.SimpleNamespace(
        computeJointTorqueRegressor=compute_joint_torque_regressor
    )
    robot = FakeRobot()
    regressor = PinocchioRegressor(robot, pinocchio=fake_pin)

    y = regressor(np.array([0.1, 0.2]), np.array([0.3, 0.4]), np.array([0.5, 0.6]))

    np.testing.assert_allclose(y, [[1, 2, 3, 4], [5, 6, 7, 8]])
    assert calls[0][0] is robot.model
    assert calls[0][1] is robot.data
    np.testing.assert_allclose(calls[0][2], [0.1, 0.2])


def test_pinocchio_regressor_inserts_motor_inertia_columns_after_each_link_block():
    raw_regressor = np.arange(1.0, 41.0).reshape(2, 20)

    def compute_joint_torque_regressor(_model, _data, _q, _qd, _qdd):
        return raw_regressor

    fake_pin = types.SimpleNamespace(
        computeJointTorqueRegressor=compute_joint_torque_regressor
    )
    regressor = PinocchioRegressor(
        FakeRobot(),
        include_motor_dynamics=True,
        link_parameter_count=10,
        pinocchio=fake_pin,
    )

    y = regressor(np.zeros(2), np.zeros(2), np.array([0.5, -0.25]))

    assert y.shape == (2, 22)
    np.testing.assert_allclose(y[:, :10], raw_regressor[:, :10])
    np.testing.assert_allclose(y[:, 10], [0.5, 0.0])
    np.testing.assert_allclose(y[:, 11:21], raw_regressor[:, 10:20])
    np.testing.assert_allclose(y[:, 21], [0.0, -0.25])


def test_pinocchio_regressor_raises_clear_error_when_dependency_missing():
    with pytest.raises(ModuleNotFoundError, match="pinocchio"):
        PinocchioRegressor(FakeRobot(), pinocchio=None)


def test_build_pinocchio_regressor_from_urdf_uses_pinocchio_robot_wrapper(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='demo'/>", encoding="utf-8")
    calls = []

    class FakeRobotWrapper:
        def __init__(self, model, collision_model, visual_model):
            self.model = model
            self.data = object()
            self.collision_model = collision_model
            self.visual_model = visual_model

    def build_models_from_urdf(path, verbose=False):
        calls.append((path, verbose))
        return FakeModel(), object(), object()

    fake_pin = types.SimpleNamespace(buildModelsFromUrdf=build_models_from_urdf)

    regressor = build_pinocchio_regressor_from_urdf(
        urdf,
        include_motor_dynamics=True,
        pinocchio=fake_pin,
        robot_wrapper_cls=FakeRobotWrapper,
    )

    assert isinstance(regressor, PinocchioRegressor)
    assert regressor.include_motor_dynamics
    assert calls == [(str(urdf), False)]


def test_build_pinocchio_model_regressor_from_urdf_uses_model_only_loader(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='demo'/>", encoding="utf-8")
    calls = []

    class FakeModelWithData(FakeModel):
        def createData(self):
            return object()

    def build_model_from_urdf(path):
        calls.append(path)
        return FakeModelWithData()

    fake_pin = types.SimpleNamespace(buildModelFromUrdf=build_model_from_urdf)

    regressor = build_pinocchio_model_regressor_from_urdf(
        urdf,
        include_motor_dynamics=True,
        pinocchio=fake_pin,
    )

    assert isinstance(regressor, PinocchioRegressor)
    assert regressor.include_motor_dynamics
    assert calls == [str(urdf)]
