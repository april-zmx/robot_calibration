import numpy as np

from robot_calibration.metrics import condition_number, relative_residual_error, rmse


def test_relative_residual_error_per_joint():
    measured = np.array([[2.0, 2.0], [4.0, 4.0]])
    predicted = np.array([[1.0, 2.0], [2.0, 4.0]])

    error = relative_residual_error(measured, predicted, axis=0)

    np.testing.assert_allclose(error, [50.0, 0.0])


def test_rmse_and_condition_number():
    np.testing.assert_allclose(rmse([1.0, 3.0], [1.0, 1.0]), np.sqrt(2.0))
    assert condition_number(np.eye(3)) == 1.0
