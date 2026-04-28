import numpy as np

from robot_calibration.dynamics import identify_bounded_ols


def test_bounded_ols_matches_unconstrained_solution_when_inside_bounds():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    torque = matrix @ np.array([2.0, 3.0])

    result = identify_bounded_ols(
        matrix,
        torque,
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([10.0, 10.0]),
    )

    np.testing.assert_allclose(result.parameters, [2.0, 3.0], atol=1e-10)
    assert result.success


def test_bounded_ols_enforces_nonnegative_parameter_bounds():
    matrix = np.eye(2)
    torque = np.array([-2.0, 3.0])

    result = identify_bounded_ols(
        matrix,
        torque,
        lower_bounds=np.array([0.0, 0.0]),
    )

    np.testing.assert_allclose(result.parameters, [0.0, 3.0], atol=1e-10)
    assert result.active_lower_bounds.tolist() == [0]


def test_bounded_ols_supports_index_based_nonnegative_constraints():
    matrix = np.eye(3)
    torque = np.array([-1.0, -2.0, 3.0])

    result = identify_bounded_ols(
        matrix,
        torque,
        nonnegative_indices=[0, 2],
    )

    np.testing.assert_allclose(result.parameters, [0.0, -2.0, 3.0], atol=1e-10)
