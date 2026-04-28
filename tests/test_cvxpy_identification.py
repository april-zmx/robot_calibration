import numpy as np
import pytest

from robot_calibration.dynamics import (
    PhysicalParameterBlock,
    identify_cvxpy_constrained_ols,
)


def test_cvxpy_constrained_ols_imports_without_required_dependency():
    assert callable(identify_cvxpy_constrained_ols)


def test_cvxpy_constrained_ols_enforces_nonnegative_indices():
    pytest.importorskip("cvxpy")
    matrix = np.eye(2)
    torque = np.array([-2.0, 3.0])

    result = identify_cvxpy_constrained_ols(
        matrix,
        torque,
        nonnegative_indices=[0],
    )

    np.testing.assert_allclose(result.parameters, [0.0, 3.0], atol=1e-6)
    assert result.success


def test_cvxpy_constrained_ols_enforces_mass_bounds():
    pytest.importorskip("cvxpy")
    matrix = np.eye(3)
    torque = np.array([5.0, 2.0, -1.0])

    result = identify_cvxpy_constrained_ols(
        matrix,
        torque,
        mass_indices=[0, 1],
        mass_upper_bounds=[4.0, 3.0],
    )

    np.testing.assert_allclose(result.parameters[:2], [4.0, 2.0], atol=1e-6)
    assert result.parameters[2] == pytest.approx(-1.0, abs=1e-6)


def test_cvxpy_constrained_ols_enforces_pinocchio_pseudo_inertia_block():
    pytest.importorskip("cvxpy")
    # Pinocchio-style layout:
    # [m, hx, hy, hz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz, Im]
    target = np.array(
        [
            -1.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.2,
            0.0,
            0.0,
            0.2,
            -0.5,
        ]
    )

    result = identify_cvxpy_constrained_ols(
        np.eye(11),
        target,
        physical_parameter_blocks=[PhysicalParameterBlock(start=0, has_motor_inertia=True)],
    )

    assert result.success
    assert result.parameters[0] >= -1e-7
    assert result.parameters[10] >= -1e-7
    inertia = np.array(
        [
            [result.parameters[4], result.parameters[5], result.parameters[7]],
            [result.parameters[5], result.parameters[6], result.parameters[8]],
            [result.parameters[7], result.parameters[8], result.parameters[9]],
        ]
    )
    h = result.parameters[1:4]
    pseudo = np.block(
        [
            [0.5 * np.trace(inertia) * np.eye(3) - inertia, h[:, None]],
            [h[None, :], result.parameters[0:1, None]],
        ]
    )
    assert np.linalg.eigvalsh(pseudo).min() >= -1e-6
