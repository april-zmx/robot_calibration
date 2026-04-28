import subprocess
import sys

import numpy as np

from robot_calibration.filtering import central_difference


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "robot_calibration.cli.main", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "kinematics-demo" in result.stdout
    assert "dynamics-demo" in result.stdout


def test_cli_kinematics_demo():
    result = run_cli("kinematics-demo")

    assert result.returncode == 0
    assert "kinematics residual rmse" in result.stdout


def test_cli_dynamics_demo():
    result = run_cli("dynamics-demo")

    assert result.returncode == 0
    assert "dynamics residual rmse" in result.stdout


def test_cli_current_dynamics_demo():
    result = run_cli("current-dynamics-demo")

    assert result.returncode == 0
    assert "current-driven dynamics residual rmse" in result.stdout


def test_cli_identify_ur_current_with_acceleration_diagonal_regressor(tmp_path):
    csv_path = tmp_path / "ur.csv"
    drive_gains = np.array([10.0, 20.0])
    q = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.4, 0.3],
            [0.6, -0.2],
            [0.8, 0.5],
            [1.0, -0.4],
            [1.2, 0.7],
            [1.4, -0.6],
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
            [2.2, -1.7],
            [-2.4, 1.9],
        ]
    )
    qd_full = np.column_stack([qd, np.zeros((q.shape[0], 4))])
    qdd_full = central_difference(np.arange(q.shape[0], dtype=float), qd_full)
    dynamic_parameters = np.array([1.5, -2.0, 0.0, 0.0, 0.0, 0.0])
    currents = (qdd_full * dynamic_parameters) / np.array([10.0, 20.0, 1.0, 1.0, 1.0, 1.0])
    desired = np.zeros_like(currents)
    rows = np.column_stack(
        [
            np.arange(q.shape[0], dtype=float),
            np.column_stack([q, np.zeros((q.shape[0], 4))]),
            qd_full,
            currents,
            desired,
            desired,
        ]
    )
    np.savetxt(csv_path, rows, delimiter=",")

    result = run_cli(
        "identify-ur-current",
        str(csv_path),
        "--drive-gains",
        "10,20,1,1,1,1",
        "--regressor",
        "acceleration-diagonal",
        "--no-filter",
        "--no-friction",
    )

    assert result.returncode == 0
    assert "ur current identification residual rmse" in result.stdout
    assert "estimated dynamic parameters: 1.5 -2 0 0 0 0" in result.stdout


def test_cli_identify_ur_current_pinocchio_requires_urdf(tmp_path):
    csv_path = tmp_path / "ur.csv"
    np.savetxt(csv_path, np.zeros((2, 31)), delimiter=",")

    result = run_cli(
        "identify-ur-current",
        str(csv_path),
        "--drive-gains",
        "1,1,1,1,1,1",
        "--regressor",
        "pinocchio",
    )

    assert result.returncode != 0
    assert "--urdf is required" in result.stderr


def test_cli_identify_ur_current_supports_bounded_method(tmp_path):
    csv_path = tmp_path / "ur.csv"
    time = np.array([0.0, 1.0, 2.0])
    q = np.zeros((3, 6))
    qd = np.array([[0.0], [1.0], [2.0]])
    qd = np.column_stack([qd, np.zeros((3, 5))])
    currents = np.array([[-1.0], [-1.0], [-1.0]])
    currents = np.column_stack([currents, np.zeros((3, 5))])
    desired = np.zeros_like(currents)
    rows = np.column_stack([time, q, qd, currents, desired, desired])
    np.savetxt(csv_path, rows, delimiter=",")

    result = run_cli(
        "identify-ur-current",
        str(csv_path),
        "--drive-gains",
        "1,1,1,1,1,1",
        "--regressor",
        "acceleration-diagonal",
        "--no-filter",
        "--no-friction",
        "--method",
        "bounded",
        "--nonnegative-indices",
        "0",
    )

    assert result.returncode == 0
    assert "estimated dynamic parameters: 0" in result.stdout


def test_cli_identify_ur_current_help_lists_cvxpy_method():
    result = run_cli("identify-ur-current", "--help")

    assert result.returncode == 0
    assert "{ols,bounded,cvxpy}" in result.stdout
    assert "--physical-consistency" in result.stdout
