"""Command-line entry points for robot calibration."""

from __future__ import annotations

import argparse

import numpy as np

from robot_calibration.data import CalibrationDataset
from robot_calibration.data.loaders import load_ur_csv
from robot_calibration.dynamics import (
    build_pinocchio_regressor_from_urdf,
    identify_current_driven_dynamics,
    identify_ols,
    linear_friction_regressor,
    predict_torque,
)
from robot_calibration.filtering import preprocess_dataset
from robot_calibration.kinematics import DHParameter, calibrate_dh_offsets, forward_kinematics
from robot_calibration.metrics import rmse


def _parse_float_list(value: str) -> np.ndarray:
    try:
        parsed = np.array([float(item) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if parsed.size == 0:
        raise argparse.ArgumentTypeError("expected at least one float")
    return parsed


def _parse_int_list(value: str) -> list[int]:
    if value.strip() == "":
        return []
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def _format_vector(values: np.ndarray, *, zero_tolerance: float = 1e-12) -> str:
    clean = np.asarray(values, dtype=float).copy()
    clean[np.abs(clean) < zero_tolerance] = 0.0
    return " ".join(f"{value:.6g}" for value in clean)


def run_kinematics_demo() -> int:
    true_params = [
        DHParameter(a=1.1, alpha=0.0, d=0.0, theta=0.0),
        DHParameter(a=0.7, alpha=0.0, d=0.0, theta=0.0),
    ]
    initial_params = [
        DHParameter(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        DHParameter(a=0.6, alpha=0.0, d=0.0, theta=0.0),
    ]
    joint_positions = np.array(
        [
            [0.0, 0.0],
            [np.pi / 4.0, -np.pi / 6.0],
            [np.pi / 2.0, -np.pi / 4.0],
            [-np.pi / 3.0, np.pi / 5.0],
        ]
    )
    observations = np.array(
        [forward_kinematics(true_params, q)[:3, 3] for q in joint_positions]
    )
    result = calibrate_dh_offsets(
        initial_params,
        joint_positions,
        observations,
        estimate=("a",),
    )
    print(f"kinematics residual rmse: {rmse(result.residuals, 0.0):.6g}")
    print("estimated link lengths:", _format_vector(np.array([row.a for row in result.parameters])))
    return 0 if result.success else 1


def run_dynamics_demo() -> int:
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=float,
    )
    true_parameters = np.array([2.0, -3.0])
    torque = matrix @ true_parameters
    result = identify_ols(matrix, torque)
    predicted = predict_torque(matrix, result.parameters)
    print(f"dynamics residual rmse: {rmse(torque, predicted):.6g}")
    print("estimated parameters:", _format_vector(result.parameters))
    return 0


def run_current_dynamics_demo() -> int:
    q = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.4, 0.3],
            [0.6, -0.2],
            [0.8, 0.5],
            [1.0, -0.4],
        ],
        dtype=float,
    )
    qd = np.array(
        [
            [0.5, -0.7],
            [-0.8, 0.4],
            [1.2, -1.1],
            [-1.5, 0.9],
            [1.7, -1.3],
            [-2.0, 1.5],
        ],
        dtype=float,
    )
    qdd = np.array(
        [
            [1.0, 0.2],
            [1.5, -0.3],
            [2.0, 0.5],
            [2.5, -0.7],
            [3.0, 0.9],
            [3.5, -1.1],
        ],
        dtype=float,
    )
    dynamic_parameters = np.array([1.5, -2.0])
    friction_parameters = np.array([0.2, 0.4, 0.1, 0.3, 0.5, -0.2])
    drive_gains = np.array([10.0, 20.0])

    def regressor(_q, _qd, sample_qdd):
        return np.array([[sample_qdd[0], 0.0], [0.0, sample_qdd[1]]])

    torque = []
    for sample_q, sample_qd, sample_qdd in zip(q, qd, qdd):
        y = regressor(sample_q, sample_qd, sample_qdd)
        y_friction = linear_friction_regressor(sample_qd)
        torque.append(y @ dynamic_parameters + y_friction @ friction_parameters)
    torque = np.asarray(torque)
    dataset = CalibrationDataset(
        time=np.arange(q.shape[0], dtype=float),
        positions=q,
        velocities=qd,
        accelerations=qdd,
        currents=torque / drive_gains,
    )

    result = identify_current_driven_dynamics(
        dataset,
        regressor,
        drive_gains=drive_gains,
        include_friction=True,
    )
    print(f"current-driven dynamics residual rmse: {rmse(result.measured_torque, result.predicted_torque):.6g}")
    print(
        "estimated dynamic parameters:",
        _format_vector(result.dynamic_parameters),
    )
    print(
        "estimated friction parameters:",
        _format_vector(result.friction_parameters),
    )
    return 0


def acceleration_diagonal_regressor(_q, _qd, qdd):
    """Simple joint-wise inertia regressor useful for CLI smoke tests."""

    return np.diag(np.asarray(qdd, dtype=float))


def run_identify_ur_current(args: argparse.Namespace) -> int:
    if args.regressor == "acceleration-diagonal":
        regressor = acceleration_diagonal_regressor
    elif args.regressor == "pinocchio":
        if args.urdf is None:
            raise ValueError("--urdf is required when --regressor pinocchio")
        regressor = build_pinocchio_regressor_from_urdf(
            args.urdf,
            include_motor_dynamics=args.include_motor_dynamics,
            link_parameter_count=args.link_parameter_count,
        )
    else:
        raise ValueError(f"unsupported regressor: {args.regressor}")

    dataset = load_ur_csv(
        args.csv,
        start_index=args.start_index,
        stop_index=args.stop_index,
    )
    if not args.no_filter:
        dataset = preprocess_dataset(
            dataset,
            velocity_cutoff_hz=args.velocity_cutoff,
            current_cutoff_hz=args.current_cutoff,
            acceleration_cutoff_hz=args.acceleration_cutoff,
        )
    elif dataset.accelerations is None:
        # UR logs do not include accelerations. In no-filter mode this command
        # expects callers to use a regressor that can work with finite differences.
        from robot_calibration.filtering import central_difference

        dataset = CalibrationDataset(
            time=dataset.time,
            positions=dataset.positions,
            velocities=dataset.velocities,
            accelerations=central_difference(dataset.time, dataset.velocities),
            currents=dataset.currents,
            desired_currents=dataset.desired_currents,
            desired_torques=dataset.desired_torques,
            metadata=dataset.metadata,
        )

    result = identify_current_driven_dynamics(
        dataset,
        regressor,
        drive_gains=args.drive_gains,
        include_friction=not args.no_friction,
        method=args.method,
        nonnegative_indices=args.nonnegative_indices,
        physical_consistency=args.physical_consistency,
        physical_block_width=args.link_parameter_count
        + (1 if args.include_motor_dynamics else 0),
        physical_has_motor_inertia=args.include_motor_dynamics,
    )
    print(
        f"ur current identification residual rmse: "
        f"{rmse(result.measured_torque, result.predicted_torque):.6g}"
    )
    print(
        "estimated dynamic parameters:",
        _format_vector(result.dynamic_parameters),
    )
    if result.friction_parameters is not None:
        print(
            "estimated friction parameters:",
            _format_vector(result.friction_parameters),
        )
    print(f"observation matrix shape: {result.observation_matrix.shape}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-calib",
        description="Robot kinematics and dynamics calibration utilities.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "kinematics-demo",
        help="Run a synthetic 2R DH calibration demo.",
    )
    subparsers.add_parser(
        "dynamics-demo",
        help="Run a synthetic OLS dynamics identification demo.",
    )
    subparsers.add_parser(
        "current-dynamics-demo",
        help="Run a synthetic current-driven dynamics identification demo.",
    )
    identify_parser = subparsers.add_parser(
        "identify-ur-current",
        help="Identify dynamics from a UR-style current CSV.",
    )
    identify_parser.add_argument("csv", help="Path to a UR-style no-header CSV file.")
    identify_parser.add_argument(
        "--drive-gains",
        type=_parse_float_list,
        required=True,
        help="Comma-separated current-to-torque gains, one per joint.",
    )
    identify_parser.add_argument("--start-index", type=int, default=None)
    identify_parser.add_argument("--stop-index", type=int, default=None)
    identify_parser.add_argument(
        "--regressor",
        choices=["acceleration-diagonal", "pinocchio"],
        default="acceleration-diagonal",
        help="Regressor backend to use.",
    )
    identify_parser.add_argument(
        "--urdf",
        default=None,
        help="URDF path for the Pinocchio regressor backend.",
    )
    identify_parser.add_argument(
        "--include-motor-dynamics",
        action="store_true",
        help="Append one reflected motor inertia column per joint.",
    )
    identify_parser.add_argument(
        "--link-parameter-count",
        type=int,
        default=10,
        help="Number of rigid-body dynamic parameters per joint link.",
    )
    identify_parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip low-pass filtering and estimate acceleration by central difference.",
    )
    identify_parser.add_argument(
        "--no-friction",
        action="store_true",
        help="Do not append the linear friction regressor.",
    )
    identify_parser.add_argument(
        "--method",
        choices=["ols", "bounded", "cvxpy"],
        default="ols",
        help="Identification method.",
    )
    identify_parser.add_argument(
        "--nonnegative-indices",
        type=_parse_int_list,
        default=None,
        help="Comma-separated parameter indices constrained to be nonnegative.",
    )
    identify_parser.add_argument(
        "--physical-consistency",
        action="store_true",
        help="Add Pinocchio-style pseudo-inertia PSD constraints for CVXPY.",
    )
    identify_parser.add_argument("--velocity-cutoff", type=float, default=2.0)
    identify_parser.add_argument("--current-cutoff", type=float, default=2.0)
    identify_parser.add_argument("--acceleration-cutoff", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from robot_calibration import __version__

        print(__version__)
        return 0
    if args.command == "kinematics-demo":
        return run_kinematics_demo()
    if args.command == "dynamics-demo":
        return run_dynamics_demo()
    if args.command == "current-dynamics-demo":
        return run_current_dynamics_demo()
    if args.command == "identify-ur-current":
        try:
            return run_identify_ur_current(args)
        except (ModuleNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
