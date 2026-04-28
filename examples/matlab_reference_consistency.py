"""Compare Python results against the reference MATLAB UR10e workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from robot_calibration.data.loaders import load_ur_csv
from robot_calibration.dynamics import (
    CurrentDrivenBaseDynamicsIdentifier,
    PinocchioDynamicsModel,
)
from robot_calibration.filtering import preprocess_reference_ur10e_dataset
from robot_calibration.trajectory import (
    ExcitationConditionNumberProblem,
    ExcitationTrajectoryGenerator,
    mixed_trajectory,
)

DEFAULT_REFERENCE_ROOT = Path("/home/april/robotics/dynamic_calibration")
DEFAULT_NOTEBOOK = DEFAULT_REFERENCE_ROOT / "robot_dynamics.ipynb"
DEFAULT_URDF = DEFAULT_REFERENCE_ROOT / "URe/urdf/ur10e.urdf"
DEFAULT_DATASET = (
    DEFAULT_REFERENCE_ROOT
    / "dataset_ur10e/identification_data/ur-20_02_10-30sec_12harm.csv"
)
DEFAULT_TRAJECTORY_MAT = (
    DEFAULT_REFERENCE_ROOT
    / "trajectory_optmzn/optimal_trjctrs/ptrnSrch_N12T30QR.mat"
)
DEFAULT_OUTPUT = Path("outputs/matlab_reference_consistency.json")
DEFAULT_DRIVE_GAINS = np.array([14.87, 13.26, 11.13, 10.62, 11.03, 11.47])
DEFAULT_START_INDEX = 635
DEFAULT_STOP_INDEX = 3510
DEFAULT_BASE_PARAMETER_SAMPLE_COUNT = 50
DEFAULT_BASE_PARAMETER_SEED = 0
_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|inf", re.IGNORECASE)


def _extract_first_numbers(
    text: str,
    marker: str,
    count: int,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    start = text.index(marker) + len(marker)
    values = []
    for token in _NUMBER_PATTERN.findall(text[start:]):
        values.append(np.inf if token.lower() == "inf" else float(token))
        if len(values) == count:
            break
    array = np.asarray(values, dtype=float)
    if array.size != count:
        raise ValueError(f"expected {count} numbers after marker {marker!r}")
    return array.reshape(shape)


def _difference_metrics(
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
    }


def load_reference_notebook_results(path: str | Path) -> dict[str, NDArray[np.float64]]:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    base_text = "".join(notebook["cells"][11]["outputs"][0]["text"])
    solution_text = "".join(notebook["cells"][21]["outputs"][0]["text"])
    return {
        "pi_base_urdf": _extract_first_numbers(
            base_text,
            "最小动力学参数集: (40, 1) ",
            40,
            (40, 1),
        ),
        "est_std": _extract_first_numbers(solution_text, "est_std=array([", 58, (58,)),
        "est_rel_std": _extract_first_numbers(
            solution_text,
            "est_rel_std=array([",
            58,
            (58,),
        ),
        "pi_base": _extract_first_numbers(solution_text, "pi_base=array([", 40, (40, 1)),
        "pi_full": _extract_first_numbers(solution_text, "pi_full=array([", 66, (66, 1)),
        "pi_frictions": _extract_first_numbers(
            solution_text,
            "pi_frictions=array([",
            18,
            (18, 1),
        ),
    }


def load_reference_trajectory(path: str | Path) -> dict[str, NDArray[np.float64] | float | int]:
    matlab = loadmat(path, struct_as_record=False, squeeze_me=False)
    traj_par = matlab["traj_par"][0, 0]
    return {
        "a": np.asarray(matlab["a"], dtype=float),
        "b": np.asarray(matlab["b"], dtype=float),
        "c_pol": np.asarray(matlab["c_pol"], dtype=float),
        "T": float(traj_par.T[0, 0]),
        "wf": float(traj_par.wf[0, 0]),
        "t": np.asarray(traj_par.t, dtype=float).reshape(-1),
        "N": int(traj_par.N[0, 0]),
        "q0": np.asarray(traj_par.q0, dtype=float).reshape(-1),
        "q_min": np.asarray(traj_par.q_min, dtype=float).reshape(-1),
        "q_max": np.asarray(traj_par.q_max, dtype=float).reshape(-1),
        "qd_max": np.asarray(traj_par.qd_max, dtype=float).reshape(-1),
        "q2d_max": np.asarray(traj_par.q2d_max, dtype=float).reshape(-1),
    }


def build_report(
    *,
    notebook_path: str | Path = DEFAULT_NOTEBOOK,
    urdf_path: str | Path = DEFAULT_URDF,
    dataset_path: str | Path = DEFAULT_DATASET,
    trajectory_mat_path: str | Path = DEFAULT_TRAJECTORY_MAT,
    start_index: int = DEFAULT_START_INDEX,
    stop_index: int = DEFAULT_STOP_INDEX,
    drive_gains: NDArray[np.float64] = DEFAULT_DRIVE_GAINS,
) -> dict[str, object]:
    notebook_reference = load_reference_notebook_results(notebook_path)
    trajectory_reference = load_reference_trajectory(trajectory_mat_path)

    model = PinocchioDynamicsModel.from_urdf(
        urdf_path,
        include_motor_dynamics=True,
        model_only=True,
    )
    base_parameters = model.extract_base_parameters(
        DEFAULT_BASE_PARAMETER_SAMPLE_COUNT,
        rng=np.random.default_rng(DEFAULT_BASE_PARAMETER_SEED),
    )

    dataset = load_ur_csv(
        dataset_path,
        start_index=start_index,
        stop_index=stop_index,
    )
    processed = preprocess_reference_ur10e_dataset(dataset)
    identifier = CurrentDrivenBaseDynamicsIdentifier(
        model,
        base_parameters,
        drive_gains=drive_gains,
    )
    identification = identifier.identify(processed, method="sdp")

    generator = ExcitationTrajectoryGenerator(
        q0=trajectory_reference["q0"],
        sine_coefficients=trajectory_reference["a"],
        cosine_coefficients=trajectory_reference["b"],
        fundamental_frequency=float(trajectory_reference["wf"]),
        duration=float(trajectory_reference["T"]),
    )
    trajectory = generator.build()
    recomputed_c_pol = trajectory.boundary_polynomial.coefficients
    matlab_q, matlab_qd, matlab_qdd = mixed_trajectory(
        trajectory_reference["t"],
        trajectory_reference["c_pol"],
        trajectory_reference["a"],
        trajectory_reference["b"],
        float(trajectory_reference["wf"]),
    )
    python_q, python_qd, python_qdd = trajectory.sample(trajectory_reference["t"])

    matlab_vector = np.concatenate(
        [
            trajectory_reference["a"].reshape(-1),
            trajectory_reference["b"].reshape(-1),
        ]
    )
    observation_problem = ExcitationConditionNumberProblem(
        q0=trajectory_reference["q0"],
        n_harmonics=int(trajectory_reference["N"]),
        fundamental_frequency=float(trajectory_reference["wf"]),
        duration=float(trajectory_reference["T"]),
        sample_count=int(trajectory_reference["t"].size),
        coefficient_bounds=None,
        regressor=model.regressor,
        base_parameter_mapping=base_parameters.mapping,
        friction_regressor=model.friction_regressor,
        joint_position_limits=np.column_stack(
            [trajectory_reference["q_min"], trajectory_reference["q_max"]]
        ),
        joint_velocity_limits=trajectory_reference["qd_max"],
        joint_acceleration_limits=trajectory_reference["q2d_max"],
    )
    observation_metrics = observation_problem.evaluate_observation_metrics(matlab_vector)

    return {
        "reference_paths": {
            "notebook": str(Path(notebook_path)),
            "urdf": str(Path(urdf_path)),
            "dataset": str(Path(dataset_path)),
            "trajectory_mat": str(Path(trajectory_mat_path)),
        },
        "reference_identification_setup": {
            "start_index": start_index,
            "stop_index": stop_index,
            "drive_gains": np.asarray(drive_gains, dtype=float).tolist(),
            "base_parameter_sample_count": DEFAULT_BASE_PARAMETER_SAMPLE_COUNT,
            "base_parameter_seed": DEFAULT_BASE_PARAMETER_SEED,
        },
        "identification_comparison": {
            "base_parameter_count": int(base_parameters.num_base_params),
            "full_parameter_count": int(base_parameters.full_parameters.size),
            "pi_base_urdf": _difference_metrics(
                base_parameters.base_parameters.reshape(-1, 1),
                notebook_reference["pi_base_urdf"],
            ),
            "pi_base": _difference_metrics(
                identification.base_parameters.reshape(-1, 1),
                notebook_reference["pi_base"],
            ),
            "pi_full": _difference_metrics(
                identification.full_parameters.reshape(-1, 1),
                notebook_reference["pi_full"],
            ),
            "pi_frictions": _difference_metrics(
                identification.friction_parameters.reshape(-1, 1),
                notebook_reference["pi_frictions"],
            ),
            "est_std": _difference_metrics(
                identification.estimated_standard_deviation.reshape(-1),
                notebook_reference["est_std"].reshape(-1),
            ),
            "est_rel_std": _difference_metrics(
                identification.estimated_relative_standard_deviation.reshape(-1),
                notebook_reference["est_rel_std"].reshape(-1),
            ),
        },
        "trajectory_comparison": {
            "a_shape": list(trajectory_reference["a"].shape),
            "b_shape": list(trajectory_reference["b"].shape),
            "c_pol_shape": list(trajectory_reference["c_pol"].shape),
            "boundary_polynomial": _difference_metrics(
                recomputed_c_pol,
                trajectory_reference["c_pol"],
            ),
            "sampled_q": _difference_metrics(python_q, matlab_q),
            "sampled_qd": _difference_metrics(python_qd, matlab_qd),
            "sampled_qdd": _difference_metrics(python_qdd, matlab_qdd),
            "matlab_coefficients_in_python_observation": {
                "condition_number": float(observation_metrics.condition_number),
                "observation_rank": int(observation_metrics.observation_rank),
                "target_rank": int(observation_metrics.target_rank),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK))
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--trajectory-mat", default=str(DEFAULT_TRAJECTORY_MAT))
    parser.add_argument("--start-index", type=int, default=DEFAULT_START_INDEX)
    parser.add_argument("--stop-index", type=int, default=DEFAULT_STOP_INDEX)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    report = build_report(
        notebook_path=args.notebook,
        urdf_path=args.urdf,
        dataset_path=args.dataset,
        trajectory_mat_path=args.trajectory_mat,
        start_index=args.start_index,
        stop_index=args.stop_index,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    identification = report["identification_comparison"]
    trajectory = report["trajectory_comparison"]
    print(
        "pi_full max_abs diff:",
        f"{identification['pi_full']['max_abs']:.6g}",
    )
    print(
        "pi_frictions max_abs diff:",
        f"{identification['pi_frictions']['max_abs']:.6g}",
    )
    print(
        "pi_base max_abs diff:",
        f"{identification['pi_base']['max_abs']:.6g}",
    )
    print(
        "c_pol max_abs diff:",
        f"{trajectory['boundary_polynomial']['max_abs']:.6g}",
    )
    print(f"wrote report to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
