"""Physically constrained CVXPY identification for Piper excitation data.

Usage:
    source venv/bin/activate
    python examples/piper_dynamics_cvxpy_identification.py

Optional:
    python examples/piper_dynamics_cvxpy_identification.py \
        --data-dir piper_data/piper_excitation_20260430_170440 \
        --urdf agx_arm_urdf/piper/urdf/piper_description.urdf \
        --output-dir outputs/piper_dynamics_identification_20260430_170440_cvxpy \
        --solver CLARABEL
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np

from robot_calibration.dynamics import PinocchioDynamicsModel
from robot_calibration.dynamics.friction import linear_friction_regressor
from robot_calibration.filtering import central_difference, lowpass_zero_phase


PARAMETER_NAMES = [
    "m",
    "hx",
    "hy",
    "hz",
    "Ixx",
    "Ixy",
    "Iyy",
    "Ixz",
    "Iyz",
    "Izz",
]


def build_dataset(collection_path: Path, cutoff_hz: float, edge_trim_sec: float) -> dict[str, np.ndarray]:
    data = np.load(collection_path, allow_pickle=True)
    time = data["left_feedback_time"].astype(float)
    q = data["left_feedback_position"][:, :6].astype(float)
    qd = data["left_feedback_velocity"][:, :6].astype(float)
    tau = data["left_feedback_effort"][:, :6].astype(float)

    command_time = data["left_command_time"].astype(float)
    command_phase = data["left_command_phase"].astype(int)
    trajectory_time = command_time[command_phase == 1]
    start = float(trajectory_time[0] + edge_trim_sec)
    stop = float(trajectory_time[-1] - edge_trim_sec)
    mask = (time >= start) & (time <= stop)

    time = time[mask]
    time = time - time[0]
    sample_rate = 1.0 / float(np.median(np.diff(time)))
    q = q[mask]
    qd = lowpass_zero_phase(
        qd[mask],
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        order=4,
    )
    qdd = lowpass_zero_phase(
        central_difference(time, qd),
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        order=4,
    )
    tau = lowpass_zero_phase(
        tau[mask],
        sample_rate=sample_rate,
        cutoff_hz=cutoff_hz,
        order=4,
    )
    return {
        "time": time,
        "q": q,
        "qd": qd,
        "qdd": qdd,
        "tau": tau,
        "sample_rate": np.array(sample_rate),
    }


def build_observation(
    dynamics: PinocchioDynamicsModel,
    base_projection: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> np.ndarray:
    rows = []
    for qi, qdi, qddi in zip(q, qd, qdd):
        y_base = dynamics.regressor(qi, qdi, qddi) @ base_projection
        y_friction = linear_friction_regressor(qdi)
        rows.append(np.concatenate([y_base, y_friction], axis=1))
    return np.vstack(rows)


def pseudo_inertia_constraint(pi_dynamic, block_start: int):
    inertia = cp.bmat(
        [
            [
                pi_dynamic[block_start + 4],
                pi_dynamic[block_start + 5],
                pi_dynamic[block_start + 7],
            ],
            [
                pi_dynamic[block_start + 5],
                pi_dynamic[block_start + 6],
                pi_dynamic[block_start + 8],
            ],
            [
                pi_dynamic[block_start + 7],
                pi_dynamic[block_start + 8],
                pi_dynamic[block_start + 9],
            ],
        ]
    )
    first_moment = pi_dynamic[block_start + 1 : block_start + 4]
    return cp.bmat(
        [
            [0.5 * cp.trace(inertia) * np.eye(3) - inertia, first_moment[:, None]],
            [first_moment[None, :], cp.reshape(pi_dynamic[block_start], (1, 1), order="C")],
        ]
    )


def metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    residual = measured - predicted
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    centered = measured - measured.mean(axis=0)
    fit = 100.0 * (1.0 - np.linalg.norm(residual, axis=0) / np.linalg.norm(centered, axis=0))
    return {
        "rmse_per_joint": rmse.tolist(),
        "fit_percent_per_joint": fit.tolist(),
        "overall_rmse": float(np.sqrt(np.mean(residual**2))),
        "overall_relative_rmse": float(
            np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(centered**2))
        ),
    }


def write_parameters(path: Path, full: np.ndarray, friction: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "joint", "parameter", "value"])
        for joint in range(6):
            block = full[joint * 10 : joint * 10 + 10]
            for name, value in zip(PARAMETER_NAMES, block):
                writer.writerow(["rigid_body", joint + 1, name, value])
        for joint in range(6):
            block = friction[joint * 3 : joint * 3 + 3]
            for name, value in zip(["viscous", "coulomb", "offset"], block):
                writer.writerow(["friction", joint + 1, name, value])


def write_comparison(path: Path, nominal: np.ndarray, full: np.ndarray, friction: np.ndarray) -> None:
    identified = np.concatenate([full, friction])
    before = np.concatenate([nominal, np.zeros(18)])
    names = PARAMETER_NAMES * 6 + ["viscous", "coulomb", "offset"] * 6
    groups = ["rigid_body"] * 60 + ["friction"] * 18
    joints = [joint for joint in range(1, 7) for _ in range(10)]
    joints += [joint for joint in range(1, 7) for _ in range(3)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "joint", "parameter", "before_urdf", "after_cvxpy", "delta"])
        for group, joint, name, old, new in zip(groups, joints, names, before, identified):
            writer.writerow([group, joint, name, old, new, new - old])


def save_torque_plots(
    output_dir: Path,
    time: np.ndarray,
    measured: np.ndarray,
    initial: np.ndarray,
    predicted: np.ndarray,
) -> list[str]:
    matplotlib_config = output_dir / ".matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure_path = output_dir / "piper_torque_cvxpy_physical.png"
    residual_path = output_dir / "piper_torque_residual_cvxpy_physical.png"
    rmse_path = output_dir / "piper_torque_rmse_cvxpy_physical.png"

    fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True, layout="constrained")
    for joint, ax in enumerate(axes, start=1):
        ax.plot(
            time,
            measured[:, joint - 1],
            color="black",
            linewidth=1.1,
            label="Measured" if joint == 1 else None,
        )
        ax.plot(
            time,
            initial[:, joint - 1],
            color="#d55e00",
            linewidth=0.9,
            alpha=0.9,
            label="Initial URDF + zero friction" if joint == 1 else None,
        )
        ax.plot(
            time,
            predicted[:, joint - 1],
            color="#0072b2",
            linewidth=1.0,
            label="CVXPY physical" if joint == 1 else None,
        )
        ax.set_ylabel(f"J{joint}\nNm")
        ax.margins(x=0)
    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Time in trajectory window (s)")
    fig.suptitle("Piper Torque Prediction: CVXPY Physical Constraints", fontsize=14)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    initial_residual = measured - initial
    residual = measured - predicted
    fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True, layout="constrained")
    for joint, ax in enumerate(axes, start=1):
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.plot(
            time,
            initial_residual[:, joint - 1],
            color="#d55e00",
            linewidth=0.9,
            alpha=0.9,
            label="Measured - initial" if joint == 1 else None,
        )
        ax.plot(
            time,
            residual[:, joint - 1],
            color="#0072b2",
            linewidth=1.0,
            label="Measured - CVXPY physical" if joint == 1 else None,
        )
        ax.set_ylabel(f"J{joint}\nNm")
        ax.margins(x=0)
    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Time in trajectory window (s)")
    fig.suptitle("Piper Torque Residuals: CVXPY Physical Constraints", fontsize=14)
    fig.savefig(residual_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    initial_rmse = np.sqrt(np.mean(initial_residual**2, axis=0))
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(6)
    width = 0.36
    ax.bar(x - width / 2, initial_rmse, width, color="#d55e00", label="Initial")
    ax.bar(x + width / 2, rmse, width, color="#0072b2", label="CVXPY physical")
    ax.set_xticks(np.arange(6), [f"J{joint}" for joint in range(1, 7)])
    ax.set_ylabel("RMSE (Nm)")
    ax.set_title("Torque Prediction RMSE by Joint")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.35)
    fig.savefig(rmse_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return [figure_path.name, residual_path.name, rmse_path.name]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("piper_data/piper_excitation_20260430_170440"))
    parser.add_argument("--urdf", type=Path, default=Path("agx_arm_urdf/piper/urdf/piper_description.urdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/piper_dynamics_identification_20260430_170440_cvxpy"))
    parser.add_argument("--cutoff-hz", type=float, default=8.0)
    parser.add_argument("--edge-trim-sec", type=float, default=0.25)
    parser.add_argument("--base-sample-count", type=int, default=1000)
    parser.add_argument("--mass-error-range", type=float, default=0.25)
    parser.add_argument("--solver", default="CLARABEL")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.data_dir / "piper_excitation_collection.npz", args.cutoff_hz, args.edge_trim_sec)
    dynamics = PinocchioDynamicsModel.from_urdf(args.urdf, model_only=True)
    base = dynamics.extract_base_parameters(
        args.base_sample_count,
        rng=np.random.default_rng(0),
    )
    observation = build_observation(
        dynamics,
        base.base_projection,
        dataset["q"],
        dataset["qd"],
        dataset["qdd"],
    )
    measured_torque = dataset["tau"].reshape(-1)

    nb = base.num_base_params
    nd = base.num_dep_params
    pi_base = cp.Variable(nb)
    pi_dep = cp.Variable(nd)
    pi_friction = cp.Variable(18)
    reconstruction = base.mapping.reconstruction_matrix()
    pi_dynamic = reconstruction @ cp.hstack([pi_base, pi_dep])

    constraints = []
    nominal_dynamic = dynamics.dynamic_parameter_vector()
    nominal_masses = nominal_dynamic[0::10]
    mass_indices = np.arange(0, nominal_dynamic.size, 10)
    constraints.append(pi_dynamic[mass_indices] >= nominal_masses * (1.0 - args.mass_error_range))
    constraints.append(pi_dynamic[mass_indices] <= nominal_masses * (1.0 + args.mass_error_range))
    for block_start in range(0, nominal_dynamic.size, 10):
        constraints.append(pseudo_inertia_constraint(pi_dynamic, block_start) >> 0)
    constraints.append(pi_friction[0::3] >= 0.0)
    constraints.append(pi_friction[1::3] >= 0.0)

    parameters = cp.hstack([pi_base, pi_friction])
    residual = observation @ parameters - measured_torque
    problem = cp.Problem(cp.Minimize(cp.sum_squares(residual)), constraints)
    problem.solve(solver=args.solver, verbose=False)

    if pi_base.value is None or pi_dep.value is None or pi_friction.value is None:
        raise RuntimeError(f"CVXPY solve failed with status {problem.status}")
    base_value = np.asarray(pi_base.value, dtype=float).reshape(-1)
    dep_value = np.asarray(pi_dep.value, dtype=float).reshape(-1)
    friction_value = np.asarray(pi_friction.value, dtype=float).reshape(-1)
    full_value = reconstruction @ np.concatenate([base_value, dep_value])
    identified = np.concatenate([base_value, friction_value])
    predicted = (observation @ identified).reshape((-1, 6))
    measured = dataset["tau"]
    nominal_base = base.mapping.base_parameter_vector(nominal_dynamic)
    nominal_identified = np.concatenate([nominal_base, np.zeros(18)])
    initial_predicted = (observation @ nominal_identified).reshape((-1, 6))

    write_parameters(args.output_dir / "piper_dynamics_parameters_cvxpy.csv", full_value, friction_value)
    write_comparison(
        args.output_dir / "piper_dynamics_parameter_comparison_cvxpy.csv",
        nominal_dynamic,
        full_value,
        friction_value,
    )
    prediction_table = np.column_stack(
        [dataset["time"], measured, initial_predicted, predicted]
    )
    header = (
        ["time"]
        + [f"tau_measured_j{i}" for i in range(1, 7)]
        + [f"tau_initial_j{i}" for i in range(1, 7)]
        + [f"tau_cvxpy_j{i}" for i in range(1, 7)]
    )
    np.savetxt(
        args.output_dir / "piper_torque_prediction_cvxpy.csv",
        prediction_table,
        delimiter=",",
        header=",".join(header),
        comments="",
    )
    np.savez(
        args.output_dir / "piper_dynamics_identification_cvxpy.npz",
        base_parameters=base_value,
        dependent_parameters=dep_value,
        full_dynamic_parameters=full_value,
        friction_parameters=friction_value,
        observation_matrix=observation,
        measured_torque=measured_torque,
        predicted_torque=predicted.reshape(-1),
        q=dataset["q"],
        qd=dataset["qd"],
        qdd=dataset["qdd"],
        tau=dataset["tau"],
        time=dataset["time"],
    )
    plot_outputs = save_torque_plots(
        args.output_dir,
        dataset["time"],
        measured,
        initial_predicted,
        predicted,
    )
    summary = {
        "backend": "pinocchio+cvxpy",
        "solver": args.solver,
        "status": str(problem.status),
        "success": problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE},
        "objective_value": float(problem.value),
        "mass_error_range": args.mass_error_range,
        "sample_rate_hz": float(dataset["sample_rate"]),
        "used_samples": int(dataset["time"].size),
        "base_parameters": int(nb),
        "dependent_parameters": int(nd),
        "observation_shape": list(observation.shape),
        "initial_urdf_zero_friction_metrics": metrics(measured, initial_predicted),
        "metrics": metrics(measured, predicted),
        "outputs": [
            "piper_dynamics_parameters_cvxpy.csv",
            "piper_dynamics_parameter_comparison_cvxpy.csv",
            "piper_torque_prediction_cvxpy.csv",
            "piper_dynamics_identification_cvxpy.npz",
            *plot_outputs,
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
