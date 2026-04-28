"""Generate and visualize a Piper excitation trajectory from a URDF."""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
from numpy.typing import NDArray

from robot_calibration.dynamics import PinocchioDynamicsModel
from robot_calibration.trajectory import ConditionNumberExcitationOptimizer

BASE_PARAMETER_SAMPLE_COUNT = 50
DEFAULT_MAX_EVALUATIONS = 1_000_000
DEFAULT_COEFFICIENT_BOUND: float | None = None
PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO0p0qsAAAAASUVORK5CYII="
)


@dataclass(frozen=True)
class JointLimits:
    names: list[str]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]


def dynamics_variant_label(include_motor_dynamics: bool) -> str:
    """Return a compact label for the selected dynamics model."""

    return "motor_on" if include_motor_dynamics else "motor_off"


def parse_revolute_joint_limits(urdf_path: str | Path) -> JointLimits:
    """Read revolute joint limits from a URDF file in document order."""

    root = ET.parse(urdf_path).getroot()
    names = []
    position = []
    velocity = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"joint {joint.attrib.get('name', '<unnamed>')} has no limit")
        names.append(joint.attrib["name"])
        position.append([float(limit.attrib["lower"]), float(limit.attrib["upper"])])
        velocity.append(float(limit.attrib["velocity"]))
    if not names:
        raise ValueError("URDF contains no revolute joints")
    return JointLimits(
        names=names,
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
    )


def write_trajectory_outputs(
    *,
    output_dir: str | Path,
    time: NDArray[np.float64],
    q: NDArray[np.float64],
    qd: NDArray[np.float64],
    qdd: NDArray[np.float64],
    sine_coefficients: NDArray[np.float64],
    cosine_coefficients: NDArray[np.float64],
    condition_number: float,
    objective_value: float,
    observation_rank: int,
    target_rank: int,
    joint_names: list[str],
    metadata: dict,
) -> None:
    """Write trajectory arrays, summary metadata, and a joint plot."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "piper_excitation.npz",
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        sine_coefficients=sine_coefficients,
        cosine_coefficients=cosine_coefficients,
        joint_names=np.asarray(joint_names),
    )
    summary = {
        **metadata,
        "condition_number": float(condition_number),
        "objective_value": float(objective_value),
        "observation_rank": int(observation_rank),
        "target_rank": int(target_rank),
        "joint_names": joint_names,
        "sample_count": int(time.size),
        "duration": float(time[-1] - time[0]),
        "max_abs_position": np.max(np.abs(q), axis=1).tolist(),
        "max_abs_velocity": np.max(np.abs(qd), axis=1).tolist(),
        "max_abs_acceleration": np.max(np.abs(qdd), axis=1).tolist(),
    }
    (output / "piper_excitation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_path = output / "piper_excitation_plot.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        labels = ("q [rad]", "qd [rad/s]", "qdd [rad/s^2]")
        for ax, values, label in zip(axes, (q, qd, qdd), labels):
            for joint, name in enumerate(joint_names):
                ax.plot(time, values[joint], label=name)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        axes[0].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
    except Exception:
        plot_path.write_bytes(PLACEHOLDER_PNG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        default="agx_arm_urdf/piper/urdf/piper_description.urdf",
        help="Path to the Piper URDF.",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--n-harmonics", type=int, default=3)
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument(
        "--coefficient-bound",
        type=float,
        default=DEFAULT_COEFFICIENT_BOUND,
        help="Optional symmetric bound for Fourier coefficients. Omit to match MATLAB and rely on trajectory constraints only.",
    )
    parser.set_defaults(include_motor_dynamics=True)
    parser.add_argument(
        "--include-motor-dynamics",
        dest="include_motor_dynamics",
        action="store_true",
        help="Include one reflected motor inertia parameter per joint (default).",
    )
    parser.add_argument(
        "--no-motor-dynamics",
        dest="include_motor_dynamics",
        action="store_false",
        help="Use rigid-body-only dynamics parameters and annotate outputs as motor_off.",
    )
    parser.add_argument("--acceleration-limit", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/piper_excitation")
    args = parser.parse_args(argv)
    if args.coefficient_bound is not None and args.coefficient_bound < 0.0:
        parser.error("--coefficient-bound must be non-negative")

    urdf = Path(args.urdf)
    limits = parse_revolute_joint_limits(urdf)
    q0 = np.mean(limits.position, axis=1)
    dynamics_model = PinocchioDynamicsModel.from_urdf(
        urdf,
        include_motor_dynamics=args.include_motor_dynamics,
        model_only=True,
    )
    if dynamics_model.nq != len(limits.names):
        raise ValueError(
            f"Pinocchio model nq={dynamics_model.nq} does not match "
            f"{len(limits.names)} revolute URDF joints"
        )
    base_parameters = dynamics_model.extract_base_parameters(
        BASE_PARAMETER_SAMPLE_COUNT,
        rng=np.random.default_rng(args.seed),
    )

    optimizer = ConditionNumberExcitationOptimizer(
        q0=q0,
        n_harmonics=args.n_harmonics,
        fundamental_frequency=2.0 * np.pi / args.duration,
        duration=args.duration,
        sample_count=args.sample_count,
        coefficient_bounds=(
            None
            if args.coefficient_bound is None
            else (-args.coefficient_bound, args.coefficient_bound)
        ),
        regressor=dynamics_model.regressor,
        base_parameter_mapping=base_parameters.mapping,
        friction_regressor=dynamics_model.friction_regressor,
        joint_position_limits=limits.position,
        joint_velocity_limits=limits.velocity,
        joint_acceleration_limits=np.full(len(limits.names), args.acceleration_limit),
    )
    result = optimizer.optimize(max_evaluations=args.max_evaluations, seed=args.seed)
    time, q, qd, qdd = result.trajectory.sample_uniform(args.sample_count)

    write_trajectory_outputs(
        output_dir=args.output_dir,
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        sine_coefficients=result.sine_coefficients,
        cosine_coefficients=result.cosine_coefficients,
        condition_number=result.condition_number,
        objective_value=result.objective_value,
        observation_rank=result.observation_rank,
        target_rank=result.target_rank,
        joint_names=limits.names,
        metadata={
            "urdf": str(urdf),
            "include_motor_dynamics": args.include_motor_dynamics,
            "dynamics_variant": dynamics_variant_label(args.include_motor_dynamics),
            "n_harmonics": args.n_harmonics,
            "max_evaluations": args.max_evaluations,
            "coefficient_bound": args.coefficient_bound,
            "acceleration_limit": args.acceleration_limit,
            "base_dynamic_parameter_count": base_parameters.num_base_params,
            "base_parameter_count": base_parameters.num_base_params,
            "full_dynamic_parameter_count": base_parameters.full_parameters.size,
            "friction_parameter_count": len(limits.names) * 3,
            "augmented_target_rank": result.target_rank,
            "base_parameter_sample_count": BASE_PARAMETER_SAMPLE_COUNT,
            "used_initial_guess": result.used_initial_guess,
        },
    )
    print(f"condition number: {result.condition_number:.6g}")
    print(f"objective value: {result.objective_value:.6g}")
    print(f"observation rank: {result.observation_rank}/{result.target_rank}")
    if result.used_initial_guess:
        print(
            "warning: pymoo did not return a finite feasible solution; "
            "wrote the feasible zero-coefficient initial trajectory. "
            "Increase --max-evaluations for a real excitation trajectory."
        )
    print(f"wrote outputs to: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
