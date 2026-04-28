"""Visualize a generated Piper excitation trajectory in ROS2 RViz."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrajectoryData:
    time: NDArray[np.float64]
    q: NDArray[np.float64]
    joint_names: list[str]


def write_rviz_urdf(
    source_urdf: str | Path,
    agx_root: str | Path,
    output_urdf: str | Path,
) -> Path:
    """Rewrite Piper package mesh URIs to local file URIs for RViz."""

    root = Path(agx_root).resolve()
    text = Path(source_urdf).read_text(encoding="utf-8")
    replacements = {
        "package://agx_arm_description/agx_arm_urdf/": f"file://{root}/",
        "package://agx_arm_urdf/": f"file://{root}/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    output = Path(output_urdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def load_trajectory(path: str | Path) -> TrajectoryData:
    data = np.load(path)
    time_array = np.asarray(data["time"], dtype=float)
    q = np.asarray(data["q"], dtype=float)
    joint_names = [str(name) for name in data["joint_names"].tolist()]
    if time_array.ndim != 1:
        raise ValueError("time must be 1D")
    if q.ndim != 2 or q.shape[1] != time_array.size:
        raise ValueError("q must have shape (n_joints, n_samples)")
    if q.shape[0] != len(joint_names):
        raise ValueError("joint_names length must match q rows")
    return TrajectoryData(time=time_array, q=q, joint_names=joint_names)


def ros_environment(*, software_rendering: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    log_dir = Path("/tmp/ros-log")
    log_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("ROS_LOG_DIR", str(log_dir))
    if software_rendering:
        env["LIBGL_ALWAYS_SOFTWARE"] = "1"
        env["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"
        env["QT_X11_NO_MITSHM"] = "1"
    return env


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"{name} was not found. Run: source /opt/ros/humble/setup.bash"
        )


def _start_robot_state_publisher(urdf_path: Path, env: dict[str, str]) -> subprocess.Popen:
    _require_executable("ros2")
    return subprocess.Popen(
        [
            "ros2",
            "run",
            "robot_state_publisher",
            "robot_state_publisher",
            str(urdf_path),
        ],
        env=env,
    )


def publish_joint_states(
    trajectory: TrajectoryData,
    *,
    rate_scale: float = 1.0,
    loop: bool = True,
) -> None:
    """Publish trajectory samples to /joint_states using rclpy."""

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    if rate_scale <= 0.0:
        raise ValueError("rate_scale must be positive")

    rclpy.init()
    node = Node("piper_excitation_joint_state_player")
    publisher = node.create_publisher(JointState, "joint_states", 10)
    try:
        while rclpy.ok():
            start = time.monotonic()
            for sample in range(trajectory.time.size):
                if not rclpy.ok():
                    break
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name = trajectory.joint_names
                msg.position = trajectory.q[:, sample].astype(float).tolist()
                publisher.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                if sample + 1 < trajectory.time.size:
                    dt = trajectory.time[sample + 1] - trajectory.time[sample]
                    target = start + float(trajectory.time[sample + 1] / rate_scale)
                    time.sleep(max(0.0, min(float(dt / rate_scale), target - time.monotonic())))
            if not loop:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        default="outputs/piper_excitation/piper_excitation.npz",
        help="Trajectory NPZ produced by examples/piper_excitation_trajectory.py.",
    )
    parser.add_argument(
        "--urdf",
        default="agx_arm_urdf/piper/urdf/piper_description.urdf",
        help="Original Piper URDF.",
    )
    parser.add_argument(
        "--agx-root",
        default="agx_arm_urdf",
        help="Path to the cloned agx_arm_urdf repository.",
    )
    parser.add_argument("--output-dir", default="outputs/piper_rviz")
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--software-rendering",
        action="store_true",
        help="Set software rendering environment variables for child ROS processes.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    rviz_urdf = write_rviz_urdf(
        args.urdf,
        args.agx_root,
        output / "piper_rviz.urdf",
    )
    trajectory = load_trajectory(args.trajectory)
    env = ros_environment(software_rendering=args.software_rendering)
    processes: list[subprocess.Popen] = []
    try:
        processes.append(_start_robot_state_publisher(rviz_urdf, env))
        print("Publishing Piper trajectory to /joint_states.")
        print(f"RViz URDF: {rviz_urdf}")
        print("Open RViz2 yourself and add RobotModel/TF displays.")
        publish_joint_states(
            trajectory,
            rate_scale=args.rate_scale,
            loop=not args.no_loop,
        )
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
