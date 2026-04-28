import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "piper_rviz_visualization.py"
    spec = importlib.util.spec_from_file_location("piper_rviz_visualization", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rewrite_piper_mesh_uris_to_local_files(tmp_path):
    module = load_example_module()
    urdf = tmp_path / "robot.urdf"
    agx_root = tmp_path / "agx_arm_urdf"
    agx_root.mkdir()
    urdf.write_text(
        """
        <robot name="demo">
          <link name="base">
            <visual>
              <geometry>
                <mesh filename="package://agx_arm_description/agx_arm_urdf/piper/meshes/link1.stl"/>
              </geometry>
            </visual>
          </link>
        </robot>
        """,
        encoding="utf-8",
    )
    output = tmp_path / "rviz.urdf"

    module.write_rviz_urdf(urdf, agx_root, output)

    text = output.read_text(encoding="utf-8")
    assert "package://" not in text
    assert f"file://{agx_root}/piper/meshes/link1.stl" in text


def test_load_trajectory_npz_returns_joint_names(tmp_path):
    module = load_example_module()
    path = tmp_path / "trajectory.npz"
    np.savez(
        path,
        time=np.array([0.0, 1.0]),
        q=np.zeros((2, 2)),
        joint_names=np.array(["joint1", "joint2"]),
    )

    trajectory = module.load_trajectory(path)

    assert trajectory.joint_names == ["joint1", "joint2"]
    np.testing.assert_allclose(trajectory.time, [0.0, 1.0])
    assert trajectory.q.shape == (2, 2)


def test_ros_environment_can_enable_software_rendering(monkeypatch):
    module = load_example_module()
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    monkeypatch.delenv("MESA_LOADER_DRIVER_OVERRIDE", raising=False)

    env = module.ros_environment(software_rendering=True)

    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert env["MESA_LOADER_DRIVER_OVERRIDE"] == "llvmpipe"
    assert env["QT_X11_NO_MITSHM"] == "1"
