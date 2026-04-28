import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "piper_excitation_trajectory.py"
    spec = importlib.util.spec_from_file_location("piper_excitation_trajectory", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_revolute_joint_limits_from_urdf(tmp_path):
    module = load_example_module()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
        <robot name="demo">
          <joint name="fixed" type="fixed"/>
          <joint name="joint1" type="revolute">
            <limit lower="-1.0" upper="2.0" velocity="3.0"/>
          </joint>
          <joint name="joint2" type="revolute">
            <limit lower="-0.5" upper="0.5" velocity="4.0"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )

    limits = module.parse_revolute_joint_limits(urdf)

    assert limits.names == ["joint1", "joint2"]
    np.testing.assert_allclose(limits.position, [[-1.0, 2.0], [-0.5, 0.5]])
    np.testing.assert_allclose(limits.velocity, [3.0, 4.0])


def test_example_uses_reference_default_max_evaluations():
    module = load_example_module()

    assert module.DEFAULT_MAX_EVALUATIONS == 1_000_000


def test_example_uses_unbounded_coefficients_by_default():
    module = load_example_module()

    assert module.DEFAULT_COEFFICIENT_BOUND is None


def test_write_trajectory_outputs_creates_data_summary_and_plot(tmp_path):
    module = load_example_module()
    time = np.linspace(0.0, 1.0, 5)
    q = np.vstack([time, -time])
    qd = np.vstack([np.ones_like(time), -np.ones_like(time)])
    qdd = np.zeros_like(q)

    module.write_trajectory_outputs(
        output_dir=tmp_path,
        time=time,
        q=q,
        qd=qd,
        qdd=qdd,
        sine_coefficients=np.ones((2, 1)),
        cosine_coefficients=np.zeros((2, 1)),
        condition_number=12.3,
        objective_value=45.6,
        observation_rank=4,
        target_rank=6,
        joint_names=["joint1", "joint2"],
        metadata={"urdf": "demo.urdf"},
    )

    assert (tmp_path / "piper_excitation.npz").exists()
    assert (tmp_path / "piper_excitation_summary.json").exists()
    assert (tmp_path / "piper_excitation_plot.png").exists()


def test_main_defaults_to_motor_dynamics_enabled_and_annotates_outputs(
    monkeypatch,
    tmp_path,
):
    module = load_example_module()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
        <robot name="demo">
          <joint name="joint1" type="revolute">
            <limit lower="-1.0" upper="1.0" velocity="2.0"/>
          </joint>
          <joint name="joint2" type="revolute">
            <limit lower="-1.5" upper="1.5" velocity="3.0"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeBaseParameters:
        mapping = object()
        num_base_params = 41
        full_parameters = np.zeros(66)

    class FakeDynamicsModel:
        nq = 2
        regressor = staticmethod(lambda _q, _qd, _qdd: np.zeros((2, 2)))
        friction_regressor = staticmethod(lambda _qd: np.zeros((2, 6)))

        def extract_base_parameters(self, _sample_count, *, rng):
            assert rng is not None
            return FakeBaseParameters()

    class FakeTrajectory:
        def sample_uniform(self, sample_count):
            time = np.linspace(0.0, 1.0, sample_count)
            q = np.zeros((2, sample_count))
            return time, q, q, q

    class FakeResult:
        sine_coefficients = np.zeros((2, 1))
        cosine_coefficients = np.zeros((2, 1))
        condition_number = 12.3
        objective_value = 12.3
        observation_rank = 4
        target_rank = 4
        used_initial_guess = False
        trajectory = FakeTrajectory()

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["optimizer_kwargs"] = kwargs

        def optimize(self, *, max_evaluations, seed):
            captured["optimize_kwargs"] = {
                "max_evaluations": max_evaluations,
                "seed": seed,
            }
            return FakeResult()

    def fake_from_urdf(_urdf_path, *, include_motor_dynamics, model_only):
        captured["include_motor_dynamics"] = include_motor_dynamics
        captured["model_only"] = model_only
        return FakeDynamicsModel()

    def fake_write_trajectory_outputs(**kwargs):
        captured["write_kwargs"] = kwargs

    monkeypatch.setattr(module.PinocchioDynamicsModel, "from_urdf", fake_from_urdf)
    monkeypatch.setattr(module, "ConditionNumberExcitationOptimizer", FakeOptimizer)
    monkeypatch.setattr(module, "write_trajectory_outputs", fake_write_trajectory_outputs)

    exit_code = module.main(["--urdf", str(urdf), "--output-dir", str(tmp_path / "out")])

    assert exit_code == 0
    assert captured["include_motor_dynamics"] is True
    assert captured["model_only"] is True
    assert captured["write_kwargs"]["metadata"]["include_motor_dynamics"] is True
    assert captured["write_kwargs"]["metadata"]["dynamics_variant"] == "motor_on"
    assert captured["write_kwargs"]["metadata"]["base_dynamic_parameter_count"] == 41
    assert captured["write_kwargs"]["metadata"]["full_dynamic_parameter_count"] == 66
    assert captured["write_kwargs"]["metadata"]["friction_parameter_count"] == 6
    assert captured["write_kwargs"]["metadata"]["augmented_target_rank"] == 4


def test_main_supports_motor_off_annotation(monkeypatch, tmp_path):
    module = load_example_module()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
        <robot name="demo">
          <joint name="joint1" type="revolute">
            <limit lower="-1.0" upper="1.0" velocity="2.0"/>
          </joint>
        </robot>
        """,
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeBaseParameters:
        mapping = object()
        num_base_params = 36
        full_parameters = np.zeros(60)

    class FakeDynamicsModel:
        nq = 1
        regressor = staticmethod(lambda _q, _qd, _qdd: np.zeros((1, 1)))
        friction_regressor = staticmethod(lambda _qd: np.zeros((1, 3)))

        def extract_base_parameters(self, _sample_count, *, rng):
            assert rng is not None
            return FakeBaseParameters()

    class FakeTrajectory:
        def sample_uniform(self, sample_count):
            time = np.linspace(0.0, 1.0, sample_count)
            q = np.zeros((1, sample_count))
            return time, q, q, q

    class FakeResult:
        sine_coefficients = np.zeros((1, 1))
        cosine_coefficients = np.zeros((1, 1))
        condition_number = 5.0
        objective_value = 5.0
        observation_rank = 1
        target_rank = 1
        used_initial_guess = False
        trajectory = FakeTrajectory()

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured["optimizer_kwargs"] = kwargs

        def optimize(self, *, max_evaluations, seed):
            captured["optimize_kwargs"] = {
                "max_evaluations": max_evaluations,
                "seed": seed,
            }
            return FakeResult()

    def fake_from_urdf(_urdf_path, *, include_motor_dynamics, model_only):
        captured["include_motor_dynamics"] = include_motor_dynamics
        captured["model_only"] = model_only
        return FakeDynamicsModel()

    def fake_write_trajectory_outputs(**kwargs):
        captured["write_kwargs"] = kwargs

    monkeypatch.setattr(module.PinocchioDynamicsModel, "from_urdf", fake_from_urdf)
    monkeypatch.setattr(module, "ConditionNumberExcitationOptimizer", FakeOptimizer)
    monkeypatch.setattr(module, "write_trajectory_outputs", fake_write_trajectory_outputs)

    exit_code = module.main(
        [
            "--urdf",
            str(urdf),
            "--output-dir",
            str(tmp_path / "out"),
            "--no-motor-dynamics",
        ]
    )

    assert exit_code == 0
    assert captured["include_motor_dynamics"] is False
    assert captured["model_only"] is True
    assert captured["write_kwargs"]["metadata"]["include_motor_dynamics"] is False
    assert captured["write_kwargs"]["metadata"]["dynamics_variant"] == "motor_off"
    assert captured["write_kwargs"]["metadata"]["base_dynamic_parameter_count"] == 36
    assert captured["write_kwargs"]["metadata"]["full_dynamic_parameter_count"] == 60
    assert captured["write_kwargs"]["metadata"]["friction_parameter_count"] == 3
    assert captured["write_kwargs"]["metadata"]["augmented_target_rank"] == 1
