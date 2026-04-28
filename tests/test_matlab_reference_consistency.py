import importlib.util
from pathlib import Path
import sys

import pytest


REFERENCE_URDF = Path("/home/april/robotics/dynamic_calibration/URe/urdf/ur10e.urdf")
REFERENCE_DATASET = Path(
    "/home/april/robotics/dynamic_calibration/dataset_ur10e/"
    "identification_data/ur-20_02_10-30sec_12harm.csv"
)
REFERENCE_NOTEBOOK = Path("/home/april/robotics/dynamic_calibration/robot_dynamics.ipynb")
REFERENCE_TRAJECTORY_MAT = Path(
    "/home/april/robotics/dynamic_calibration/trajectory_optmzn/optimal_trjctrs/"
    "ptrnSrch_N12T30QR.mat"
)


def load_example_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "matlab_reference_consistency.py"
    )
    spec = importlib.util.spec_from_file_location("matlab_reference_consistency", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    any(
        not path.exists()
        for path in (
            REFERENCE_URDF,
            REFERENCE_DATASET,
            REFERENCE_NOTEBOOK,
            REFERENCE_TRAJECTORY_MAT,
        )
    ),
    reason="reference MATLAB assets are not available",
)
def test_build_report_matches_reference_matlab_assets():
    module = load_example_module()

    report = module.build_report()
    identification = report["identification_comparison"]
    trajectory = report["trajectory_comparison"]

    assert identification["base_parameter_count"] == 40
    assert identification["full_parameter_count"] == 66
    assert identification["pi_base_urdf"]["max_abs"] < 1.0e-8
    assert identification["pi_base"]["max_abs"] < 1.0e-6
    assert identification["pi_full"]["max_abs"] < 1.0e-4
    assert identification["pi_frictions"]["max_abs"] < 1.0e-7
    assert identification["est_std"]["max_abs"] < 1.0e-7
    assert identification["est_rel_std"]["max_abs"] < 1.0e-4
    assert trajectory["boundary_polynomial"]["max_abs"] < 1.0e-10
    assert trajectory["sampled_q"]["max_abs"] < 1.0e-10
    assert trajectory["sampled_qd"]["max_abs"] < 1.0e-10
    assert trajectory["sampled_qdd"]["max_abs"] < 1.0e-10
    assert trajectory["matlab_coefficients_in_python_observation"]["observation_rank"] == 58
    assert trajectory["matlab_coefficients_in_python_observation"]["target_rank"] == 58
