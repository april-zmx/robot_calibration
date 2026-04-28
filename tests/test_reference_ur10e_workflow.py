from pathlib import Path

import numpy as np
import pytest

from robot_calibration.data.loaders import load_ur_csv
from robot_calibration.dynamics import (
    CurrentDrivenBaseDynamicsIdentifier,
    PinocchioDynamicsModel,
)
from robot_calibration.filtering import (
    preprocess_dataset,
    preprocess_reference_ur10e_dataset,
)


REFERENCE_URDF = Path("/home/april/robotics/dynamic_calibration/URe/urdf/ur10e.urdf")
REFERENCE_UR_CSV = Path(
    "/home/april/robotics/dynamic_calibration/dataset_ur10e/"
    "identification_data/ur-20_02_10-30sec_12harm.csv"
)
REFERENCE_DRIVE_GAINS = np.array([14.87, 13.26, 11.13, 10.62, 11.03, 11.47])


@pytest.mark.skipif(
    (not REFERENCE_URDF.exists()) or (not REFERENCE_UR_CSV.exists()),
    reason="reference UR10e assets are not available",
)
def test_reference_ur10e_notebook_style_workflow_aligns_with_reference_assets():
    dataset = load_ur_csv(REFERENCE_UR_CSV, start_index=635, stop_index=3510)
    generic = preprocess_dataset(
        dataset,
        velocity_cutoff_hz=2.0,
        current_cutoff_hz=2.0,
        acceleration_cutoff_hz=2.0,
    )
    reference = preprocess_reference_ur10e_dataset(dataset)

    model = PinocchioDynamicsModel.from_urdf(
        REFERENCE_URDF,
        include_motor_dynamics=True,
        model_only=True,
    )
    base = model.extract_base_parameters(50, rng=np.random.default_rng(0))
    identifier = CurrentDrivenBaseDynamicsIdentifier(
        model,
        base,
        drive_gains=REFERENCE_DRIVE_GAINS,
    )

    generic_result = identifier.identify(generic, method="least_square")
    reference_result = identifier.identify(reference, method="least_square")
    generic_residual = np.linalg.norm(
        generic_result.measured_torque - generic_result.predicted_torque
    )
    reference_residual = np.linalg.norm(
        reference_result.measured_torque - reference_result.predicted_torque
    )

    assert base.num_base_params == 40
    assert reference_result.observation_matrix.shape == (dataset.n_samples * 6, 58)
    assert reference_result.observation_matrix.shape[1] == base.num_base_params + 18
    assert np.linalg.matrix_rank(reference_result.observation_matrix) == 58
    assert reference_residual < generic_residual
