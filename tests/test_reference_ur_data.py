from pathlib import Path

import pytest

from robot_calibration.data.loaders import load_ur_csv
from robot_calibration.filtering import (
    preprocess_dataset,
    preprocess_reference_ur10e_dataset,
)


REFERENCE_UR_CSV = Path(
    "/home/april/robotics/dynamic_calibration/dataset_ur10e/"
    "identification_data/ur-20_02_10-30sec_12harm.csv"
)


@pytest.mark.skipif(
    not REFERENCE_UR_CSV.exists(),
    reason="reference UR dataset is not available",
)
def test_reference_ur_csv_loads_and_preprocesses_small_window():
    dataset = load_ur_csv(REFERENCE_UR_CSV, start_index=635, stop_index=900)

    processed = preprocess_dataset(
        dataset,
        velocity_cutoff_hz=2.0,
        current_cutoff_hz=2.0,
        acceleration_cutoff_hz=2.0,
    )

    assert processed.n_samples == 265
    assert processed.n_joints == 6
    assert processed.time[0] == 0.0
    assert processed.velocities.shape == (265, 6)
    assert processed.accelerations.shape == (265, 6)
    assert processed.currents.shape == (265, 6)
    assert processed.desired_currents.shape == (265, 6)
    assert processed.desired_torques.shape == (265, 6)


@pytest.mark.skipif(
    not REFERENCE_UR_CSV.exists(),
    reason="reference UR dataset is not available",
)
def test_reference_ur_csv_supports_notebook_style_preprocessing():
    dataset = load_ur_csv(REFERENCE_UR_CSV, start_index=635, stop_index=900)

    processed = preprocess_reference_ur10e_dataset(dataset)

    assert processed.n_samples == 265
    assert processed.n_joints == 6
    assert processed.time[0] == 0.0
    assert processed.velocities.shape == (265, 6)
    assert processed.accelerations.shape == (265, 6)
    assert processed.currents.shape == (265, 6)
    assert processed.desired_currents.shape == (265, 6)
    assert processed.desired_torques.shape == (265, 6)
