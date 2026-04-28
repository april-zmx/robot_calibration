import numpy as np
import pytest

from robot_calibration.data import CalibrationDataset


def test_dataset_infers_sample_and_joint_count():
    dataset = CalibrationDataset(
        time=np.array([0.0, 0.1, 0.2]),
        positions=np.zeros((3, 2)),
    )

    assert dataset.n_samples == 3
    assert dataset.n_joints == 2


def test_dataset_rejects_non_monotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        CalibrationDataset(
            time=np.array([0.0, 0.2, 0.1]),
            positions=np.zeros((3, 2)),
        )


def test_dataset_rejects_mismatched_signal_shape():
    with pytest.raises(ValueError, match="velocities"):
        CalibrationDataset(
            time=np.array([0.0, 0.1, 0.2]),
            positions=np.zeros((3, 2)),
            velocities=np.zeros((3, 3)),
        )


def test_dataset_crop_by_index_preserves_optional_signals():
    dataset = CalibrationDataset(
        time=np.array([0.0, 0.1, 0.2, 0.3]),
        positions=np.arange(8, dtype=float).reshape(4, 2),
        currents=np.ones((4, 2)),
        desired_currents=2.0 * np.ones((4, 2)),
        desired_torques=3.0 * np.ones((4, 2)),
        metadata={"robot": "demo"},
    )

    cropped = dataset.crop_by_index(1, 3)

    np.testing.assert_allclose(cropped.time, [0.1, 0.2])
    np.testing.assert_allclose(cropped.positions, [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(cropped.currents, np.ones((2, 2)))
    np.testing.assert_allclose(cropped.desired_currents, 2.0 * np.ones((2, 2)))
    np.testing.assert_allclose(cropped.desired_torques, 3.0 * np.ones((2, 2)))
    assert cropped.metadata == {"robot": "demo"}
