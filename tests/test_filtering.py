import numpy as np
from scipy.signal import butter, filtfilt

from robot_calibration.data import CalibrationDataset
from robot_calibration.filtering import (
    central_difference,
    lowpass_zero_phase,
    preprocess_reference_ur10e_dataset,
    preprocess_dataset,
)


def test_central_difference_estimates_polynomial_derivative():
    time = np.linspace(0.0, 1.0, 11)
    signal = np.column_stack([time**2, 3.0 * time**2])

    derivative = central_difference(time, signal)

    np.testing.assert_allclose(derivative[1:-1, 0], 2.0 * time[1:-1], atol=1e-12)
    np.testing.assert_allclose(derivative[1:-1, 1], 6.0 * time[1:-1], atol=1e-12)


def test_lowpass_zero_phase_preserves_shape():
    time = np.linspace(0.0, 2.0, 200)
    signal = np.column_stack([np.sin(2 * np.pi * time)])

    filtered = lowpass_zero_phase(signal, sample_rate=100.0, cutoff_hz=5.0)

    assert filtered.shape == signal.shape


def test_preprocess_dataset_returns_new_dataset_with_accelerations():
    time = np.linspace(0.0, 2.0, 200)
    velocities = np.column_stack([np.sin(time), np.cos(time)])
    currents = np.column_stack([np.sin(2 * time), np.cos(2 * time)])
    dataset = CalibrationDataset(
        time=time,
        positions=np.zeros((time.size, 2)),
        velocities=velocities,
        currents=currents,
        desired_currents=currents + 1.0,
        desired_torques=currents + 2.0,
    )

    processed = preprocess_dataset(
        dataset,
        velocity_cutoff_hz=5.0,
        current_cutoff_hz=5.0,
    )

    assert processed is not dataset
    assert processed.velocities.shape == velocities.shape
    assert processed.accelerations.shape == velocities.shape
    assert processed.currents.shape == currents.shape
    assert processed.desired_currents.shape == currents.shape
    assert processed.desired_torques.shape == currents.shape
    assert np.linalg.norm(processed.desired_currents - dataset.desired_currents) > 0.0
    assert np.linalg.norm(processed.desired_torques - dataset.desired_torques) > 0.0


def test_preprocess_reference_ur10e_dataset_matches_notebook_filtering():
    time = np.linspace(0.0, 2.0, 200)
    velocities = np.column_stack([np.sin(time), np.cos(0.5 * time)])
    currents = np.column_stack([np.sin(2.0 * time), np.cos(1.5 * time)])
    dataset = CalibrationDataset(
        time=time,
        positions=np.zeros((time.size, 2)),
        velocities=velocities,
        currents=currents,
        desired_currents=currents + 1.0,
        desired_torques=currents + 2.0,
    )

    processed = preprocess_reference_ur10e_dataset(dataset)

    vel_b, vel_a = butter(5, 0.15)
    expected_velocities = filtfilt(
        vel_b,
        vel_a,
        velocities,
        axis=0,
        padtype=None,
    )
    qdd = np.zeros_like(expected_velocities)
    qdd[1:-1] = (
        expected_velocities[2:] - expected_velocities[:-2]
    ) / (time[2:, None] - time[:-2, None])
    acc_b, acc_a = butter(5, 0.15)
    expected_accelerations = filtfilt(
        acc_b,
        acc_a,
        qdd,
        axis=0,
        padtype=None,
    )
    cur_b, cur_a = butter(5, 0.2)
    expected_currents = filtfilt(
        cur_b,
        cur_a,
        currents,
        axis=0,
        padtype=None,
    )

    np.testing.assert_allclose(processed.velocities, expected_velocities)
    np.testing.assert_allclose(processed.accelerations, expected_accelerations)
    np.testing.assert_allclose(processed.currents, expected_currents)
    np.testing.assert_allclose(processed.desired_currents, dataset.desired_currents)
    np.testing.assert_allclose(processed.desired_torques, dataset.desired_torques)
