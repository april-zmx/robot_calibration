"""Composable preprocessing pipelines for calibration data."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from robot_calibration.data import CalibrationDataset
from robot_calibration.filtering.butterworth import lowpass_zero_phase
from robot_calibration.filtering.differentiation import central_difference


def _sample_rate(time: np.ndarray) -> float:
    dt = np.diff(time)
    return float(1.0 / np.median(dt))


def preprocess_dataset(
    dataset: CalibrationDataset,
    *,
    velocity_cutoff_hz: float | None = None,
    current_cutoff_hz: float | None = None,
    acceleration_cutoff_hz: float | None = None,
) -> CalibrationDataset:
    """Filter measured signals and estimate accelerations when possible."""

    rate = _sample_rate(dataset.time)
    velocities = dataset.velocities
    currents = dataset.currents
    desired_currents = dataset.desired_currents
    desired_torques = dataset.desired_torques

    if velocities is not None and velocity_cutoff_hz is not None:
        velocities = lowpass_zero_phase(
            velocities,
            sample_rate=rate,
            cutoff_hz=velocity_cutoff_hz,
        )

    accelerations = dataset.accelerations
    if accelerations is None and velocities is not None:
        accelerations = central_difference(dataset.time, velocities)

    if accelerations is not None and acceleration_cutoff_hz is not None:
        accelerations = lowpass_zero_phase(
            accelerations,
            sample_rate=rate,
            cutoff_hz=acceleration_cutoff_hz,
        )

    if currents is not None and current_cutoff_hz is not None:
        currents = lowpass_zero_phase(
            currents,
            sample_rate=rate,
            cutoff_hz=current_cutoff_hz,
        )
    if desired_currents is not None and current_cutoff_hz is not None:
        desired_currents = lowpass_zero_phase(
            desired_currents,
            sample_rate=rate,
            cutoff_hz=current_cutoff_hz,
        )
    if desired_torques is not None and current_cutoff_hz is not None:
        desired_torques = lowpass_zero_phase(
            desired_torques,
            sample_rate=rate,
            cutoff_hz=current_cutoff_hz,
        )

    return replace(
        dataset,
        positions=dataset.positions.copy(),
        velocities=None if velocities is None else velocities.copy(),
        accelerations=None if accelerations is None else accelerations.copy(),
        currents=None if currents is None else currents.copy(),
        desired_currents=None
        if desired_currents is None
        else desired_currents.copy(),
        desired_torques=None if desired_torques is None else desired_torques.copy(),
        torques=None if dataset.torques is None else dataset.torques.copy(),
        metadata=dict(dataset.metadata),
    )


def preprocess_reference_ur10e_dataset(
    dataset: CalibrationDataset,
    *,
    velocity_normalized_cutoff: float = 0.15,
    current_normalized_cutoff: float = 0.2,
    acceleration_normalized_cutoff: float = 0.15,
    filter_order: int = 5,
) -> CalibrationDataset:
    """Match the UR10e notebook/MATLAB preprocessing used for identification."""

    if dataset.velocities is None:
        raise ValueError("velocities are required")
    if dataset.currents is None:
        raise ValueError("currents are required")

    rate = _sample_rate(dataset.time)
    velocities = lowpass_zero_phase(
        dataset.velocities,
        sample_rate=rate,
        normalized_cutoff=velocity_normalized_cutoff,
        order=filter_order,
        padtype=None,
    )
    accelerations = np.zeros_like(velocities)
    accelerations[1:-1] = (
        velocities[2:] - velocities[:-2]
    ) / (dataset.time[2:, None] - dataset.time[:-2, None])
    accelerations = lowpass_zero_phase(
        accelerations,
        sample_rate=rate,
        normalized_cutoff=acceleration_normalized_cutoff,
        order=filter_order,
        padtype=None,
    )
    currents = lowpass_zero_phase(
        dataset.currents,
        sample_rate=rate,
        normalized_cutoff=current_normalized_cutoff,
        order=filter_order,
        padtype=None,
    )

    return replace(
        dataset,
        positions=dataset.positions.copy(),
        velocities=velocities.copy(),
        accelerations=accelerations.copy(),
        currents=currents.copy(),
        desired_currents=None
        if dataset.desired_currents is None
        else dataset.desired_currents.copy(),
        desired_torques=None
        if dataset.desired_torques is None
        else dataset.desired_torques.copy(),
        torques=None if dataset.torques is None else dataset.torques.copy(),
        metadata={**dataset.metadata, "preprocess": "reference_ur10e"},
    )
