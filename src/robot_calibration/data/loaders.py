"""Load calibration datasets from common file formats."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from robot_calibration.data.dataset import CalibrationDataset


def _column_stack(data: np.ndarray, columns: Sequence[str] | None) -> np.ndarray | None:
    if columns is None:
        return None
    missing = [column for column in columns if column not in data.dtype.names]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    return np.column_stack([data[column] for column in columns])


def load_csv(
    path: str | Path,
    *,
    time_column: str = "time",
    position_columns: Sequence[str],
    velocity_columns: Sequence[str] | None = None,
    acceleration_columns: Sequence[str] | None = None,
    torque_columns: Sequence[str] | None = None,
    current_columns: Sequence[str] | None = None,
    desired_current_columns: Sequence[str] | None = None,
    desired_torque_columns: Sequence[str] | None = None,
) -> CalibrationDataset:
    """Load a calibration dataset from a headered CSV file."""

    csv_path = Path(path)
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.dtype.names is None:
        raise ValueError("CSV file must contain a header row")
    if time_column not in data.dtype.names:
        raise ValueError(f"missing time column: {time_column}")

    return CalibrationDataset(
        time=np.atleast_1d(data[time_column]),
        positions=_column_stack(data, position_columns),
        velocities=_column_stack(data, velocity_columns),
        accelerations=_column_stack(data, acceleration_columns),
        torques=_column_stack(data, torque_columns),
        currents=_column_stack(data, current_columns),
        desired_currents=_column_stack(data, desired_current_columns),
        desired_torques=_column_stack(data, desired_torque_columns),
        metadata={"source": str(csv_path)},
    )


def load_ur_csv(
    path: str | Path,
    *,
    start_index: int | None = None,
    stop_index: int | None = None,
) -> CalibrationDataset:
    """Load a UR-style CSV without a header.

    The expected column layout follows the reference MATLAB parser:
    time, q1..q6, qd1..qd6, i1..i6, i_des1..i_des6, tau_des1..tau_des6.
    ``start_index`` and ``stop_index`` use Python slicing semantics.
    """

    csv_path = Path(path)
    data = np.loadtxt(csv_path, delimiter=",")
    data = np.atleast_2d(data)
    if data.shape[1] < 31:
        raise ValueError("UR CSV must contain at least 31 columns")

    start = 0 if start_index is None else start_index
    stop = data.shape[0] if stop_index is None else stop_index
    if start < 0 or stop > data.shape[0] or start >= stop:
        raise ValueError("invalid UR CSV crop range")
    window = data[start:stop]
    time = window[:, 0] - window[0, 0]

    return CalibrationDataset(
        time=time,
        positions=window[:, 1:7],
        velocities=window[:, 7:13],
        currents=window[:, 13:19],
        desired_currents=window[:, 19:25],
        desired_torques=window[:, 25:31],
        metadata={"source": str(csv_path), "format": "ur"},
    )
