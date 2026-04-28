"""Observation matrix assembly for dynamic identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from robot_calibration.data import CalibrationDataset


class Regressor(Protocol):
    """Maps a single sample state to an inverse-dynamics regressor."""

    def __call__(
        self,
        q: NDArray[np.float64],
        qd: NDArray[np.float64],
        qdd: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        ...


@dataclass(frozen=True)
class ObservationMatrix:
    matrix: NDArray[np.float64]
    torque: NDArray[np.float64]


def assemble_observation_matrix(
    dataset: CalibrationDataset,
    regressor: Regressor,
) -> ObservationMatrix:
    """Stack per-sample regressors and measured torques."""

    if dataset.velocities is None or dataset.accelerations is None:
        raise ValueError("velocities and accelerations are required")
    if dataset.torques is None:
        raise ValueError("torques are required")

    rows = []
    torque = []
    for q, qd, qdd, tau in zip(
        dataset.positions,
        dataset.velocities,
        dataset.accelerations,
        dataset.torques,
    ):
        yi = np.asarray(regressor(q, qd, qdd), dtype=float)
        if yi.ndim != 2 or yi.shape[0] != dataset.n_joints:
            raise ValueError("regressor output must have one row per joint")
        rows.append(yi)
        torque.append(tau)

    return ObservationMatrix(
        matrix=np.vstack(rows),
        torque=np.asarray(torque, dtype=float).reshape(-1),
    )
