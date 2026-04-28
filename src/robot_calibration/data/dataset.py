"""Calibration trajectory data containers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_1d(name: str, value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def _as_2d(name: str, value: ArrayLike | None) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return array


@dataclass(frozen=True)
class CalibrationDataset:
    """Time-aligned joint-space signals used for calibration."""

    time: ArrayLike
    positions: ArrayLike
    velocities: ArrayLike | None = None
    accelerations: ArrayLike | None = None
    torques: ArrayLike | None = None
    currents: ArrayLike | None = None
    desired_currents: ArrayLike | None = None
    desired_torques: ArrayLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = _as_1d("time", self.time)
        positions = _as_2d("positions", self.positions)
        if positions is None:
            raise ValueError("positions are required")
        if positions.shape[0] != time.shape[0]:
            raise ValueError("positions must have one row per timestamp")
        if time.shape[0] > 1 and not np.all(np.diff(time) > 0.0):
            raise ValueError("time must be strictly increasing")

        object.__setattr__(self, "time", time)
        object.__setattr__(self, "positions", positions)

        for name in (
            "velocities",
            "accelerations",
            "torques",
            "currents",
            "desired_currents",
            "desired_torques",
        ):
            value = _as_2d(name, getattr(self, name))
            if value is not None and value.shape != positions.shape:
                raise ValueError(
                    f"{name} must have shape {positions.shape}, got {value.shape}"
                )
            object.__setattr__(self, name, value)

        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_samples(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_joints(self) -> int:
        return int(self.positions.shape[1])

    def crop_by_index(self, start: int, stop: int) -> "CalibrationDataset":
        """Return a dataset sliced over sample indices."""

        if start < 0 or stop > self.n_samples or start >= stop:
            raise ValueError("invalid crop range")
        window = slice(start, stop)
        return replace(
            self,
            time=self.time[window].copy(),
            positions=self.positions[window].copy(),
            velocities=None
            if self.velocities is None
            else self.velocities[window].copy(),
            accelerations=None
            if self.accelerations is None
            else self.accelerations[window].copy(),
            torques=None if self.torques is None else self.torques[window].copy(),
            currents=None if self.currents is None else self.currents[window].copy(),
            desired_currents=None
            if self.desired_currents is None
            else self.desired_currents[window].copy(),
            desired_torques=None
            if self.desired_torques is None
            else self.desired_torques[window].copy(),
            metadata=dict(self.metadata),
        )
