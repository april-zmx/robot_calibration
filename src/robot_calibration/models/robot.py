"""Robot model primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DHParameter:
    """Standard Denavit-Hartenberg parameter row."""

    a: float
    alpha: float
    d: float
    theta: float


@dataclass(frozen=True)
class CalibrationResult:
    """Generic calibration result."""

    success: bool
    parameters: list[DHParameter]
    residuals: NDArray[np.float64]
    cost: float
    message: str = ""
