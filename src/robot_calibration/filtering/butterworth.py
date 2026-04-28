"""Butterworth filtering helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lowpass_zero_phase(
    signal: ArrayLike,
    *,
    sample_rate: float | None = None,
    cutoff_hz: float | None = None,
    normalized_cutoff: float | None = None,
    order: int = 5,
    padtype: str | None = "odd",
) -> NDArray[np.float64]:
    """Apply zero-phase Butterworth low-pass filtering along samples."""

    try:
        from scipy.signal import butter, filtfilt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for lowpass_zero_phase; install robot-calibration with dev dependencies"
        ) from exc

    array = np.asarray(signal, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("signal must be a 1D or 2D array")
    if (cutoff_hz is None) == (normalized_cutoff is None):
        raise ValueError(
            "exactly one of cutoff_hz or normalized_cutoff must be provided"
        )
    if cutoff_hz is not None:
        if sample_rate is None or sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive")
        if cutoff_hz <= 0.0 or cutoff_hz >= sample_rate / 2.0:
            raise ValueError("cutoff_hz must be between 0 and Nyquist frequency")
        wn = cutoff_hz / (0.5 * sample_rate)
    else:
        wn = float(normalized_cutoff)
        if wn <= 0.0 or wn >= 1.0:
            raise ValueError("normalized_cutoff must be between 0 and 1")

    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, array, axis=0, padtype=padtype)
