from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def estimate_curve_n_eff(y: np.ndarray, *, tolerance: float) -> float:
    """Estimate an effective number of resolved features in one smooth curve."""
    y = np.asarray(y, dtype=float)
    tolerance = float(tolerance)

    if y.ndim != 1 or y.size < 11:
        raise ValueError("curve must be one-dimensional with at least 11 values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("curve must contain only finite values.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite.")

    window_length = min(15, y.size if y.size % 2 else y.size - 1)
    y_smooth = savgol_filter(y, window_length=window_length, polyorder=3)

    noise = np.median(np.abs(y - y_smooth))
    prominence_threshold = max(5.0 * noise, np.finfo(float).eps)

    n_eff = 0.0
    for sign in (1.0, -1.0):
        _, properties = find_peaks(sign * y_smooth, prominence=prominence_threshold)
        for prominence in properties["prominences"]:
            n_eff += max(1.0, np.log(float(prominence) / tolerance))

    return max(1.0, float(n_eff))
