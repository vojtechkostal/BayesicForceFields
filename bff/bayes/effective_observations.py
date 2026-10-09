"""Effective number of independent observations in a reference QoI.

Neighbouring values of a curve (an RDF, a free-energy profile) are not
independent: a model that deviates at one bin deviates at its neighbours too.
The deviations are modelled as correlated along the curve, with a
squared-exponential correlation ``R_ij = exp(-(i - j)**2 / (2 l**2))`` whose
length ``l`` (in bins) is fitted to the reference curve. A Gaussian likelihood
of the mean squared deviation then has

    n_eff = (tr R)**2 / tr(R @ R)

degrees of freedom (Satterthwaite), about ``n_bins / (sqrt(pi) l)``: one per
stretch of the curve over which deviations are correlated. ``n_eff`` does not
depend on how finely the curve is binned. Scalar QoIs (and curves of fewer
than 3 values) count one observation per value.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def correlation_length(curve: np.ndarray) -> float:
    """Correlation length of a curve, in bins.

    It is the length scale of a squared-exponential Gaussian process fitted to
    the standardized curve by maximum marginal likelihood. A flat curve has no
    structure: its length is infinite, so it counts as one observation.
    """
    y = np.asarray(curve, dtype=float)
    if y.ndim != 1 or y.size < 3 or not np.all(np.isfinite(y)):
        raise ValueError("A curve must be one-dimensional, finite, and have 3+ values.")
    if np.ptp(y) <= 1e-12 * max(1.0, np.abs(y).max()):
        return float("inf")
    y = (y - y.mean()) / y.std()
    x = np.arange(y.size, dtype=float)
    squared_distances = (x[:, None] - x[None, :]) ** 2

    def negative_log_marginal_likelihood(log_params: np.ndarray) -> float:
        length, amplitude, noise = np.exp(log_params)
        K = amplitude**2 * np.exp(-0.5 * squared_distances / length**2)
        K[np.diag_indices_from(K)] += noise**2 + 1e-8
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L, y)
        return 0.5 * alpha @ alpha + np.log(np.diag(L)).sum()

    fits = [
        minimize(
            negative_log_marginal_likelihood,
            np.log([start, 1.0, 0.05]),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-6},
        )
        for start in (y.size / 50, y.size / 20, y.size / 5)
    ]
    best = min(fits, key=lambda fit: fit.fun)
    return float(np.clip(np.exp(best.x[0]), 0.5, y.size))


def curve_n_eff(curve: np.ndarray) -> float:
    """Effective number of independent values in one curve."""
    length = correlation_length(curve)
    x = np.arange(len(curve), dtype=float)
    R = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / length**2)
    return float(max(1.0, np.trace(R) ** 2 / np.sum(R * R)))


def effective_observations(y_ref: np.ndarray, n_curves: int) -> float:
    """Effective observations in ``y_ref``, which concatenates ``n_curves``
    equally long curves; values of curves shorter than 3 count individually."""
    curves = np.asarray(y_ref, dtype=float).reshape(n_curves, -1)
    if curves.shape[1] < 3:
        return float(curves.size)
    return float(sum(curve_n_eff(curve) for curve in curves))
