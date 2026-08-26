"""
Jensen's Inequality Bias Correction for Log-Observable Gaussian Processes.
===========================================================================
Derives heteroscedastic noise scaling via the Delta method to correct
finite-sampling bias when modeling logarithmic Quantities of Interest.
"""

from __future__ import annotations

import numpy as np


class JensenLogObservableCorrection:
    """
    Heteroscedastic Delta-method noise mapping and unbiased physical recovery.
    """

    @staticmethod
    def compute_log_noise_variance(
        observable_mean: np.ndarray,
        observable_var: np.ndarray,
        n_frames: int,
        autocorr_tau: float = 1.0,
    ) -> np.ndarray:
        """
        Compute heteroscedastic log-noise variance using the Delta method:
            sigma_ln^2 = ln(1 + sigma_MD^2 / (N_eff * O^2)) ~= sigma_MD^2 / (N_eff * O^2)
        """
        obs = np.maximum(np.asarray(observable_mean, dtype=float), 1e-6)
        var = np.maximum(np.asarray(observable_var, dtype=float), 1e-12)
        n_eff = max(1.0, float(n_frames) / (2.0 * autocorr_tau + 1.0))

        relative_variance = var / (n_eff * (obs ** 2))
        return np.log1p(relative_variance)

    @staticmethod
    def recover_unbiased_physical_expectation(
        mu_log: np.ndarray,
        sigma_log_sq: np.ndarray,
    ) -> np.ndarray:
        """
        Recover the unbiased physical expectation E[O(theta)] from log-GP predictions:
            E[O(theta)] = exp(mu_log(theta) + 0.5 * sigma_log_sq(theta))
        """
        mu = np.asarray(mu_log, dtype=float)
        var = np.maximum(np.asarray(sigma_log_sq, dtype=float), 0.0)
        return np.exp(mu + 0.5 * var)
