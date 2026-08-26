"""
PlausibilityGateV2: Epistemic Uncertainty-Penalized Plausibility Gating.
========================================================================
Resolves the 'ignorance paradox' in classical Mahalanobis distance gating
by strictly penalizing unexplored parameter domains where GP variance diverges.
"""

from __future__ import annotations

import numpy as np


class PlausibilityGateV2:
    """
    Plausibility Gating with Epistemic Uncertainty Penalty.

    Standard Mahalanobis gating:
        score = [mu(theta) - mu0]^2 / [sigma_GP(theta)^2 + sigma0^2] <= tau^2
    suffers from the ignorance paradox: as theta moves far outside the domain,
    sigma_GP -> inf, driving the ratio to 0 and spuriously accepting unphysical points.

    PlausibilityGateV2 formulates:
        score = |mu(theta) - mu0| + kappa * sigma_GP(theta) <= tau * sigma0
    guaranteeing that unexplored regions are strictly penalized and excluded.
    """

    def __init__(self, target_mean: float | np.ndarray, target_std: float | np.ndarray, kappa: float = 2.0, tau: float = 3.0):
        self.target_mean = np.asarray(target_mean, dtype=float)
        self.target_std = np.asarray(target_std, dtype=float)
        self.kappa = kappa
        self.tau = tau

    def evaluate(self, mu: np.ndarray, sigma_gp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate plausibility score and boolean acceptance mask.

        Parameters
        ----------
        mu : np.ndarray, shape (N,) or (N, D_qoi)
            Gaussian process mean prediction.
        sigma_gp : np.ndarray, shape (N,) or (N, D_qoi)
            Gaussian process epistemic standard deviation.

        Returns
        -------
        scores : np.ndarray, shape (N,)
            Normalized plausibility score (lower is more plausible).
        is_plausible : np.ndarray, shape (N,) of bool
            Boolean mask of plausible candidates.
        """
        mu = np.asarray(mu, dtype=float)
        sigma_gp = np.asarray(sigma_gp, dtype=float)

        delta = np.abs(mu - self.target_mean)
        # Conservative upper bound on observable deviation
        conservative_bound = delta + self.kappa * sigma_gp
        threshold = self.tau * self.target_std

        if conservative_bound.ndim == 1:
            scores = conservative_bound / np.maximum(threshold, 1e-12)
            is_plausible = scores <= 1.0
        else:
            # Multi-QoI norm
            normalized_scores = conservative_bound / np.maximum(threshold, 1e-12)
            scores = np.max(normalized_scores, axis=1)
            is_plausible = scores <= 1.0

        return scores, is_plausible
