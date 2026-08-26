"""
Functional PCA (FPCA) SVD Spatial Observable Surrogate.
========================================================
Compresses continuous spatial quantities of interest (such as radial distribution
functions g(r) or spatial angular distributions) into M orthogonal spatial modes
fitted with independent Gaussian Processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C


@dataclass
class FPCASurrogateResult:
    mean_curve: np.ndarray
    basis_modes: np.ndarray  # Shape: (M, n_bins)
    explained_variance_ratio: np.ndarray  # Shape: (M,)
    singular_values: np.ndarray


class FPCASurrogate:
    """Functional Principal Component Analysis spatial basis emulator."""

    def __init__(self, n_components: int = 5, nu: float = 2.5):
        self.n_components = n_components
        self.nu = nu
        self.mean_curve: np.ndarray | None = None
        self.basis_modes: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.models: list[GaussianProcessRegressor] = []

    def fit(self, X: np.ndarray, Y_curves: np.ndarray) -> FPCASurrogate:
        """
        Fit the FPCA basis and independent GPs for each spatial mode.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)
            Force field parameter samples.
        Y_curves : np.ndarray, shape (N, n_bins)
            Observed continuous spatial curves (e.g. g(r)).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y_curves, dtype=float)
        N, n_bins = Y.shape

        # 1. Compute empirical mean curve
        self.mean_curve = np.mean(Y, axis=0)
        centered_Y = Y - self.mean_curve

        # 2. Singular Value Decomposition (SVD)
        # centered_Y = U * S * V^T
        U, S, Vt = np.linalg.svd(centered_Y, full_matrices=False)
        M = min(self.n_components, len(S))
        self.basis_modes = Vt[:M, :]  # (M, n_bins)

        # Variance explained
        total_var = np.sum(S ** 2)
        self.explained_variance_ratio_ = (S[:M] ** 2) / total_var

        # Projection coefficients: C = centered_Y @ basis_modes.T  -> Shape (N, M)
        C_coeffs = centered_Y @ self.basis_modes.T

        # 3. Fit independent Gaussian Process for each component
        self.models = []
        for j in range(M):
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), nu=self.nu) + WhiteKernel(noise_level=1e-3)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=42 + j)
            gp.fit(X, C_coeffs[:, j])
            self.models.append(gp)

        return self

    def predict(self, X_new: np.ndarray, return_std: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Predict full continuous spatial curves for new parameters.

        Parameters
        ----------
        X_new : np.ndarray, shape (N_new, D)

        Returns
        -------
        Y_pred : np.ndarray, shape (N_new, n_bins)
        """
        X_new = np.asarray(X_new, dtype=float)
        N_new = X_new.shape[0]
        M = len(self.models)

        c_preds = np.zeros((N_new, M))
        c_stds = np.zeros((N_new, M))

        for j, gp in enumerate(self.models):
            mu, std = gp.predict(X_new, return_std=True)
            c_preds[:, j] = mu
            c_stds[:, j] = std

        # Reconstruct spatial curves: Y = mean + c_preds @ basis_modes
        Y_pred = self.mean_curve + (c_preds @ self.basis_modes)

        if not return_std:
            return Y_pred

        # Propagate epistemic variance to spatial curve bins: Var(Y(r)) = sum_j Var(c_j) * phi_j(r)^2
        spatial_variance = (c_stds ** 2) @ (self.basis_modes ** 2)
        spatial_std = np.sqrt(spatial_variance)
        return Y_pred, spatial_std
