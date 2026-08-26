"""
Unit tests for Functional PCA, PlausibilityGateV2, and Jensen bias correction.
"""

import numpy as np
import pytest

from bff.bayes.fpca_surrogate import FPCASurrogate
from bff.bayes.plausibility_v2 import PlausibilityGateV2
from bff.bayes.jensen_correction import JensenLogObservableCorrection


def test_fpca_surrogate():
    np.random.seed(42)
    N, D, n_bins = 25, 3, 50
    X = np.random.uniform(-1, 1, size=(N, D))
    r = np.linspace(0.1, 1.0, n_bins)
    
    # Synthetic curves with 2 dominant modes
    Y = np.zeros((N, n_bins))
    for i in range(N):
        Y[i] = (
            X[i, 0] * np.exp(-((r - 0.3)**2) / 0.01) +
            X[i, 1] * np.sin(10 * r) +
            np.random.normal(0, 0.01, n_bins)
        )

    surrogate = FPCASurrogate(n_components=3)
    surrogate.fit(X, Y)

    assert surrogate.basis_modes.shape == (3, n_bins)
    assert len(surrogate.models) == 3
    assert np.sum(surrogate.explained_variance_ratio_) > 0.80

    # Predict
    X_test = np.random.uniform(-1, 1, size=(5, D))
    Y_pred, Y_std = surrogate.predict(X_test, return_std=True)

    assert Y_pred.shape == (5, n_bins)
    assert Y_std.shape == (5, n_bins)
    assert np.all(Y_std >= 0.0)


def test_plausibility_gate_v2():
    target_mu = 3.92
    target_sigma = 0.08
    gate = PlausibilityGateV2(target_mean=target_mu, target_std=target_sigma, kappa=2.0, tau=3.0)

    # Point 1: At optimum with low uncertainty -> Highly plausible
    score1, pass1 = gate.evaluate(np.array([3.92]), np.array([0.01]))
    assert bool(pass1[0]) is True
    assert score1[0] < 1.0

    # Point 2: Distant boundary point with huge uncertainty -> Must be rejected (ignorance penalty)
    score2, pass2 = gate.evaluate(np.array([3.92]), np.array([2.50]))
    assert bool(pass2[0]) is False
    assert score2[0] > 1.0


def test_jensen_log_observable_correction():
    obs_mean = np.array([4.0, 2.5])
    obs_var = np.array([0.16, 0.09])
    n_frames = 1000

    log_var = JensenLogObservableCorrection.compute_log_noise_variance(
        obs_mean, obs_var, n_frames=n_frames, autocorr_tau=2.0
    )
    assert log_var.shape == (2,)
    assert np.all(log_var > 0.0)

    # Unbiased expectation recovery
    recovered = JensenLogObservableCorrection.recover_unbiased_physical_expectation(
        mu_log=np.log(obs_mean), sigma_log_sq=log_var
    )
    assert recovered.shape == (2,)
    assert np.all(recovered >= obs_mean)
