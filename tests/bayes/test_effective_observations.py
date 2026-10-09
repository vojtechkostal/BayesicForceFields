import numpy as np
import pytest

from bff.bayes.effective_observations import (
    correlation_length,
    curve_n_eff,
    effective_observations,
)


def _smooth_curve(n_bins: int, length: float, seed: int = 0) -> np.ndarray:
    """A random curve with squared-exponential correlation ``length`` (bins)."""
    x = np.arange(n_bins)
    R = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / length**2)
    L = np.linalg.cholesky(R + 1e-8 * np.eye(n_bins))
    return L @ np.random.default_rng(seed).standard_normal(n_bins)


def test_correlation_length_recovers_the_length_of_a_smooth_curve() -> None:
    curve = _smooth_curve(200, 8.0)
    assert correlation_length(curve) == pytest.approx(8.0, rel=0.3)
    # About one observation per sqrt(pi) correlation lengths.
    assert curve_n_eff(curve) == pytest.approx(200 / (np.sqrt(np.pi) * 8.0), rel=0.3)


def test_n_eff_does_not_depend_on_the_binning() -> None:
    r = np.linspace(0.0, 10.0, 400)
    rdf = 1.0 + np.exp(-(r - 3.0) ** 2 / 0.1) - 0.3 * np.exp(-(r - 5.0) ** 2 / 0.3)
    assert curve_n_eff(rdf) == pytest.approx(curve_n_eff(rdf[::2]), rel=0.15)


def test_scalars_count_individually_and_flat_curves_once() -> None:
    assert effective_observations(np.array([4.7, 1.2, 0.3]), n_curves=3) == 3.0
    assert effective_observations(np.ones(50), n_curves=1) == 1.0
    two_curves = np.concatenate([_smooth_curve(100, 5.0), _smooth_curve(100, 5.0, 1)])
    assert effective_observations(two_curves, n_curves=2) == pytest.approx(
        curve_n_eff(two_curves[:100]) + curve_n_eff(two_curves[100:])
    )
