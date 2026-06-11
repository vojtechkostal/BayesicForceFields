import numpy as np
import pytest

from bff.bayes.effective_observations import estimate_curve_n_eff


def test_estimate_curve_n_eff_returns_at_least_one_for_flat_curve() -> None:
    curve = np.ones(25)

    assert estimate_curve_n_eff(curve, tolerance=0.1) == 1.0


def test_estimate_curve_n_eff_counts_resolved_peak() -> None:
    x = np.linspace(-3.0, 3.0, 101)
    curve = np.exp(-x**2)

    assert estimate_curve_n_eff(curve, tolerance=0.1) > 1.0


def test_estimate_curve_n_eff_validates_inputs() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_curve_n_eff(np.ones((2, 6)), tolerance=0.1)

    with pytest.raises(ValueError, match="at least 11"):
        estimate_curve_n_eff(np.ones(10), tolerance=0.1)

    with pytest.raises(ValueError, match="tolerance"):
        estimate_curve_n_eff(np.ones(25), tolerance=0.0)
