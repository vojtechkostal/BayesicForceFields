import numpy as np
import pytest
import torch

from bff.bayes.fit import find_map, laplace_approximation
from bff.bayes.priors import Prior, Priors
from bff.bayes.utils import initialize_walkers
from bff.domain.specs import Specs


def _bounds(low: float, high: float, n: int = 1):
    return np.full(n, low), np.full(n, high)


def test_find_map_converges_on_a_quadratic_in_few_evaluations() -> None:
    center = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)

    result = find_map(
        lambda x: -torch.sum((x - center) ** 2),
        torch.zeros(3, dtype=torch.float64),
        bounds=_bounds(-10, 10, 3),
    )

    assert result.converged
    assert torch.allclose(result.theta, center, atol=1e-5)
    assert result.evaluations < 20
    assert result.value == pytest.approx(0.0, abs=1e-8)


def test_find_map_keeps_the_solution_inside_the_bounds() -> None:
    result = find_map(
        lambda x: -torch.sum((x - 5.0) ** 2),
        torch.zeros(1, dtype=torch.float64),
        bounds=_bounds(-1, 1),
    )

    assert result.theta.item() == pytest.approx(1.0)
    assert not result.converged  # the optimum sits on the bound


def test_find_map_backs_off_from_points_where_the_objective_fails() -> None:
    def objective(x):
        if x.item() > 2.0:
            return torch.tensor(torch.nan, dtype=torch.float64)
        return -((x - 1.9) ** 2).sum()

    result = find_map(
        objective,
        torch.zeros(1, dtype=torch.float64),
        bounds=_bounds(-10, 10),
    )

    assert result.theta.item() == pytest.approx(1.9, abs=1e-4)


def test_find_map_uses_other_starts_when_the_first_search_fails() -> None:
    def objective(x):  # two optima; the one near +4 is higher
        return torch.log(
            torch.exp(-((x - 4.0) ** 2)) + 0.3 * torch.exp(-((x + 4.0) ** 2))
        ).sum()

    result = find_map(
        objective,
        torch.tensor([-4.0], dtype=torch.float64),
        bounds=_bounds(-4, 8),
        starts=[torch.tensor([3.0], dtype=torch.float64)],
    )

    assert result.theta.item() == pytest.approx(4.0, abs=1e-3)


def test_laplace_covariance_is_finite_where_the_curvature_is_not_positive() -> None:
    theta = torch.zeros(2, dtype=torch.float64)

    cov = laplace_approximation(
        lambda x: -2.0 * x[0] ** 2 + 0.5 * x[1] ** 2, theta
    )

    assert cov[0, 0].item() == pytest.approx(0.25)
    assert cov[1, 1].item() == pytest.approx(1 / 1e-3)
    assert torch.isfinite(cov).all()


def test_initialize_walkers_supports_uniform_priors_and_constraints() -> None:
    priors = Priors([Prior("uniform", 0.0, 1.0), Prior("normal", 5.0, 0.1)])
    walkers = initialize_walkers(priors, 50)
    assert walkers.shape == (50, 2)
    assert ((walkers[:, 0] >= 0) & (walkers[:, 0] <= 1)).all()

    specs = Specs(
        {
            "bounds": {"sigma A": [0.5, 1.0]},
            "charge_constraints": [],
        }
    )

    constrained = initialize_walkers(priors, 20, specs)
    assert constrained.shape == (20, 2)
    assert (constrained[:, 0] > 0.5).all()
