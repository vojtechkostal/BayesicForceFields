from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bff.bayes.likelihoods import (
    gaussian_log_likelihood,
    gaussian_log_likelihood_by_qoi,
    loo_log_likelihood,
)


class FakeModel:
    def __init__(
        self,
        prediction: torch.Tensor,
        *,
        nuisance: float | None = None,
        n_eff: float = 2.0,
    ) -> None:
        self.prediction = prediction
        self.nuisance = nuisance
        self.n_eff = n_eff

    def predict(self, params: torch.Tensor) -> torch.Tensor:
        return self.prediction[: len(params)].to(params.device)


def test_loo_log_likelihood_returns_one_value_per_theta() -> None:
    theta = torch.tensor([[0.0, 0.0, -2.0], [0.1, 0.2, -1.5]])
    X = torch.tensor([[0.0], [1.0], [2.0]])
    y = torch.tensor([[0.0], [1.0], [0.0]])

    out = loo_log_likelihood(theta, X, y)

    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    individual = torch.cat([
        loo_log_likelihood(row[None, :], X, y)
        for row in theta
    ])
    assert torch.allclose(out, individual)


def test_gaussian_log_likelihood_uses_free_and_fixed_nuisance() -> None:
    problem = SimpleNamespace(
        n_params=1,
        constraint=None,
        observations={
            "free": torch.tensor([1.0, 2.0]),
            "fixed": torch.tensor([0.0, 1.0]),
        },
        models={
            "free": FakeModel(torch.tensor([[1.0, 1.5], [1.0, 2.0]]), nuisance=None),
            "fixed": FakeModel(torch.tensor([[0.0, 1.0], [1.0, 1.0]]), nuisance=0.5),
        },
    )
    theta = torch.tensor([[0.1, np.log(0.25)], [0.2, np.log(1.0)]], dtype=torch.float32)

    out = gaussian_log_likelihood(theta, problem)
    contributions = gaussian_log_likelihood_by_qoi(theta, problem)

    assert out.shape == (2,)
    assert torch.isfinite(out).all()
    assert out[0] != out[1]
    assert set(contributions) == {"free", "fixed"}
    assert torch.allclose(out, contributions["free"] + contributions["fixed"])


def test_gaussian_log_likelihood_masks_invalid_parameters() -> None:
    class Constraint:
        def __call__(self, params: torch.Tensor) -> torch.Tensor:
            return params[:, 0] >= 0.0

    problem = SimpleNamespace(
        n_params=1,
        constraint=Constraint(),
        observations={"qoi": torch.tensor([0.0])},
        models={"qoi": FakeModel(torch.tensor([[0.0], [0.0]]), nuisance=1.0)},
    )
    theta = torch.tensor([[0.5], [-0.5]])

    out = gaussian_log_likelihood(theta, problem)

    assert torch.isfinite(out[0])
    assert torch.isneginf(out[1])


@pytest.mark.parametrize("nuisance", [None, 0.5])
def test_gaussian_log_likelihood_scales_complete_term_by_n_eff(
    nuisance: float | None,
) -> None:
    sigma = 0.5
    problem = SimpleNamespace(
        n_params=1,
        constraint=None,
        observations={"qoi": torch.tensor([1.0, 3.0])},
        models={
            "qoi": FakeModel(
                torch.tensor([[0.0, 1.0]]),
                nuisance=nuisance,
                n_eff=4.0,
            )
        },
    )
    theta = torch.tensor(
        [[0.0, np.log(sigma)]] if nuisance is None else [[0.0]],
        dtype=torch.float32,
    )

    result = gaussian_log_likelihood(theta, problem)

    mse = ((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2.0
    expected = -0.5 * 4.0 * mse / sigma**2 - 4.0 * np.log(sigma)
    assert result.item() == pytest.approx(expected)


def test_gaussian_log_likelihood_is_invariant_to_duplicated_bins() -> None:
    theta = torch.tensor([[0.0]])
    base = SimpleNamespace(
        n_params=1,
        constraint=None,
        observations={"qoi": torch.tensor([1.0, 3.0])},
        models={
            "qoi": FakeModel(
                torch.tensor([[0.0, 1.0]]),
                nuisance=0.5,
                n_eff=3.0,
            )
        },
    )
    duplicated = SimpleNamespace(
        n_params=1,
        constraint=None,
        observations={"qoi": torch.tensor([1.0, 3.0, 1.0, 3.0])},
        models={
            "qoi": FakeModel(
                torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
                nuisance=0.5,
                n_eff=3.0,
            )
        },
    )

    assert torch.allclose(
        gaussian_log_likelihood(theta, base),
        gaussian_log_likelihood(theta, duplicated),
    )
