import importlib

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import beta, norm

pd = pytest.importorskip("pandas")
module = importlib.import_module("examples.calvados.prepare_priors")
StickinessPrior = module.StickinessPrior


def test_tophat_support_and_nonfinite_inputs():
    prior = StickinessPrior("uniform", ("lambda A", "lambda B"))
    values = prior.log_prob(
        torch.tensor(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [-0.01, 0.5],
                [0.5, 1.01],
                [float("nan"), 0.5],
            ]
        )
    )
    torch.testing.assert_close(values[:2], torch.zeros(2))
    assert torch.isneginf(values[2:]).all()


def test_joint_kde_matches_gaussian_mixture_and_has_correlations():
    centers = np.array([[0.2, 0.2], [0.8, 0.8]])
    prior = StickinessPrior("kde", ("lambda A", "lambda B"), centers=centers)
    points = torch.tensor(
        [[0.2, 0.2], [0.2, 0.8]], dtype=torch.float64, requires_grad=True
    )
    actual = prior.log_prob(points)
    expected = (
        logsumexp(
            norm.logpdf(points.detach().numpy()[:, None, :], centers, 0.05).sum(axis=2),
            axis=1,
        )
        - np.log(2)
        - prior.log_normalizer
    )
    np.testing.assert_allclose(actual.detach().numpy(), expected)
    assert actual[0] > actual[1]
    actual.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert torch.isneginf(prior.log_prob(torch.tensor([-0.1, 0.2]))).all()


def test_bounded_kde_integrates_to_one():
    prior = StickinessPrior(
        "kde", ("lambda A",), centers=np.array([[0.0], [0.5], [1.0]])
    )

    def density(value):
        return prior.log_prob(torch.tensor([value], dtype=torch.float64)).exp().item()

    assert quad(density, 0, 1)[0] == pytest.approx(1, abs=1e-8)


def test_independent_beta_matches_moments_and_product_density():
    scales = np.array([[0.0, 0.1], [0.3, 0.4], [0.7, 0.6], [1.0, 0.9]])
    alpha, beta_shape = module.beta_parameters(scales)
    np.testing.assert_allclose(alpha / (alpha + beta_shape), scales.mean(axis=0))
    np.testing.assert_allclose(
        alpha * beta_shape / ((alpha + beta_shape) ** 2 * (alpha + beta_shape + 1)),
        scales.var(axis=0),
    )
    prior = StickinessPrior(
        "beta", ("lambda A", "lambda B"), alpha=alpha, beta=beta_shape
    )
    points = torch.tensor([[0.2, 0.7], [0.8, 0.4]], dtype=torch.float64)
    expected = beta.logpdf(points, alpha, beta_shape).sum(axis=1)
    np.testing.assert_allclose(prior.log_prob(points), expected)
    restored = StickinessPrior(**prior.to_dict())
    torch.testing.assert_close(prior.log_prob(points), restored.log_prob(points))
    assert torch.isneginf(prior.log_prob(torch.tensor([0.0, 0.5]))).all()


def test_scales_are_reordered_by_residue_id():
    scales = pd.DataFrame(
        {"scale1": [0.8, 0.1], "scale2": [0.9, 0.2]}, index=["R", "A"]
    )
    residues = pd.DataFrame({"three": ["ARG", "ALA"], "one": ["R", "A"]})
    actual = module.ordered_scales(scales, residues, ("lambda ALA", "lambda ARG"))
    np.testing.assert_allclose(actual, [[0.1, 0.8], [0.2, 0.9]])


def test_prior_can_be_used_by_bff_log_posterior():
    from bff.bayes.posterior import log_posterior

    prior = StickinessPrior("beta", ("lambda A",), alpha=[2.0], beta=[3.0])
    points = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    result = log_posterior(points, prior, lambda values: -(values[:, 0] ** 2))
    torch.testing.assert_close(result, prior.log_prob(points) - points[:, 0] ** 2)
