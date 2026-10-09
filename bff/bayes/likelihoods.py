"""Gaussian log likelihood of the surrogate predictions against references."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .kernels import gaussian_kernel

if TYPE_CHECKING:
    from .learning import LearningProblem


def _split_parameters_and_sigmas(
    theta: torch.Tensor,
    problem: LearningProblem,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit parameters and one scalar nuisance per QoI."""
    n_params = problem.n_params
    params = theta[:, :n_params]
    nuisance_free = theta[:, n_params:]

    sigma_columns = []
    j = 0
    for model in problem.models.values():
        if model.nuisance is None:
            sigma_columns.append(nuisance_free[:, j].exp())
            j += 1
        else:
            sigma_columns.append(
                torch.full(
                    (len(theta),),
                    float(model.nuisance),
                    device=theta.device,
                    dtype=theta.dtype,
                )
            )
    return params, torch.stack(sigma_columns, dim=1)


def _valid_parameter_mask(
    params: torch.Tensor,
    problem: LearningProblem,
) -> torch.Tensor:
    if problem.specs is None:
        return torch.ones(len(params), dtype=bool, device=params.device)
    return problem.specs.is_valid(params)


def loo_log_likelihood(
    theta: torch.Tensor, X: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Leave One Out Log Likelihood for Gaussian Process Regression.
    From Sundarajan & Keerthi (2001):
    "Predictive Approaches for Choosing Hyperparameters in Gaussian Processes",
    Equation 8.
    doi: 10.1162/08997660151134343

    Parameters
    ----------
    theta : torch.Tensor
        Hyperparameters of the Gaussian Process model.
        Shape: (n_samples, n_hyperparams).
    X : torch.Tensor
        Input data points.
        Shape: (n_samples, n_features).
    y : torch.Tensor
        Outputs.
        Shape: (n_samples, n_outputs).

    Returns
    -------
    torch.Tensor
        Leave One Out Log Likelihood for each hyperparameter set.
        Shape: (n_samples,).
    """

    theta = theta.exp()
    length, amplitude, noise = theta[:, :-2], theta[:, -2], theta[:, -1]
    n_samples, n_y = len(X), y.shape[1]
    identity = torch.eye(n_samples, dtype=theta.dtype, device=theta.device)

    Kdd = gaussian_kernel(X, X, length, amplitude) + noise[:, None, None] * identity
    Kdd = 0.5 * (Kdd + Kdd.transpose(1, 2))
    L, info = torch.linalg.cholesky_ex(Kdd)
    failed = info > 0
    if failed.any():
        # Hyperparameters whose covariance is not positive definite are
        # impossible: -inf likelihood, and a harmless matrix in their place.
        L, _ = torch.linalg.cholesky_ex(
            torch.where(failed[:, None, None], identity, Kdd)
        )
    Kdd_inv = torch.cholesky_inverse(L)
    Kdd_inv_diagonal = torch.diagonal(Kdd_inv, 0, dim1=1, dim2=2)
    log_Kdd_inv_ii = torch.log(Kdd_inv_diagonal)

    norm = torch.sqrt(Kdd_inv_diagonal).unsqueeze(1)
    term_1 = (Kdd_inv @ y).transpose(1, 2) / norm
    term_1 = 1 / (2 * n_samples) * torch.sum(term_1**2, dim=(1, 2))
    term_2 = n_y / (2 * n_samples) * torch.sum(log_Kdd_inv_ii, dim=1)
    term_3 = (n_y / 2) * math.log(2 * math.pi)

    return torch.where(failed, -torch.inf, -(term_1 - term_2 + term_3))


def gaussian_log_likelihood(
    theta: torch.Tensor,
    problem: LearningProblem,
) -> torch.Tensor:

    """Compute the gaussian log likelihood.

    Parameters
    ----------
    theta : torch.Tensor
        Parameters of the surrogate model, shape (n_samples, n_params + n_sigma).
    problem : LearningProblem
        Complete inference problem including surrogates, observations,
        and the optional parameter specification.

    Returns
    -------
    torch.Tensor
        Log likelihood for each sample in `theta`, shape (n_samples,).
    """

    contributions = gaussian_log_likelihood_by_qoi(theta, problem)
    return torch.stack(tuple(contributions.values())).sum(dim=0)


def gaussian_log_likelihood_by_qoi(
    theta: torch.Tensor,
    problem: LearningProblem,
) -> dict[str, torch.Tensor]:
    """Compute one Gaussian log-likelihood contribution per QoI."""
    params, sigmas = _split_parameters_and_sigmas(theta, problem)
    valid = _valid_parameter_mask(params, problem)

    contributions = {}
    for (qoi, model), sigma in zip(problem.models.items(), sigmas.T):
        # Every row is evaluated and the invalid ones masked afterwards, so
        # nothing depends on the values on the device.
        diff = problem.observations[qoi] - model.predict(params)
        mse = torch.mean(diff**2, dim=1)
        n_eff = float(model.n_eff)
        # The learned noise sigma and the accepted deviation (tolerance)
        # add up to the variance of each effective observation.
        variance = sigma**2 + float(model.tolerance) ** 2
        value = -0.5 * n_eff * mse / variance - 0.5 * n_eff * torch.log(variance)
        contributions[qoi] = torch.where(valid, value, -torch.inf)

    return contributions
