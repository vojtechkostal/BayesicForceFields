from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .kernels import gaussian_kernel

if TYPE_CHECKING:
    from .learning import InferenceProblem


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

    # Ensure that all the inputs are on the same device
    device = theta.device
    theta = theta.exp()
    length = theta[:, :-2][:, None].to(device)
    width = theta[:, -2][:, None, None].to(device)
    noise = theta[:, -1][:, None, None].to(device)

    # Check the dimensions of X and y
    n_batch = len(theta)
    n_samples = len(X)
    n_y = y.shape[1]
    X = X.unsqueeze(0).repeat(n_batch, 1, 1).to(device)

    identity = torch.eye(n_samples, device=device).expand(n_batch, n_samples, n_samples)

    # Compute the kernel matrix and its inverse
    Kdd = gaussian_kernel(X, X, length, width) + identity * noise
    Kdd_inv = torch.linalg.inv(Kdd)
    Kdd_inv_diagonal = torch.diagonal(Kdd_inv, 0, dim1=1, dim2=2)
    log_Kdd_inv_ii = torch.log(Kdd_inv_diagonal)

    # Compute the terms for the log likelihood
    norm = torch.sqrt(Kdd_inv_diagonal + 1e-9).unsqueeze(1)
    term_1 = (Kdd_inv @ y).transpose(1, 2) / norm
    term_1 = 1 / (2 * n_samples) * torch.sum(term_1**2, dim=(1, 2))

    term_2 = n_y / (2 * n_samples) * torch.sum(log_Kdd_inv_ii, dim=1)

    pi_tensor = torch.tensor(torch.pi, device=device)
    term_3 = (n_y / 2) * torch.log(2 * pi_tensor)

    return - (term_1 - term_2 + term_3)


def gaussian_log_likelihood(
    theta: torch.Tensor,
    problem: InferenceProblem,
) -> torch.Tensor:

    """Compute the gaussian log likelihood.

    Parameters
    ----------
    theta : torch.Tensor
        Parameters of the surrogate model, shape (n_samples, n_params + n_sigma).
    problem : InferenceProblem
        Complete inference problem including surrogates, observations,
        and the optional parameter constraint.

    Returns
    -------
    torch.Tensor
        Log likelihood for each sample in `theta`, shape (n_samples,).
    """

    # Assign device
    device = theta.device

    # Split the theta into parameters and nuisances
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
                    device=device,
                    dtype=theta.dtype,
                )
            )
    sigmas = torch.stack(sigma_columns, dim=1)

    # Check if the parameters are within the valid bounds
    if problem.constraint is not None:
        mask = problem.constraint(params)
    else:
        mask = torch.ones(len(theta), dtype=bool, device=device)
    params = params[mask]
    sigmas = sigmas[mask]

    # Compute the log-likelihod
    log_likelihood = torch.full((len(theta), ), -torch.inf, device=device)
    if mask.any():
        log_like_valid = torch.zeros(mask.sum(), device=device)
        for (qoi, model), sigma in zip(problem.models.items(), sigmas.T):
            y_trial = model.predict(params)
            N = model.observations
            diff = problem.observations[qoi] - y_trial
            ssq = torch.sum(diff**2, dim=1)
            log_like_valid += -0.5 * ssq / sigma**2 - N * torch.log(sigma)

        log_likelihood[mask] = log_like_valid

    return log_likelihood
