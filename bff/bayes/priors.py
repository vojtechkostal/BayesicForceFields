import torch
import numpy as np
from torch.distributions import Normal, Uniform

from typing import List, Union


ArrayLike = Union[np.ndarray, torch.Tensor]


def define_param_priors(
    bounds: ArrayLike,
    dist_type: str = "normal",
    n_nuisance: int = 0
) -> List[torch.distributions.Distribution]:

    if dist_type == "normal":
        means = np.mean(bounds, axis=1).squeeze()
        widths = 1 / 5 * np.diff(bounds, axis=1).squeeze()
        param_priors = [
            Normal(mean, width) for mean, width in zip(means, widths)
        ]

    elif dist_type == "uniform":
        lowers = bounds[:, 0].squeeze()
        uppers = bounds[:, 1].squeeze()
        param_priors = [
            Uniform(lower, upper, validate_args=False)
            for lower, upper in zip(lowers, uppers)
        ]

    else:
        raise ValueError(
            f'Unknown prior type "{dist_type}". '
            'Options are "normal" or "uniform".'
        )

    nuisance_priors = [Normal(-2, 2) for _ in range(n_nuisance)]

    return param_priors + nuisance_priors


def define_hyper_priors(n_params: int) -> List[Normal]:
    """
    Define Gaussian priors for model parameters:
    - `lengths` (per parameter): N(-2, 2)
    - `width`: N(-2, 2)
    - `noise`: N(-2, 3)

    Parameters
    ----------
        n_params : int
            Number of model parameters (length scales).

    Returns
    -------
        List[Normal]
            List of Normal distributions representing the priors.
    """

    length_scale_priors = [Normal(-2, 2) for _ in range(n_params)]
    width_prior = Normal(-2, 2)
    noise_prior = Normal(-2, 3)

    return length_scale_priors + [width_prior, noise_prior]


def log_prior(
    theta: torch.Tensor, priors: List[torch.distributions.Distribution]
) -> torch.Tensor:
    """
    Compute the log prior probabilities for the parameters.

    Parameters
    ----------
    theta : torch.Tensor
        Tensor of shape (n_samples, n_params) containing parameter values.
    priors : list of torch.distributions.Distribution
        List of prior distributions for each parameter.

    Returns
    -------
    torch.Tensor
        Log prior probabilities for each parameter set.
    """

    device = theta.device
    log_probabilities = [
        p.log_prob(theta[:, i]).to(device) for i, p in enumerate(priors)
    ]
    return torch.stack(log_probabilities, dim=1).sum(dim=1)
