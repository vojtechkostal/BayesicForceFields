import torch
import numpy as np
from torch.distributions import Normal, Uniform


def define_param_priors(
    param_bounds: dict[list], QoI: dict, dist_type: str
) -> dict[list]:
    """
    Define priors for the model parameters and nuisance parameters.

    Parameters
    ----------
    param_bounds : dict[list]
        Dict of tuples defining the bounds for each parameter.
    QoI : dict
        Dictionary of quantities of interest (QoI).
    dist_type : str
        Type of distribution for the priors, either 'normal' or 'uniform'.

    Returns
    -------
    dict
        Dict of prior distributions for the parameters and nuisance parameters.
    """

    if dist_type == 'normal':
        param_priors = {
            param: Normal(np.mean(bound), 1 / 5 * np.diff(bound)[0])
            for param, bound in param_bounds.items()
        }
    elif dist_type == 'uniform':
        param_priors = {
            param: Uniform(bound[0], bound[1], validate_args=False)
            for param, bound in param_bounds.items()
        }
    else:
        raise ValueError(
            f'Unknown prior type "{dist_type}". '
            'Options are "normal" or "uniform".'
        )

    nuisance_priors = {f'nuisance {q}': Normal(-2, 2) for q in QoI}

    return param_priors | nuisance_priors


def define_hyper_priors(n_params) -> list[Normal]:
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

    lengths_priors = {f'length {i}': Normal(-2, 2) for i in range(n_params)}
    width_prior = {'width': Normal(-2, 2)}
    sigma_prior = {'sigma': Normal(-2, 3)}
    return lengths_priors | width_prior | sigma_prior


def log_prior(
    theta: torch.Tensor, priors: list[torch.distributions.Distribution]
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
