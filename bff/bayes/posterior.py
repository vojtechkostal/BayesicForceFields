import torch
import numpy as np

from typing import List, Callable
from .priors import Priors, log_prior
from .utils import check_tensor


def log_posterior(
    theta: torch.Tensor,
    priors: Priors | List[torch.distributions.Distribution],
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    numpy_output: bool = True
) -> np.ndarray:

    """
    Computes the log-posterior = log-prior + log-likelihood.

    Parameters
    ----------
    theta : torch.Tensor
        Tensor of parameters for which to compute the log-posterior.
    priors : List[torch.distributions.Distribution]
        List of prior distributions for the parameters.
    log_likelihood_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that computes the log-likelihood given the parameters.
        It must have signature `log_likelihood_fn(theta: torch.Tensor) -> torch.Tensor`.
    device : str
        Device on which the computations should be performed (e.g., 'cuda:0' or 'cpu').
    numpy_output : bool, optional
        If True, returns the log-posterior as a NumPy array.
        If False, returns it as a PyTorch tensor. Defaults to True.

    Returns
    -------
    np.ndarray
        Log-posterior values for the given parameters.
    """

    theta = check_tensor(theta, device)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    if theta.isnan().any():
        log_prob = torch.full((theta.shape[0],), -1e10, device=device)
        if theta._grad_fn:
            log_prob = log_prob.requires_grad_()

    else:
        log_likelihood = log_likelihood_fn(theta)
        log_likelihood = torch.where(
            torch.isnan(log_likelihood), -torch.inf, log_likelihood)

        log_prob = log_prior(theta, priors) + log_likelihood

    if log_prob.ndim == 1:
        log_prob = log_prob.squeeze(0)
    if numpy_output:
        return log_prob.detach().cpu().numpy()
    return log_prob
