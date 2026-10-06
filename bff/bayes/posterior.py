"""Log-posterior of parameter vectors under priors and a likelihood."""

from typing import Callable

import torch

from .priors import Priors, log_prior
from .utils import check_tensor


def log_posterior(
    theta: torch.Tensor,
    priors: Priors | list[torch.distributions.Distribution],
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str,
) -> torch.Tensor:
    """Log-prior plus log-likelihood for each row of ``theta``.

    Rows containing NaN, and rows whose likelihood is NaN, get ``-inf``. A
    single parameter vector returns a scalar tensor.
    """
    theta = check_tensor(theta, device)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    valid = ~theta.isnan().any(dim=1)
    log_prob = torch.full(
        (theta.shape[0],), -torch.inf, device=theta.device, dtype=theta.dtype
    )
    if valid.any():
        values = log_prior(theta[valid], priors) + log_likelihood_fn(theta[valid])
        log_prob[valid] = torch.where(torch.isnan(values), -torch.inf, values)
    return log_prob.squeeze(0)
