"""Log-posterior of parameter vectors under priors and a likelihood."""

from typing import Callable

import torch

from .priors import Priors


def log_posterior(
    theta: torch.Tensor,
    priors: Priors,
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Log-prior plus log-likelihood for each row of ``theta``.

    Rows containing NaN, and rows whose result is NaN, get ``-inf``. A single
    parameter vector returns a scalar tensor. Everything stays on the device
    of ``theta``; nothing is synchronized with the host.
    """
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    bad = theta.isnan().any(dim=1)
    theta = torch.where(bad.unsqueeze(1), torch.zeros_like(theta), theta)
    values = priors.log_prob(theta) + log_likelihood_fn(theta)
    values = torch.where(bad | values.isnan(), -torch.inf, values)
    return values.squeeze(0)
