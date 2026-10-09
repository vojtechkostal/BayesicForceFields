"""Gaussian kernel of the surrogates."""

import torch


def gaussian_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    """Gaussian (squared-exponential) kernel matrix.

    ``amplitude**2 * exp(-|x1 - x2|**2 / (2 lengthscales**2))``, with the
    length scale applied per input. Distances use explicit differences, so the
    kernel is differentiable to any order and exact for close points.

    Parameters
    ----------
    x1, x2 : torch.Tensor
        Inputs, shapes ``(..., n1, n_inputs)`` and ``(..., n2, n_inputs)``.
    lengthscales : torch.Tensor
        Shape ``(..., n_inputs)``, broadcast against the leading dimensions.
    amplitude : torch.Tensor
        Shape ``(...)``, broadcast against the leading dimensions.

    Returns
    -------
    torch.Tensor
        Shape ``(..., n1, n2)``.
    """
    scale = lengthscales.unsqueeze(-2).unsqueeze(-2)
    diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / scale
    sqdist = diff.pow(2).sum(dim=-1)
    return amplitude[..., None, None].pow(2) * torch.exp(-0.5 * sqdist)
