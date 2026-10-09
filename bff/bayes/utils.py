"""Tensor checks, walker initialization, and numerical helpers."""

from pathlib import Path
from typing import Union

import numpy as np
import torch

PathLike = Union[str, Path]


def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute the symmetric mean absolute percentage error."""
    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true).to(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(-1)
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"SMAPE inputs must have matching shapes, got "
            f"{tuple(y_true.shape)} and {tuple(y_pred.shape)}."
        )

    abs_diff = torch.sum(torch.abs(y_true - y_pred), dim=1)
    norm = torch.sum(torch.abs(y_true), dim=1) + torch.sum(torch.abs(y_pred), dim=1)
    ratios = torch.where(norm > 0, abs_diff / norm, torch.zeros_like(norm))
    return float(torch.mean(ratios).item())


def evenly_spaced_indices(n: int, max_n: int | None) -> np.ndarray:
    """Indices of at most ``max_n`` evenly spaced items out of ``n``."""
    if max_n is None or max_n < 0 or n <= max_n:
        return np.arange(n)
    return np.linspace(0, n - 1, max_n, dtype=int)


def initialize_walkers(
    priors,
    n_walkers: int,
    specs=None,
    max_attempts: int = 1000,
) -> torch.Tensor:
    """Draw ``n_walkers`` starting points from the ``priors``.

    With ``specs``, only points whose sampled parameters keep every parameter
    within its bounds are kept.
    """

    def draw(n: int) -> torch.Tensor:
        columns = [prior.distribution.sample((n,)) for prior in priors]
        return torch.stack(columns, dim=1).float()

    if specs is None:
        return draw(n_walkers)
    n_explicit = len(specs.explicit_names)
    accepted: list[torch.Tensor] = []
    n_accepted = 0
    for _ in range(max_attempts):
        trial = draw(2 * (n_walkers - n_accepted))
        valid = trial[specs.is_valid(trial[:, :n_explicit])]
        accepted.append(valid)
        n_accepted += len(valid)
        if n_accepted >= n_walkers:
            return torch.cat(accepted)[:n_walkers]
    raise RuntimeError(
        "Failed to initialize constrained walkers from the priors. "
        f"Accepted {n_accepted}/{n_walkers} samples after {max_attempts} batches."
    )


def resolve_device(device: str) -> str:
    """The device a stage works on, decided once: ``auto`` is ``cuda`` when
    available and ``cpu`` otherwise; an explicit unavailable device raises."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if device.startswith("mps") and not torch.mps.is_available():
        raise RuntimeError("MPS is not available.")
    return device
