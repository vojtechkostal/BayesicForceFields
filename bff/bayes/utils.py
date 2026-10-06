from pathlib import Path
from typing import Callable, Union

import torch

PathLike = Union[str, Path]


def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute the symmetric mean absolute percentage error."""

    device = y_true.device if isinstance(y_true, torch.Tensor) else 'cpu'

    y_true = check_tensor(y_true, device=device)
    y_pred = check_tensor(y_pred, device=device)
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
    y_true_abs = torch.sum(torch.abs(y_true), dim=1)
    y_pred_abs = torch.sum(torch.abs(y_pred), dim=1)
    norm = y_true_abs + y_pred_abs

    ratios = torch.where(norm > 0, abs_diff / norm, torch.zeros_like(norm))
    return float(torch.mean(ratios).item())


def initialize_walkers(
    priors: list[torch.distributions.Distribution],
    n_walkers: int,
    constraint: Callable | None = None,
    max_attempts: int = 1000,
) -> torch.Tensor:
    """Draw ``n_walkers`` starting points from the priors.

    With a ``constraint``, draws are made in batches and only points whose
    first ``constraint.n_params`` entries satisfy it are kept.
    """

    def draw(n: int) -> torch.Tensor:
        return torch.stack([prior.sample((n,)) for prior in priors], dim=1).float()

    if constraint is None:
        return draw(n_walkers)
    accepted: list[torch.Tensor] = []
    n_accepted = 0
    for _ in range(max_attempts):
        trial = draw(2 * (n_walkers - n_accepted))
        valid = trial[constraint(trial[:, : constraint.n_params])]
        accepted.append(valid)
        n_accepted += len(valid)
        if n_accepted >= n_walkers:
            return torch.cat(accepted)[:n_walkers]
    raise RuntimeError(
        "Failed to initialize constrained walkers from the priors. "
        f"Accepted {n_accepted}/{n_walkers} samples after {max_attempts} batches."
    )


def check_tensor(
    x: Union[torch.Tensor, float, int, list, tuple],
    device: str,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert input to a torch tensor on the specified device."""
    if not isinstance(x, torch.Tensor):
        return torch.as_tensor(x, device=device, dtype=dtype)
    return x.to(device, dtype=dtype)


def check_device(device: str) -> None:
    """Check if the specified device is available."""
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
    elif device.startswith("mps"):
        if not torch.mps.is_available():
            raise RuntimeError("MPS is not available.")


@torch.no_grad()
def nearest_positive_definite(A: torch.Tensor) -> torch.Tensor:
    """Find the nearest positive definite matrix to A."""
    # Symmetrize
    A_sym = (A + A.T) / 2

    # Check if already PD
    if torch.all(torch.linalg.eigvalsh(A_sym) > 0):
        return A_sym

    # Eigen-decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)

    # Shift eigenvalues minimally
    min_eig = eigenvalues.min()
    eps = 1e-8  # small positive shift

    if min_eig < eps:
        eigenvalues = eigenvalues - min_eig + eps

    A_pd = (eigenvectors @ torch.diag(eigenvalues)) @ eigenvectors.T
    return A_pd


# Global toggle for manual squared distance
_MANUAL_MODE = False


class enable_manual_dist:
    """Enable Hessian-safe manual pairwise distances within a context."""
    def __enter__(self) -> None:
        global _MANUAL_MODE
        self._prev = _MANUAL_MODE
        _MANUAL_MODE = True

    def __exit__(self, *args) -> None:
        global _MANUAL_MODE
        _MANUAL_MODE = self._prev


def with_manual_sqdist_flag(fn: Callable) -> Callable:
    """Inject the global manual-distance flag into kernel call sites."""
    def wrapper(*args, manual_sqdist=False, **kwargs):
        return fn(*args, manual_sqdist=manual_sqdist or _MANUAL_MODE, **kwargs)
    return wrapper


