import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvergenceInfo:
    rhat: Optional[torch.Tensor] = None
    tau: Optional[torch.Tensor] = None   # shape (n_walkers, n_dim)
    ess: Optional[torch.Tensor] = None   # shape (n_dim,)
    mean_rel_change: Optional[torch.Tensor] = None
    std_rel_change: Optional[torch.Tensor] = None

    @property
    def max_rhat(self) -> Optional[float]:
        return torch.max(self.rhat).item() if self.rhat is not None else None

    @property
    def min_ess(self) -> Optional[float]:
        return torch.min(self.ess).item() if self.ess is not None else None

    @property
    def tau_cv(self) -> Optional[torch.Tensor]:
        if self.tau is None:
            return None
        if self.tau.shape[0] < 2:
            return torch.zeros(
                self.tau.shape[1],
                device=self.tau.device,
                dtype=self.tau.dtype
            )
        tau_mean = self.tau.mean(dim=0)
        tau_std = self.tau.std(dim=0, unbiased=True)
        eps = torch.tensor(1e-12, device=self.tau.device, dtype=self.tau.dtype)
        return tau_std / torch.maximum(tau_mean, eps)


def split_rhat(chain: torch.Tensor) -> torch.Tensor:
    """
    Compute split R-hat for each parameter.

    Parameters
    ----------
    chain : torch.Tensor
        Shape (n_samples, n_walkers, n_dim).

    Returns
    -------
    torch.Tensor
        Shape (n_dim,).
    """
    n_samples, n_walkers, n_dim = chain.shape
    half = n_samples // 2
    if half < 2:
        raise ValueError("Need at least 4 samples per walker for split R-hat.")

    chains = chain.permute(1, 0, 2)  # (n_walkers, n_samples, n_dim)
    chains = torch.cat([chains[:, :half], chains[:, -half:]], dim=0)

    _, n, _ = chains.shape
    chain_means = chains.mean(dim=1)
    chain_vars = chains.var(dim=1, unbiased=True)

    W = chain_vars.mean(dim=0)
    B = n * chain_means.var(dim=0, unbiased=True)
    var_hat = ((n - 1) / n) * W + B / n

    return torch.sqrt(var_hat / W)


def autocorrelation(x: torch.Tensor) -> torch.Tensor:
    """
    Compute autocorrelation along axis 0 using FFT.

    Parameters
    ----------
    x : torch.Tensor
        Shape (n_samples, ...).

    Returns
    -------
    torch.Tensor
        Shape (n_samples, ...).
    """
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if x.shape[0] < 2:
        return torch.ones_like(x)

    n = x.shape[0]
    x = x - x.mean(dim=0, keepdim=True)

    nfft = 1 << (2 * n - 1).bit_length()
    fx = torch.fft.rfft(x, n=nfft, dim=0)
    acov = torch.fft.irfft(fx * torch.conj(fx), n=nfft, dim=0)[:n]

    norm = torch.arange(n, 0, -1, device=x.device, dtype=x.dtype)
    norm = norm.view((n,) + (1,) * (x.ndim - 1))
    acov = acov / norm

    acf = torch.zeros_like(acov)
    var0 = acov[0]
    mask = var0 > 0
    acf[:, mask] = acov[:, mask] / var0[mask]
    return acf


def integrated_autocorr_time(
    x: torch.Tensor,
    max_lag: Optional[int] = None
) -> torch.Tensor:
    """
    Estimate integrated autocorrelation time along axis 0.

    Parameters
    ----------
    x : torch.Tensor
        Shape (n_samples, ...).
    max_lag : int, optional
        Maximum lag included.

    Returns
    -------
    torch.Tensor
        Shape x.shape[1:].
    """
    rho = autocorrelation(x)
    n = rho.shape[0]

    if max_lag is None:
        max_lag = n - 1
    max_lag = min(max_lag, n - 1)

    tau = torch.ones(rho.shape[1:], device=x.device, dtype=x.dtype)
    active = torch.ones_like(tau, dtype=torch.bool)

    for k in range(1, max_lag + 1):
        positive = rho[k] > 0
        mask = active & positive
        tau[mask] += 2.0 * rho[k][mask]
        active &= positive
        if not active.any():
            break

    return torch.clamp(tau, min=1.0)


def stability_metrics(
    chain: torch.Tensor,
    window_frac: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compare posterior mean and std between the last two windows.

    Parameters
    ----------
    chain : torch.Tensor
        Shape (n_samples, n_walkers, n_dim).
    window_frac : float
        Fraction of samples in each window (default 0.25).

    Returns
    -------
    mean_rel_change : torch.Tensor
    std_rel_change : torch.Tensor
    """
    n_samples, _, n_dim = chain.shape
    win = int(window_frac * n_samples)
    if win < 10 or 2 * win > n_samples:
        raise ValueError("Chain too short for stability check.")

    a = chain[-2 * win:-win].reshape(-1, n_dim)
    b = chain[-win:].reshape(-1, n_dim)

    mean_a = a.mean(dim=0)
    mean_b = b.mean(dim=0)
    std_a = a.std(dim=0, unbiased=True)
    std_b = b.std(dim=0, unbiased=True)

    eps = torch.tensor(1e-12, device=chain.device, dtype=chain.dtype)
    mean_rel_change = torch.abs(mean_b - mean_a) / torch.maximum(torch.abs(mean_a), eps)
    std_rel_change = torch.abs(std_b - std_a) / torch.maximum(std_a, eps)

    return mean_rel_change, std_rel_change
