import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConvergenceInfo:
    rhat: Optional[torch.Tensor] = None
    tau: Optional[torch.Tensor] = None   # shape (n_walkers, n_dim)
    ess: Optional[torch.Tensor] = None   # shape (n_dim,)

    @property
    def max_rhat(self) -> Optional[float]:
        return torch.max(self.rhat).item() if self.rhat is not None else None

    @property
    def min_ess(self) -> Optional[float]:
        return torch.min(self.ess).item() if self.ess is not None else None


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


def rank_normalize(values: torch.Tensor) -> torch.Tensor:
    """Rank-normalize pooled samples along the first dimension."""
    flat = values.reshape(values.shape[0], -1)
    normalized = torch.empty_like(flat)
    n = flat.shape[0]
    denom = float(n) + 0.25
    sqrt_two = torch.sqrt(torch.tensor(2.0, device=values.device))

    for i in range(flat.shape[1]):
        order = torch.argsort(flat[:, i], stable=True)
        ranks = torch.empty(n, device=values.device, dtype=values.dtype)
        ranks[order] = torch.arange(
            1,
            n + 1,
            device=values.device,
            dtype=values.dtype,
        )
        u = (ranks - 0.375) / denom
        u = torch.clamp(u, 1e-12, 1 - 1e-12)
        normalized[:, i] = sqrt_two * torch.erfinv(2 * u - 1)

    return normalized.reshape(values.shape)


def rank_normalized_split_rhat(chain: torch.Tensor) -> torch.Tensor:
    """Compute rank-normalized split R-hat with folded refinement."""
    pooled = chain.reshape(-1, chain.shape[-1])
    ranked = rank_normalize(pooled).reshape(chain.shape)
    rhat_rank = split_rhat(ranked)

    pooled_median = pooled.median(dim=0).values
    folded = torch.abs(chain - pooled_median.view(1, 1, -1))
    ranked_folded = rank_normalize(
        folded.reshape(-1, folded.shape[-1])
    ).reshape(chain.shape)
    rhat_folded = split_rhat(ranked_folded)
    return torch.maximum(rhat_rank, rhat_folded)


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
