"""Convergence diagnostics of a multi-walker chain.

:func:`diagnose` implements the rank-normalized split R-hat and the bulk and
tail effective sample sizes of Vehtari et al. (2021), Bayesian Analysis 16(2).
The sampler stops on them and :class:`bff.Results` reports the same numbers.
"""

import math
from dataclasses import dataclass

import torch

MIN_STEPS = 20  # saved steps per walker below which nothing is diagnosed


@dataclass
class Diagnostics:
    """R-hat, bulk ESS, and tail ESS of every column of a chain.

    The columns are the chain dimensions followed by the log probability; ESS
    counts effective samples over all walkers together. NaN marks a column
    that does not vary.
    """

    rhat: torch.Tensor
    ess_bulk: torch.Tensor
    ess_tail: torch.Tensor

    @property
    def max_rhat(self) -> float:
        """Largest R-hat; NaN if any column does not vary."""
        return self.rhat.max().item() if not self.rhat.isnan().any() else math.nan

    @property
    def min_ess(self) -> float:
        """Smallest bulk or tail ESS; NaN if any column does not vary."""
        ess = torch.minimum(self.ess_bulk, self.ess_tail)
        return ess.min().item() if not ess.isnan().any() else math.nan

    def converged(self, rhat_tol: float, ess_min: float) -> bool:
        """Whether R-hat is below ``rhat_tol`` and the ESS above ``ess_min``."""
        return self.max_rhat < rhat_tol and self.min_ess >= ess_min

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).cpu() for name in _FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor]) -> "Diagnostics":
        return cls(**{name: torch.as_tensor(data[name]) for name in _FIELDS})


_FIELDS = ("rhat", "ess_bulk", "ess_tail")


def diagnose(chain: torch.Tensor, log_prob: torch.Tensor) -> Diagnostics:
    """Diagnose a chain of shape ``(n_steps, n_walkers, n_dim)`` and its log
    probabilities of shape ``(n_steps, n_walkers)``.

    Every walker is split in two halves and treated as an independent chain.
    Cost grows as ``n_steps * log(n_steps)``.
    """
    if chain.shape[0] < MIN_STEPS:
        raise ValueError(f"Need at least {MIN_STEPS} saved steps to diagnose.")
    x = torch.cat([chain, log_prob.unsqueeze(-1)], dim=-1).double()
    half = x.shape[0] // 2
    x = torch.cat([x[:half], x[-half:]], dim=1)  # (half, 2 * n_walkers, n_col)
    pooled = x.reshape(-1, x.shape[-1])

    ranked = _rank_normalize(x)
    folded = _rank_normalize((x - pooled.median(0).values).abs())
    rhat = torch.maximum(_rhat(ranked), _rhat(folded))

    levels = torch.tensor([0.05, 0.95], dtype=x.dtype, device=x.device)
    low, high = torch.quantile(pooled, levels, dim=0)
    ess_tail = torch.minimum(
        _ess((x <= low).to(x.dtype)), _ess((x <= high).to(x.dtype))
    )
    return Diagnostics(rhat=rhat, ess_bulk=_ess(ranked), ess_tail=ess_tail)


def _rank_normalize(x: torch.Tensor) -> torch.Tensor:
    """Replace values by normal scores of their pooled ranks (ties averaged),
    separately for every column of ``x`` with shape ``(n, m, n_col)``."""
    n, m, n_col = x.shape
    size = n * m
    flat = x.reshape(size, n_col)
    values, order = flat.sort(dim=0, stable=True)

    # Average the ranks of tied values: Metropolis chains repeat states.
    new_group = torch.ones_like(values, dtype=torch.long)
    new_group[1:] = values[1:] != values[:-1]
    group = new_group.cumsum(0) - 1 + torch.arange(n_col, device=x.device) * size
    group = group.reshape(-1)
    ordinal = torch.arange(1, size + 1, dtype=x.dtype, device=x.device)
    ordinal = ordinal.unsqueeze(1).expand(size, n_col).reshape(-1)
    sums = torch.zeros(size * n_col, dtype=x.dtype, device=x.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, group, ordinal)
    counts.scatter_add_(0, group, torch.ones_like(ordinal))
    average = (sums / counts.clamp(min=1))[group].reshape(size, n_col)

    ranks = torch.empty_like(flat).scatter_(0, order, average)
    u = ((ranks - 0.375) / (size + 0.25)).clamp(1e-12, 1 - 1e-12)
    return (math.sqrt(2.0) * torch.erfinv(2 * u - 1)).reshape(n, m, n_col)


def _rhat(x: torch.Tensor) -> torch.Tensor:
    """R-hat of chains ``x`` with shape ``(n, m, n_col)``."""
    n = x.shape[0]
    within = x.var(dim=0, unbiased=True).mean(0)
    between = n * x.mean(0).var(dim=0, unbiased=True)
    variance = (n - 1) / n * within + between / n
    return torch.sqrt(variance / within)


def _ess(x: torch.Tensor) -> torch.Tensor:
    """Effective sample size of chains ``x`` with shape ``(n, m, n_col)``:
    pooled within-chain autocovariance, truncated by Geyer's initial monotone
    pair sequence."""
    n, m, _ = x.shape
    centered = x - x.mean(0, keepdim=True)
    n_fft = 1 << (2 * n - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=n_fft, dim=0)
    acov = torch.fft.irfft(spectrum * spectrum.conj(), n=n_fft, dim=0)[:n] / n

    within = acov[0].mean(0) * n / (n - 1)
    variance = within * (n - 1) / n + x.mean(0).var(dim=0, unbiased=True)
    rho = 1 - (within - acov.mean(1)) / variance
    rho[0] = 1.0

    pairs = rho[: n - n % 2].reshape(n // 2, 2, -1).sum(1)
    alive = torch.cumprod((pairs > 0).to(x.dtype), dim=0)
    tau = -1 + 2 * (torch.cummin(pairs, dim=0).values * alive).sum(0)
    tau = tau.clamp(min=1 / math.log10(n * m))
    return torch.where(variance > 0, n * m / tau, torch.full_like(tau, math.nan))
