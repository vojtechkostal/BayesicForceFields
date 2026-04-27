import pytest
import torch

from bff.mcmc.convergence import (
    autocorrelation,
    integrated_autocorr_time,
    rank_normalize,
    split_rhat,
)


def test_split_rhat_requires_enough_samples() -> None:
    chain = torch.randn((3, 2, 1))

    with pytest.raises(ValueError, match="at least 4 samples"):
        split_rhat(chain)


def test_split_rhat_returns_finite_values_for_similar_chains() -> None:
    base = torch.linspace(-1.0, 1.0, 40)
    chain = torch.stack(
        [
            torch.stack([base, base + 0.01], dim=1),
            torch.stack([base + 0.02, base + 0.03], dim=1),
        ],
        dim=1,
    )

    rhat = split_rhat(chain)

    assert rhat.shape == (2,)
    assert torch.isfinite(rhat).all()
    assert torch.all(rhat > 0)


def test_rank_normalize_preserves_shape_and_returns_finite_values() -> None:
    values = torch.tensor([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])

    normalized = rank_normalize(values)

    assert normalized.shape == values.shape
    assert torch.isfinite(normalized).all()


def test_autocorrelation_has_unit_lag_zero() -> None:
    x = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])

    acf = autocorrelation(x)

    assert acf.shape == x.shape
    assert torch.allclose(acf[0], torch.ones(2))


def test_integrated_autocorr_time_shape_and_lower_bound() -> None:
    x = torch.randn((20, 3, 2), generator=torch.Generator().manual_seed(5))

    tau = integrated_autocorr_time(x)

    assert tau.shape == (3, 2)
    assert torch.isfinite(tau).all()
    assert torch.all(tau >= 1.0)
