import math

import pytest
import torch

from bff.mcmc.convergence import MIN_STEPS, Diagnostics, diagnose


def ar1(n: int, m: int, d: int, phi: float, seed: int = 0) -> torch.Tensor:
    """Stationary AR(1) walkers with unit marginal variance."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn((n, m, d), generator=gen) * math.sqrt(1 - phi**2)
    x = torch.empty((n, m, d))
    x[0] = torch.randn((m, d), generator=gen)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    return x


def test_independent_draws_have_unit_rhat_and_full_ess() -> None:
    chain = ar1(500, 8, 2, phi=0.0)

    result = diagnose(chain, chain[..., 0])

    assert result.rhat.shape == (3,)
    assert result.max_rhat == pytest.approx(1.0, abs=0.02)
    # 4000 draws; antithetic estimates may exceed the count slightly.
    assert (result.ess_bulk > 3000).all()
    assert (result.ess_tail > 2000).all()


def test_ess_falls_with_autocorrelation() -> None:
    # tau = (1 + phi) / (1 - phi) = 19 for phi = 0.9.
    chain = ar1(1000, 8, 1, phi=0.9)

    result = diagnose(chain, chain[..., 0])

    expected = 8000 / 19
    assert result.ess_bulk[0] == pytest.approx(expected, rel=0.3)
    assert result.ess_tail[0] < 8000


def test_walkers_in_different_places_are_not_converged() -> None:
    chain = ar1(200, 6, 1, phi=0.5)
    chain[:, :3] += 5.0

    result = diagnose(chain, chain[..., 0])

    assert result.max_rhat > 1.5
    assert not result.converged(rhat_tol=1.01, ess_min=1)


def test_constant_column_is_reported_as_nan_and_not_converged() -> None:
    chain = ar1(100, 4, 2, phi=0.3)
    chain[..., 1] = 0.7

    result = diagnose(chain, chain[..., 0])

    assert math.isnan(result.max_rhat) and math.isnan(result.min_ess)
    assert not result.converged(rhat_tol=10.0, ess_min=0)


def test_repeated_states_from_rejected_moves_are_handled() -> None:
    chain = ar1(300, 6, 1, phi=0.2)
    chain[1::2] = chain[::2][: chain[1::2].shape[0]]  # every move rejected once

    result = diagnose(chain, chain[..., 0])

    assert torch.isfinite(result.rhat).all()
    assert (result.ess_bulk < 6 * 300).all()


def test_diagnose_needs_enough_steps() -> None:
    chain = torch.randn((MIN_STEPS - 1, 2, 1))

    with pytest.raises(ValueError, match="at least"):
        diagnose(chain, chain[..., 0])


def test_diagnostics_round_trip_through_dict() -> None:
    chain = ar1(100, 4, 2, phi=0.3)
    result = diagnose(chain, chain[..., 0])

    loaded = Diagnostics.from_dict(result.to_dict())

    assert torch.equal(loaded.rhat, result.rhat)
    assert loaded.min_ess == result.min_ess
