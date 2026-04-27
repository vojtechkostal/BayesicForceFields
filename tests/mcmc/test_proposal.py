import pytest
import torch

from bff.mcmc.proposal import AdaptiveGaussianProposal


def test_adaptive_gaussian_proposal_initializes_state() -> None:
    proposal = AdaptiveGaussianProposal(device="cpu", dtype=torch.float64)

    proposal.initialize(2)

    assert proposal.n_dim == 2
    assert proposal.L.shape == (2, 2)
    assert proposal.n == 0
    assert torch.allclose(proposal.sum_x, torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(proposal.cov, torch.eye(2, dtype=torch.float64))


def test_adaptive_gaussian_proposal_preserves_shape_and_rejects_bad_dim() -> None:
    rng = torch.Generator(device="cpu").manual_seed(123)
    proposal = AdaptiveGaussianProposal(device="cpu")
    proposal.initialize(2)

    x = torch.zeros((4, 2))
    proposed = proposal.propose(x, rng)

    assert proposed.shape == x.shape
    with pytest.raises(ValueError, match="Expected shape"):
        proposal.propose(torch.zeros((4, 3)), rng)


def test_adapt_false_leaves_proposal_state_unchanged() -> None:
    proposal = AdaptiveGaussianProposal(adapt=False, device="cpu")
    proposal.initialize(2)
    state_before = proposal.state_dict()

    proposal.adapt(10, torch.ones((3, 2)), torch.ones(3, dtype=torch.bool))

    state_after = proposal.state_dict()
    assert state_after["n"] == state_before["n"]
    assert torch.allclose(state_after["L"], state_before["L"])
    assert state_after["scale"] == state_before["scale"]


def test_adapt_updates_running_covariance_and_scale() -> None:
    proposal = AdaptiveGaussianProposal(
        adapt=True,
        adapt_start=1,
        adapt_interval=1,
        target_acceptance=0.5,
        device="cpu",
    )
    proposal.initialize(2)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])

    proposal.adapt(1, x, torch.tensor([True, True, True]))

    assert proposal.n == 3
    assert proposal.scale > 1.0
    assert proposal.L.shape == (2, 2)
    assert torch.all(torch.diagonal(proposal.cov) > 0)


def test_proposal_state_dict_round_trips() -> None:
    proposal = AdaptiveGaussianProposal(device="cpu")
    proposal.initialize(2)
    proposal.adapt(1, torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.ones(2))

    restored = AdaptiveGaussianProposal(device="cpu")
    restored.load_state_dict(proposal.state_dict())

    assert restored.n_dim == proposal.n_dim
    assert restored.n == proposal.n
    assert restored.scale == proposal.scale
    assert torch.allclose(restored.L, proposal.L)
    assert torch.allclose(restored.sum_x, proposal.sum_x)
    assert torch.allclose(restored.sum_xx, proposal.sum_xx)
