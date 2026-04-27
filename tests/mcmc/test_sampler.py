from pathlib import Path

import pytest
import torch

from bff.mcmc.proposal import AdaptiveGaussianProposal
from bff.mcmc.sampler import Checkpoint, Sampler


def _log_prob(x: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(x**2, dim=1)


def _sampler(seed: int = 11) -> Sampler:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    proposal = AdaptiveGaussianProposal(
        proposal_cov=0.05 * torch.eye(2),
        adapt=False,
        device="cpu",
    )
    return Sampler(_log_prob, proposal, rng=rng, device="cpu", dtype=torch.float32)


def test_sampler_validates_run_inputs() -> None:
    sampler = _sampler()

    with pytest.raises(ValueError, match="shape"):
        sampler._validate_inputs(torch.zeros(2), 10, 1, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        sampler._validate_inputs(torch.zeros((2, 1)), 0, 0, 1, 1)
    with pytest.raises(ValueError, match="smaller"):
        sampler._validate_inputs(torch.zeros((2, 1)), 4, 4, 1, 1)
    with pytest.raises(ValueError, match="thin"):
        sampler._validate_inputs(torch.zeros((2, 1)), 4, 1, 0, 1)
    with pytest.raises(ValueError, match="progress_stride"):
        sampler._validate_inputs(torch.zeros((2, 1)), 4, 1, 1, 0)


def test_log_prob_requires_walker_vector_and_sanitizes_nonfinite() -> None:
    sampler = Sampler(
        lambda x: torch.tensor([float("nan"), float("inf"), -1.0]),
        AdaptiveGaussianProposal(device="cpu"),
        rng=torch.Generator(device="cpu").manual_seed(1),
        device="cpu",
    )
    out = sampler._log_prob(torch.zeros((3, 2)))

    assert torch.isneginf(out[:2]).all()
    assert out[2].item() == -1.0

    bad = Sampler(
        lambda x: torch.zeros((x.shape[0], 1)),
        AdaptiveGaussianProposal(device="cpu"),
        rng=torch.Generator(device="cpu").manual_seed(1),
        device="cpu",
    )
    with pytest.raises(ValueError, match="log_prob must return"):
        bad._log_prob(torch.zeros((3, 2)))


def test_sampler_run_reports_expected_chain_shape() -> None:
    sampler = _sampler()
    p0 = torch.zeros((4, 2))

    checkpoints = list(
        sampler.run(p0, total_steps=8, warmup=2, thin=2, progress_stride=3)
    )

    assert checkpoints[-1].step == 8
    assert checkpoints[-1].posterior.shape == (3, 4, 2)
    assert sampler.chain.shape == (3, 4, 2)
    assert checkpoints[-1].acceptance_rate is not None


def test_checkpoint_round_trip_and_restore(tmp_path: Path) -> None:
    sampler = _sampler()
    checkpoint = list(
        sampler.run(torch.zeros((3, 2)), total_steps=6, warmup=1, progress_stride=6)
    )[-1]
    path = tmp_path / "checkpoint.pt"

    checkpoint.write(path)
    loaded = Checkpoint.load(path)

    restored_sampler = _sampler()
    restored_sampler.proposal.initialize(2)
    p, logp, accepted, step = loaded.restore(restored_sampler)

    assert step == checkpoint.step
    assert torch.allclose(p, checkpoint.p)
    assert torch.allclose(logp, checkpoint.logp)
    assert torch.equal(accepted.cpu(), checkpoint.accepted)
    assert torch.allclose(restored_sampler.chain.cpu(), checkpoint.posterior)


def test_restart_rejects_mismatched_checkpoint_settings(tmp_path: Path) -> None:
    checkpoint = list(
        _sampler().run(
            torch.zeros((3, 2)),
            total_steps=5,
            warmup=1,
            thin=1,
            fn_checkpoint=tmp_path / "checkpoint.pt",
            progress_stride=5,
        )
    )[-1]
    assert checkpoint.step == 5

    with pytest.raises(ValueError, match="warmup"):
        list(
            _sampler().run(
                torch.zeros((3, 2)),
                total_steps=6,
                warmup=2,
                thin=1,
                restart=True,
                fn_checkpoint=tmp_path / "checkpoint.pt",
            )
        )
