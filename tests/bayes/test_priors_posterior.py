import numpy as np
import pytest
import torch

from bff.bayes.posterior import log_posterior
from bff.bayes.priors import Prior, Priors


def test_prior_validation_and_properties() -> None:
    with pytest.raises(ValueError, match="Unknown prior"):
        Prior("bad", 0.0, 1.0)
    with pytest.raises(ValueError, match="scale"):
        Prior("normal", 0.0, 0.0)
    with pytest.raises(ValueError, match="lower < upper"):
        Prior("uniform", 1.0, 1.0)

    normal = Prior("NORMAL", 2.0, 3.0, name="x")
    uniform = Prior("uniform", -1.0, 1.0)

    assert normal.kind == "normal"
    assert normal.mean == 2.0
    assert normal.scale == 3.0
    assert uniform.mean == 0.0
    assert uniform.scale == pytest.approx(2.0 / np.sqrt(12))


def test_priors_from_bounds_names_nuisance_and_round_trip() -> None:
    priors = Priors.from_bounds(
        np.array([[-1.0, 1.0], [2.0, 4.0]]),
        dist_type="uniform",
        names=["a", "b"],
        nuisance_names=["log noise qoi"],
    )

    assert priors.names == ["a", "b", "log noise qoi"]
    assert len(priors) == 3

    loaded = Priors.from_dicts(priors.to_dicts())

    assert loaded.names == priors.names
    assert loaded.to_dicts() == priors.to_dicts()


def test_normal_priors_from_bounds_cover_the_interval() -> None:
    priors = Priors.from_bounds(np.array([[0.0, 1.0]]), names=["x"])

    assert priors[0].mean == 0.5
    assert priors[0].scale == pytest.approx(0.2)


def test_priors_log_prob_accepts_vector_and_batch() -> None:
    priors = Priors([Prior("normal", 0.0, 1.0), Prior("uniform", -1.0, 1.0)])

    vector = priors.log_prob(torch.tensor([0.0, 0.0]))
    batch = priors.log_prob(torch.tensor([[0.0, 0.0], [1.0, 0.5]]))

    assert vector.shape == (1,)
    assert batch.shape == (2,)
    assert torch.isfinite(batch).all()


def test_log_posterior_handles_shapes_nan_and_output_type() -> None:
    priors = Priors([Prior("normal", 0.0, 1.0), Prior("normal", 0.0, 1.0)])

    def likelihood(theta: torch.Tensor) -> torch.Tensor:
        return torch.where(theta[:, 0] > 0.5, torch.nan, -theta[:, 0] ** 2)

    out = log_posterior(
        torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        priors,
        likelihood,
    )

    assert out.shape == (2,)
    assert torch.isfinite(out[0])
    assert torch.isneginf(out[1])

    mixed = log_posterior(
        torch.tensor([[float("nan"), 0.0], [0.0, 0.0]]),
        priors,
        likelihood,
    )
    assert torch.isneginf(mixed[0])
    assert torch.isfinite(mixed[1])
