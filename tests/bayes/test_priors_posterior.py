from pathlib import Path

import numpy as np
import pytest
import torch

from bff.bayes.posterior import log_posterior
from bff.bayes.priors import Prior, Priors, log_prior


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


def test_priors_from_bounds_names_nuisance_and_round_trip(tmp_path: Path) -> None:
    priors = Priors.from_bounds(
        np.array([[-1.0, 1.0], [2.0, 4.0]]),
        dist_type="uniform",
        n_nuisance=1,
        names=["a", "b"],
        nuisance_names=["log_sigma_qoi"],
    )

    assert priors.names == ["a", "b", "log_sigma_qoi"]
    assert len(priors) == 3

    path = tmp_path / "priors.pt"
    priors.write(path)
    loaded = Priors.load(path)

    assert loaded.names == priors.names
    assert [item.to_dict() for item in loaded] == [item.to_dict() for item in priors]


def test_log_prior_accepts_vector_and_batch() -> None:
    priors = Priors([Prior("normal", 0.0, 1.0), Prior("uniform", -1.0, 1.0)])

    vector = log_prior(torch.tensor([0.0, 0.0]), priors)
    batch = log_prior(torch.tensor([[0.0, 0.0], [1.0, 0.5]]), priors)

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
        device="cpu",
        numpy_output=False,
    )

    assert out.shape == (2,)
    assert torch.isfinite(out[0])
    assert torch.isneginf(out[1])

    nan_out = log_posterior(
        torch.tensor([[float("nan"), 0.0]]),
        priors,
        likelihood,
        device="cpu",
    )
    assert isinstance(nan_out, np.ndarray)
    assert float(nan_out) == -1e10
