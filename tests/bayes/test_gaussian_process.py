from pathlib import Path

import numpy as np
import pytest
import torch

from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.kernels import gaussian_kernel


class ParameterDependentMean:
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X[:, :1] * torch.tensor([1.0, 2.0], device=X.device)


def test_gaussian_kernel_shapes_and_symmetry() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    lengths = torch.tensor([1.0, 2.0])

    kernel = gaussian_kernel(x, x, lengths, torch.tensor(2.0))

    assert kernel.shape == (2, 2)
    assert torch.allclose(kernel, kernel.T)

    batched = gaussian_kernel(
        x.repeat(2, 1, 1),
        x.repeat(2, 1, 1),
        lengths.repeat(2, 1),
        torch.tensor([1.0, 2.0]),
    )
    assert batched.shape == (2, 2, 2)
    assert torch.allclose(batched[:, 0, 0], torch.tensor([1.0, 4.0]))


def test_local_gaussian_process_predicts_training_values_approximately() -> None:
    X = torch.tensor([[0.0], [1.0], [2.0]])
    y = torch.tensor([[0.0], [1.0], [0.0]])
    gp = LocalGaussianProcess(
        X,
        y,
        torch.tensor([0.0]),
        torch.tensor([0.4]),
        torch.tensor(1.0),
        torch.tensor(1e-5),
    )

    pred = gp.predict(X)

    assert pred.shape == y.shape
    assert torch.allclose(pred, y, atol=1e-3)


def test_lgp_committee_averages_predictions_and_round_trips(tmp_path: Path) -> None:
    X = torch.tensor([[0.0], [1.0]])
    y1 = torch.tensor([[0.0], [1.0]])
    y2 = torch.tensor([[1.0], [2.0]])
    gp1 = LocalGaussianProcess(
        X, y1, torch.tensor([0.0]), torch.tensor([1.0]), 1.0, 1e-4)
    gp2 = LocalGaussianProcess(
        X, y2, torch.tensor([0.0]), torch.tensor([1.0]), 1.0, 1e-4)
    committee = LGPCommittee(
        [gp1, gp2],
        y_ref=np.array([0.5]),
        n_curves=1,
        nuisance=None,
    )

    pred = committee.predict(torch.tensor([[0.0]]))

    assert pred.shape == (1, 1)
    expected = torch.stack(
        [
            gp1.predict(torch.tensor([[0.0]])),
            gp2.predict(torch.tensor([[0.0]])),
        ]
    ).mean()
    assert pred.item() == expected.item()

    path = tmp_path / "model.lgp"
    committee.test_error = 12.3
    committee.write(path)
    loaded = LGPCommittee.load(path)

    assert loaded.n_members == 2
    assert loaded.n_eff == 1
    assert loaded.n_curves == 1
    assert loaded.curve_length == 1
    assert loaded.test_error == 12.3
    assert np.allclose(loaded.y_ref, [0.5])


def test_lgp_committee_round_trips_curve_count(tmp_path: Path) -> None:
    X = torch.tensor([[0.0], [1.0], [2.0]])
    y = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5, 3.5],
        [1.0, 2.0, 3.0, 4.0],
    ])
    gp = LocalGaussianProcess(
        X,
        y,
        torch.zeros(4),
        torch.tensor([1.0]),
        1.0,
        1e-4,
    )
    committee = LGPCommittee(
        [gp],
        y_ref=np.arange(4),
        n_curves=2,
    )
    path = tmp_path / "model.lgp"
    committee.write(path)

    loaded = LGPCommittee.load(path)

    assert loaded.n_eff == 4
    assert loaded.n_curves == 2
    assert loaded.curve_length == 2


def test_lgp_committee_rejects_models_of_older_versions(tmp_path: Path) -> None:
    path = tmp_path / "legacy.lgp"
    torch.save({"lgps": [], "reference_values": [0.0], "n_curves": 1}, path)

    with pytest.raises(ValueError, match="older BFF version; refit"):
        LGPCommittee.load(path)


def test_local_gaussian_process_round_trips_parameter_dependent_mean(
    tmp_path: Path,
) -> None:
    X = torch.tensor([[0.05], [0.08], [0.11]])
    mean = ParameterDependentMean()
    y = mean(X)
    gp = LocalGaussianProcess(X, y, mean, torch.ones(1), 1.0, 1e-4)
    committee = LGPCommittee([gp], y_ref=y[0].numpy(), n_curves=1)
    path = tmp_path / "dynamic-mean.lgp"

    committee.write(path)
    loaded = LGPCommittee.load(path)

    assert callable(loaded.members[0].mean)
    assert torch.allclose(loaded.predict(X), y)


def test_local_gaussian_process_prediction_matches_explicit_inverse() -> None:
    torch.manual_seed(0)
    x = torch.rand(12, 2)
    y = torch.sin(x.sum(dim=1, keepdim=True)) + torch.rand(12, 3)
    lgp = LocalGaussianProcess(x, y, 0.2, torch.tensor([0.5, 0.7]), 1.3, 1e-3)
    xi = torch.rand(5, 2)

    x64, y64, xi64 = x.double(), y.double(), xi.double()
    kernel = gaussian_kernel(x64, x64, lgp.lengthscales, lgp.amplitude)
    kernel = kernel + 1e-3 * torch.eye(12, dtype=torch.float64)
    cross = gaussian_kernel(xi64, x64, lgp.lengthscales, lgp.amplitude)
    expected = 0.2 + cross @ torch.linalg.inv(kernel) @ (y64 - 0.2)
    # Float32 input is answered in float32, from the float64 solution.
    assert lgp.predict(xi).dtype == torch.float32
    torch.testing.assert_close(lgp.predict(xi64), expected)
    torch.testing.assert_close(
        lgp.predict(xi), expected.float(), rtol=1e-4, atol=1e-5
    )
