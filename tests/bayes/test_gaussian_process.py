from pathlib import Path

import numpy as np
import torch

from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.kernels import gaussian_kernel


def test_gaussian_kernel_shapes_symmetry_and_manual_path() -> None:
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    lengths = torch.tensor([1.0, 2.0])

    kernel = gaussian_kernel(x, x, lengths, 2.0)
    manual = gaussian_kernel(x, x, lengths, 2.0, manual_sqdist=True)

    assert kernel.shape == (2, 2)
    assert torch.allclose(kernel, kernel.T)
    assert torch.allclose(kernel, manual)

    batched = gaussian_kernel(
        x.repeat(2, 1, 1),
        x.repeat(2, 1, 1),
        lengths.repeat(2, 1),
        torch.ones(2),
    )
    assert batched.shape == (2, 2, 2)


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
        "cpu",
    )

    pred = gp.predict(X)

    assert pred.shape == y.shape
    assert torch.allclose(pred, y, atol=1e-3)


def test_lgp_committee_averages_predictions_and_round_trips(tmp_path: Path) -> None:
    X = torch.tensor([[0.0], [1.0]])
    y1 = torch.tensor([[0.0], [1.0]])
    y2 = torch.tensor([[1.0], [2.0]])
    gp1 = LocalGaussianProcess(
        X, y1, torch.tensor([0.0]), torch.tensor([1.0]), 1.0, 1e-4, "cpu"
    )
    gp2 = LocalGaussianProcess(
        X, y2, torch.tensor([0.0]), torch.tensor([1.0]), 1.0, 1e-4, "cpu"
    )
    committee = LGPCommittee([gp1, gp2], 2, np.array([0.5]), nuisance=None)

    pred = committee.predict(torch.tensor([[0.0]]))

    assert pred.shape == (1,)
    expected = torch.stack(
        [
            gp1.predict(torch.tensor([[0.0]])),
            gp2.predict(torch.tensor([[0.0]])),
        ]
    ).mean()
    assert pred.item() == expected.item()

    path = tmp_path / "model.lgp"
    committee.error = 12.3
    committee.write(path)
    loaded = LGPCommittee.load(path)

    assert loaded.size == 2
    assert loaded.n_observations == 2
    assert loaded.error == 12.3
    assert np.allclose(loaded.reference_values, [0.5])
