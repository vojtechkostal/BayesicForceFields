import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.means import (
    ImportedMean,
    build_mean,
    evaluate_mean,
    rdf_sigmoid_mean,
)
from bff.qoi.dataset import QoIDataset

X = torch.tensor([[0.0], [1.0], [2.0]])
Y = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 9.0]])


def test_data_mean_is_the_average_training_output() -> None:
    mean = build_mean("data", X, Y)
    torch.testing.assert_close(mean, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(build_mean(None, X, Y), mean)


def test_constant_mean_applies_to_every_output() -> None:
    values = evaluate_mean(build_mean(0.5, X, Y), X)
    torch.testing.assert_close(values, torch.full((3, 2), 0.5))


def test_sigmoid_rises_where_each_reference_rdf_reaches_one_half() -> None:
    r = np.arange(10) + 0.5
    first = (r > 3).astype(float)
    second = (r > 6).astype(float)
    dataset = QoIDataset(
        "rdf",
        np.zeros((2, 1)),
        np.zeros((2, 20)),
        np.concatenate([first, second]),
        labels=("A", "B"),
        values_per_label=10,
        settings={"bins": 10, "range": [0.0, 10.0]},
    )

    mean = rdf_sigmoid_mean(dataset).reshape(2, 10)

    assert mean[0, 3] == pytest.approx(0.5)
    assert mean[1, 6] == pytest.approx(0.5)
    assert np.all(np.diff(mean, axis=1) > 0)


def test_sigmoid_requires_rdf_settings() -> None:
    dataset = QoIDataset("rdf", np.zeros((2, 1)), np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="needs RDF settings"):
        rdf_sigmoid_mean(dataset)


def test_custom_mean_is_imported_and_saved_by_reference(tmp_path: Path) -> None:
    module = tmp_path / "custom_mean.py"
    module.write_text(
        "import torch\n"
        "def linear(X):\n"
        "    return torch.cat([X + 1.0, 4.0 * X + 4.0], dim=1)\n"
    )
    mean = build_mean(f"{module}:linear", X, Y)
    assert isinstance(mean, ImportedMean)
    assert pickle.loads(pickle.dumps(mean)).specification == mean.specification

    member = LocalGaussianProcess(
        X, Y, mean, torch.ones(1), 1.0, 1e-4
    )
    committee = LGPCommittee([member], y_ref=np.zeros(2), n_curves=2)
    committee.write(tmp_path / "model.lgp")
    loaded = LGPCommittee.load(tmp_path / "model.lgp")

    assert isinstance(loaded.members[0].mean, ImportedMean)
    torch.testing.assert_close(loaded.predict(X), committee.predict(X))


def test_invalid_means_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown mean 'linear'"):
        build_mean("linear", X, Y)
    with pytest.raises(ValueError, match="returns shape"):
        build_mean(lambda X: X, X, Y)
