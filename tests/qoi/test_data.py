from pathlib import Path

import numpy as np
import pytest

from bff.qoi.data import QoI, QoIDataset


def test_qoi_validates_label_shape_and_round_trips_dict() -> None:
    qoi = QoI(
        name="rdf",
        values=[1.0, 2.0, 3.0, 4.0],
        labels=("a", "b"),
        values_per_label=2,
        settings={"n_bins": 2},
    )

    loaded = QoI.from_dict(qoi.to_dict())

    assert loaded.name == "rdf"
    assert loaded.labels == ("a", "b")
    assert loaded.n_values == 4
    assert loaded.settings == {"n_bins": 2}

    with pytest.raises(ValueError, match="values_per_label"):
        QoI("bad", [1.0], values_per_label=0)
    with pytest.raises(ValueError, match="labels"):
        QoI("bad", [1.0, 2.0, 3.0], labels=("a",), values_per_label=2)


def test_qoi_dataset_validates_shapes_and_round_trips_file(tmp_path: Path) -> None:
    dataset = QoIDataset(
        name="rdf",
        inputs=np.zeros((3, 2)),
        outputs=np.ones((3, 4)),
        outputs_ref=np.arange(4),
        labels=("a", "b"),
        values_per_label=2,
        nuisance=0.5,
        metadata={"source": "test"},
    )

    assert dataset.n_samples == 3
    assert dataset.n_curves == 2
    assert dataset.curve_length == 2

    path = tmp_path / "dataset.pt"
    dataset.write(path)
    loaded = QoIDataset.load(path)

    assert loaded.name == dataset.name
    assert loaded.labels == dataset.labels
    assert loaded.nuisance == 0.5
    assert np.allclose(loaded.outputs_ref, dataset.outputs_ref)


def test_qoi_dataset_counts_labeled_curves() -> None:
    dataset = QoIDataset(
        name="rdf",
        inputs=np.zeros((2, 1)),
        outputs=np.zeros((2, 8)),
        outputs_ref=np.zeros(8),
        labels=(
            "window-000:OC",
            "window-001:OC",
            "window-000:CC",
            "window-001:CC",
        ),
        values_per_label=2,
    )

    assert dataset.n_curves == 4
    assert dataset.curve_length == 2


def test_qoi_dataset_counts_unlabeled_curves_from_values_per_label() -> None:
    dataset = QoIDataset(
        name="pmf",
        inputs=np.zeros((2, 1)),
        outputs=np.zeros((2, 40)),
        outputs_ref=np.zeros(40),
        values_per_label=20,
    )

    assert dataset.n_curves == 2
    assert dataset.curve_length == 20


def test_qoi_dataset_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="input samples"):
        QoIDataset("qoi", np.zeros((2, 1)), np.zeros((3, 1)), np.zeros(1))

    with pytest.raises(ValueError, match="Output dimension"):
        QoIDataset("qoi", np.zeros((2, 1)), np.zeros((2, 2)), np.zeros(1))

    with pytest.raises(ValueError, match="labels"):
        QoIDataset(
            "qoi",
            np.zeros((2, 1)),
            np.zeros((2, 3)),
            np.zeros(3),
            labels=("a",),
            values_per_label=2,
        )
