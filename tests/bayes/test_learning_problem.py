from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bff.bayes.learning import (
    LearningProblem,
    _default_checkpoint_path,
    _effective_observations,
    _resolve_mean,
)
from bff.qoi.data import QoIDataset


class FakeLGP:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.X_train = torch.zeros(shape)


class FakeCommittee:
    def __init__(
        self,
        *,
        shape: tuple[int, int] = (3, 2),
        n_params: int = 2,
        y_size: int = 2,
        reference_values: np.ndarray | None = None,
        nuisance: float | None = None,
    ) -> None:
        self.lgps = [FakeLGP(shape)]
        self.n_params = n_params
        self.y_size = y_size
        self.reference_values = (
            np.asarray(reference_values, dtype=float)
            if reference_values is not None
            else np.zeros(y_size)
        )
        self.nuisance = nuisance


def test_learning_problem_rejects_invalid_model_structures() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LearningProblem({})

    empty = FakeCommittee()
    empty.lgps = []
    with pytest.raises(ValueError, match="without committee"):
        LearningProblem({"qoi": empty})

    inconsistent_committee = FakeCommittee()
    inconsistent_committee.lgps.append(FakeLGP((4, 2)))
    with pytest.raises(ValueError, match="inconsistent training input"):
        LearningProblem({"qoi": inconsistent_committee})

    with pytest.raises(ValueError, match="same training input shape"):
        LearningProblem(
            {"a": FakeCommittee(shape=(3, 2)), "b": FakeCommittee(shape=(4, 2))}
        )

    with pytest.raises(ValueError, match="same input dimension"):
        LearningProblem(
            {"a": FakeCommittee(n_params=2), "b": FakeCommittee(n_params=3)}
        )

    with pytest.raises(ValueError, match="reference output size"):
        LearningProblem({"qoi": FakeCommittee(y_size=2, reference_values=np.zeros(3))})


def test_learning_problem_builds_priors_from_constraint() -> None:
    constraint = SimpleNamespace(
        n_params=1,
        explicit_bounds=np.array([[-2.0, 2.0]]),
        explicit_parameter_names=["charge A"],
    )
    problem = LearningProblem(
        {"qoi": FakeCommittee(shape=(3, 1), n_params=1, nuisance=None)},
        constraint=constraint,
    )

    priors = problem.build_priors("uniform")

    assert priors.names == ["charge A", "log_sigma_qoi"]
    assert problem.n_free_nuisance == 1
    assert problem.parameter_bounds.shape == (1, 2)


def test_learning_problem_rejects_constraint_dimension_mismatch() -> None:
    constraint = SimpleNamespace(n_params=3)

    with pytest.raises(ValueError, match="disagree"):
        LearningProblem({"qoi": FakeCommittee(n_params=2)}, constraint=constraint)


def test_learning_helpers() -> None:
    from pathlib import Path

    assert _default_checkpoint_path(Path("posterior.pt")).name == "posterior.ckpt.pt"

    dataset = QoIDataset(
        name="qoi",
        inputs=np.zeros((2, 1)),
        outputs=np.zeros((2, 4)),
        outputs_ref=np.zeros(4),
        values_per_label=2,
    )
    assert _effective_observations(dataset, 0.5) == 1


def test_resolve_mean_accepts_vector_rdf_mean() -> None:
    dataset = QoIDataset(
        name="rdf",
        inputs=np.zeros((2, 1)),
        outputs=np.zeros((2, 3)),
        outputs_ref=np.zeros(3),
    )
    mean = np.ones(3)

    assert _resolve_mean(dataset, mean) is mean


def test_resolve_mean_builds_sigmoid_for_concatenated_rdfs() -> None:
    dataset = QoIDataset(
        name="rdf",
        inputs=np.zeros((2, 1)),
        outputs=np.zeros((2, 6)),
        outputs_ref=np.zeros(6),
        settings={"n_bins": 3, "r_range": (0.0, 3.0)},
    )

    mean = _resolve_mean(dataset, "sigmoid")

    assert mean.shape == (6,)
    assert np.allclose(mean[:3], mean[3:])
    assert np.all(np.diff(mean[:3]) > 0)
