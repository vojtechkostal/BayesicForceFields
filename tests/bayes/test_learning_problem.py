from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.learning import (
    LearningProblem,
    _default_checkpoint_path,
    _default_lgp_hyperpriors,
    _resolve_mean,
    fit_surrogates,
)
from bff.qoi.dataset import QoIDataset


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
        n_curves: int = 1,
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
        self.n_curves = n_curves
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

    problem = LearningProblem(
        {"a": FakeCommittee(shape=(3, 2)), "b": FakeCommittee(shape=(4, 2))}
    )
    assert problem.n_params == 2

    with pytest.raises(ValueError, match="same input dimension"):
        LearningProblem(
            {"a": FakeCommittee(n_params=2), "b": FakeCommittee(n_params=3)}
        )

    with pytest.raises(ValueError, match="reference output size"):
        LearningProblem({"qoi": FakeCommittee(y_size=2, reference_values=np.zeros(3))})

    with pytest.raises(ValueError, match="n_curves"):
        LearningProblem({"qoi": FakeCommittee(y_size=3, n_curves=2)})


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


def test_default_lgp_hyperpriors_follow_data_scales() -> None:
    X = torch.tensor(
        [[0.0, 10.0], [2.0, 14.0], [4.0, 18.0]],
        dtype=torch.float32,
    )
    residuals = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float32)

    priors = _default_lgp_hyperpriors(X, residuals)

    input_scales = X.std(dim=0, unbiased=False).numpy()
    target_scale = residuals.std(unbiased=False).item()
    expected_centers = np.log(
        [*input_scales, target_scale, 0.1 * target_scale**2]
    )
    assert priors.names == ["length_0", "length_1", "width", "noise"]
    assert priors.means == pytest.approx(expected_centers)
    assert priors.scales == pytest.approx([2.0, 2.0, 2.0, 3.0])


def test_default_lgp_hyperpriors_are_finite_for_constant_data() -> None:
    priors = _default_lgp_hyperpriors(
        torch.ones((4, 2)),
        torch.zeros((4, 1)),
    )

    assert np.all(np.isfinite(priors.means))
    assert np.exp(priors.means) == pytest.approx([1.0, 1.0, 1.0, 0.1])


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
        labels=("A", "B"),
        values_per_label=3,
        settings={"bins": 3, "range": (0.0, 3.0)},
    )

    mean = _resolve_mean(dataset, "sigmoid")

    assert mean.shape == (6,)
    assert np.allclose(mean[:3], mean[3:])
    assert np.all(np.diff(mean[:3]) > 0)


def test_resolve_mean_rejects_rdf_bin_count_mismatch() -> None:
    dataset = QoIDataset(
        name="rdf",
        inputs=np.zeros((2, 1)),
        outputs=np.zeros((2, 6)),
        outputs_ref=np.zeros(6),
        labels=("A", "B"),
        values_per_label=3,
        settings={"bins": 2, "range": (0.0, 3.0)},
    )

    with pytest.raises(ValueError, match="each RDF curve contains 3 values"):
        _resolve_mean(dataset, "sigmoid")


def test_lgp_cache_rejects_same_shape_different_data(tmp_path: Path) -> None:
    first = QoIDataset("pmf", [[0.0], [1.0]], [[1.0], [2.0]], [1.5])
    second = QoIDataset("pmf", [[0.0], [1.0]], [[1.0], [3.0]], [1.5])
    lgp = LocalGaussianProcess(
        torch.tensor(first.inputs, dtype=torch.float32),
        torch.tensor(first.outputs, dtype=torch.float32),
        0.0,
        torch.ones(1),
        1.0,
        0.1,
        "cpu",
    )
    model = LGPCommittee(
        [lgp],
        first.outputs_ref,
        n_curves=1,
        dataset_fingerprint=first.fingerprint(),
    )
    fn_model = tmp_path / "pmf.lgp"
    model.write(fn_model)
    with pytest.raises(ValueError, match="different QoI data"):
        fit_surrogates(
            [second],
            model_paths={"pmf": fn_model},
            reuse_models=True,
            device="cpu",
        )
