from pathlib import Path

import numpy as np
import pytest
import torch

from bff.bayes import fit as fit_module
from bff.bayes.fit import _default_lgp_hyperpriors, fit_surrogates
from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.learning import LearningProblem
from bff.domain.specs import Specs
from bff.qoi.dataset import QoIDataset


class FakeLGP:
    def __init__(self, shape: tuple[int, int]) -> None:
        self.X_train = torch.zeros(shape)


class FakeCommittee:
    def __init__(
        self,
        *,
        shape: tuple[int, int] = (3, 2),
        n_inputs: int = 2,
        n_outputs: int = 2,
        y_ref: np.ndarray | None = None,
        n_curves: int = 1,
        nuisance: float | None = None,
    ) -> None:
        self.members = [FakeLGP(shape)]
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.y_ref = (
            np.asarray(y_ref, dtype=float)
            if y_ref is not None
            else np.zeros(n_outputs)
        )
        self.n_curves = n_curves
        self.nuisance = nuisance


def test_learning_problem_rejects_invalid_model_structures() -> None:
    with pytest.raises(ValueError, match="at least one"):
        LearningProblem({})

    empty = FakeCommittee()
    empty.members = []
    with pytest.raises(ValueError, match="without committee"):
        LearningProblem({"qoi": empty})

    inconsistent_committee = FakeCommittee()
    inconsistent_committee.members.append(FakeLGP((4, 2)))
    with pytest.raises(ValueError, match="inconsistent training input"):
        LearningProblem({"qoi": inconsistent_committee})

    problem = LearningProblem(
        {"a": FakeCommittee(shape=(3, 2)), "b": FakeCommittee(shape=(4, 2))}
    )
    assert problem.n_params == 2

    with pytest.raises(ValueError, match="same input dimension"):
        LearningProblem(
            {"a": FakeCommittee(n_inputs=2), "b": FakeCommittee(n_inputs=3)}
        )

    with pytest.raises(ValueError, match="reference output size"):
        LearningProblem({"qoi": FakeCommittee(n_outputs=2, y_ref=np.zeros(3))})

    with pytest.raises(ValueError, match="n_curves"):
        LearningProblem({"qoi": FakeCommittee(n_outputs=3, n_curves=2)})


def test_learning_problem_builds_priors_from_specs() -> None:
    specs = Specs({"bounds": {"charge A": [-2.0, 2.0]}, "charge_constraints": []})
    problem = LearningProblem(
        {"qoi": FakeCommittee(shape=(3, 1), n_inputs=1, nuisance=None)},
        specs=specs,
    )

    priors = problem.build_priors("uniform")

    assert priors.names == ["charge A", "log noise qoi"]
    assert problem.nuisances == ["qoi"]
    assert problem.parameter_bounds.shape == (1, 2)


def test_learning_problem_rejects_specs_dimension_mismatch() -> None:
    specs = Specs(
        {"bounds": {f"sigma {c}": [0.0, 1.0] for c in "ABC"}, "charge_constraints": []}
    )

    with pytest.raises(ValueError, match="disagree"):
        LearningProblem({"qoi": FakeCommittee(n_inputs=2)}, specs=specs)


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
    assert priors.names == [
        "lengthscale_0", "lengthscale_1", "amplitude", "noise_variance"
    ]
    assert priors.means == pytest.approx(expected_centers)
    assert priors.scales == pytest.approx([2.0, 2.0, 2.0, 3.0])


def test_default_lgp_hyperpriors_are_finite_for_constant_data() -> None:
    priors = _default_lgp_hyperpriors(
        torch.ones((4, 2)),
        torch.zeros((4, 1)),
    )

    assert np.all(np.isfinite(priors.means))
    assert np.exp(priors.means) == pytest.approx([1.0, 1.0, 1.0, 0.1])


def test_saved_model_is_reused_only_for_the_same_data_and_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = QoIDataset("pmf", [[0.0], [1.0]], [[1.0], [2.0]], [1.5])
    second = QoIDataset("pmf", [[0.0], [1.0]], [[1.0], [3.0]], [1.5])
    member = LocalGaussianProcess(
        torch.tensor(first.X, dtype=torch.float32),
        torch.tensor(first.y, dtype=torch.float32),
        0.0,
        torch.ones(1),
        1.0,
        0.1,
    )
    model = LGPCommittee(
        [member],
        first.y_ref,
        n_curves=1,
        dataset_fingerprint=first.fingerprint(),
        mean_spec="data",
    )
    model.test_error = 1.0
    fn_model = tmp_path / "pmf.lgp"
    model.write(fn_model)
    refitted = []
    monkeypatch.setattr(
        fit_module,
        "fit_lgp_committee",
        lambda **kwargs: refitted.append(kwargs["mean"]) or model,
    )

    def fit(dataset, mean="data"):
        return fit_surrogates(
            [dataset], means={"pmf": mean}, model_paths={"pmf": fn_model},
        )

    fit(first)
    assert refitted == []
    fit(second)
    fit(first, mean=0.0)
    assert refitted == ["data", 0.0]


def test_qoi_likelihood_batches_shrink_after_cuda_oom(monkeypatch) -> None:
    from types import SimpleNamespace

    real_as_tensor = torch.as_tensor
    attempted_batch_sizes = []
    cache_clears = []

    def cpu_as_tensor(values, **kwargs):
        kwargs.pop("device", None)
        return real_as_tensor(values, **kwargs)

    def fake_log_likelihood(theta, problem):
        attempted_batch_sizes.append(len(theta))
        if len(theta) > 2:
            raise torch.OutOfMemoryError("synthetic CUDA OOM")
        return {"qoi": theta[:, 0]}

    monkeypatch.setattr(torch, "as_tensor", cpu_as_tensor)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(True))
    monkeypatch.setattr(
        "bff.bayes.learning.gaussian_log_likelihood_by_qoi", fake_log_likelihood
    )
    problem = SimpleNamespace(to_torch=lambda device: problem)

    result = LearningProblem.log_likelihood_by_qoi(
        problem, np.arange(10, dtype=float).reshape(5, 2), "cuda", batch_size=5
    )

    assert attempted_batch_sizes == [5, 2, 2, 1]
    assert cache_clears == [True]
    assert result["qoi"].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_learning_warns_when_bounds_extend_beyond_the_training_samples() -> None:
    class Recorder:
        messages: list[str] = []

        def warn(self, message: str, **kwargs) -> None:
            self.messages.append(message)

    specs = Specs(
        {
            "bounds": {"sigma A": [0.0, 1.0], "sigma B": [0.0, 1.0]},
            "charge_constraints": [],
        }
    )
    committee = FakeCommittee(shape=(4, 2), n_inputs=2)
    # Samples cover sigma A fully but only [0.4, 0.5] of sigma B.
    committee.members[0].X_train = torch.tensor(
        [[0.0, 0.4], [1.0, 0.5], [0.5, 0.45], [0.2, 0.42]]
    )
    problem = LearningProblem({"qoi": committee}, specs=specs)
    logger = Recorder()

    problem.warn_if_bounds_exceed_training_samples(logger)

    assert len(logger.messages) == 1
    assert "sigma B" in logger.messages[0]
    assert "sigma A" not in logger.messages[0]
