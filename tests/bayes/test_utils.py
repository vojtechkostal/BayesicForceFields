import torch

from bff.bayes.utils import find_map, find_max_stable_lr, initialize_walkers


def test_find_map_returns_best_observed_iterate(monkeypatch) -> None:
    monkeypatch.setattr(
        "bff.bayes.utils.find_max_stable_lr",
        lambda *args, **kwargs: 2.2,
    )

    result = find_map(
        lambda x: -torch.sum((x - 1.0) ** 2),
        torch.tensor([0.0]),
        max_iter=2,
        tol_grad=0.0,
    )

    assert torch.allclose(result, torch.tensor([0.0]))


def test_find_map_accepts_start_outside_default_stability_bounds() -> None:
    result = find_map(
        lambda x: -torch.sum((x + 12.0) ** 2),
        torch.tensor([-12.0]),
        lr=0.01,
    )

    assert torch.allclose(result, torch.tensor([-12.0]), atol=1e-2)


def test_initialize_walkers_supports_uniform_priors_and_constraints() -> None:
    priors = [
        torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0)),
        torch.distributions.Normal(torch.tensor(5.0), torch.tensor(0.1)),
    ]
    walkers = initialize_walkers(priors, 50)
    assert walkers.shape == (50, 2)
    assert ((walkers[:, 0] >= 0) & (walkers[:, 0] <= 1)).all()

    class AboveHalf:
        n_params = 1

        def __call__(self, values):
            return values[:, 0] > 0.5

    constrained = initialize_walkers(priors, 20, AboveHalf())
    assert constrained.shape == (20, 2)
    assert (constrained[:, 0] > 0.5).all()


def test_learning_rate_search_rejects_runs_that_diverge_to_nan() -> None:
    def objective(x):
        return torch.where(x.sum() > 0.5, torch.nan, 1.0) * x.sum() ** 2

    assert find_max_stable_lr(objective, torch.tensor([0.4]), [1.0]) is None
