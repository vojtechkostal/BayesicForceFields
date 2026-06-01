import torch

from bff.bayes.utils import find_map


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
