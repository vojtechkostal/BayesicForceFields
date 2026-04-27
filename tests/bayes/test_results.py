from pathlib import Path

import numpy as np
import pytest
import torch

from bff.bayes.priors import Prior, Priors
from bff.bayes.results import PosteriorResults


def test_posterior_results_requires_raw_chain_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        PosteriorResults(np.zeros((4, 2)))

    bad = tmp_path / "bad.pt"
    torch.save({"chain": torch.zeros((2, 2, 1))}, bad)
    with pytest.raises(KeyError, match="posterior"):
        PosteriorResults.load(bad)


def test_posterior_results_loads_pt_and_prepares_samples(tmp_path: Path) -> None:
    path = tmp_path / "posterior.pt"
    posterior = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
    torch.save({"posterior": posterior}, path)
    priors = Priors(
        [
            Prior("normal", 0.0, 1.0, name="x"),
            Prior("normal", 0.0, 1.0, name="log_sigma_rdf"),
        ]
    )

    results = PosteriorResults.load(path, priors=priors)
    results.prepare_samples(discard=1, thin=2, strip_outliers=False)

    assert results.has_prepared_samples
    assert results.prepared_samples.shape == (6, 2)
    assert results.labels == ["x", "$\\sigma_{\\mathrm{rdf}}$"]
    assert np.all(results.prepared_samples[:, 1] > 0)


def test_posterior_results_rejects_empty_prepared_samples() -> None:
    results = PosteriorResults(np.zeros((2, 2, 1)))

    with pytest.raises(ValueError, match="No posterior samples"):
        results.prepare_samples(discard=10, thin=1)


def test_posterior_results_summary_uses_prepared_samples() -> None:
    posterior = np.ones((4, 3, 1))
    results = PosteriorResults(posterior, sample_labels=["theta"])
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    summary = results.summary()

    assert set(summary) == {"theta", "autocorr_time"}
    assert summary["theta"]["mean"] == 1.0
