from pathlib import Path

import numpy as np
import pytest
import torch

from bff.bayes.priors import Prior, Priors
from bff.bayes.results import PosteriorResults
from bff.domain.specs import Specs


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
    torch.save({
        "posterior": posterior,
        "metadata": {"bff_version": "test"},
    }, path)
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
    assert results.metadata["bff_version"] == "test"
    assert results.preparation_info["discard"] == 1
    assert results.preparation_info["thin"] == 2


def test_posterior_results_loads_enriched_payload(tmp_path: Path) -> None:
    path = tmp_path / "posterior.pt"
    specs = Specs(
        {
            "bounds": {"charge A": [-1.0, 1.0], "charge B": [-1.0, 1.0]},
            "charge_constraints": [
                {
                    "selection": "name A B",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B",
                    "coefficients": {"charge A": 1.0, "charge B": 1.0},
                }
            ],
        }
    )
    torch.save(
        {
            "posterior": torch.zeros((3, 2, 1)),
            "priors": [Prior("normal", 0.0, 1.0, name="charge A").to_dict()],
            "specs": specs.to_dict(),
            "sample_labels": ["charge A"],
        },
        path,
    )

    results = PosteriorResults.load(path)

    assert results.specs == specs
    assert results.sample_labels == ["charge A"]
    assert results.priors.names == ["charge A"]


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


def test_posterior_results_expands_multiple_implicit_charges() -> None:
    specs = Specs(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "charge B": [-1.0, 1.0],
                "charge C": [0.0, 2.0],
            },
            "charge_constraints": [
                {
                    "selection": "name A B C",
                    "target": 1.0,
                    "scope": "residue",
                    "implicit": "charge C",
                    "coefficients": {
                        "charge A": 1.0,
                        "charge B": 1.0,
                        "charge C": 1.0,
                    },
                },
                {
                    "selection": "name A B",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B",
                    "coefficients": {"charge A": 1.0, "charge B": 1.0},
                },
            ],
        }
    )
    results = PosteriorResults(
        np.full((3, 2, 1), 0.2),
        sample_labels=["charge A"],
        specs=specs,
        include_implicit_charge=True,
    )

    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    assert results.labels == ["charge A", "charge B", "charge C"]
    np.testing.assert_allclose(results.prepared_samples[0], [0.2, -0.2, 1.0])


def test_sample_posterior_kde_draws_arbitrary_number() -> None:
    rng = np.random.default_rng(1)
    posterior = rng.normal(size=(20, 4, 2))
    results = PosteriorResults(posterior, sample_labels=["x", "y"])
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    draws = results.sample_posterior(
        n_samples=250,
        distribution="kde",
        random_state=2,
    )

    assert draws.shape == (250, 2)
    assert np.all(np.isfinite(draws))


def test_sample_posterior_empirical_resamples_prepared_rows() -> None:
    posterior = np.arange(12, dtype=float).reshape(6, 2, 1)
    results = PosteriorResults(posterior, sample_labels=["x"])
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    draws = results.sample_posterior(
        n_samples=20,
        distribution="empirical",
        random_state=2,
    )

    assert draws.shape == (20, 1)
    assert set(draws.ravel()) <= set(results.prepared_samples.ravel())


def test_sample_posterior_normal_preserves_multidimensional_correlation() -> None:
    rng = np.random.default_rng(3)
    cov = np.array([[1.0, 0.85], [0.85, 1.0]])
    posterior = rng.multivariate_normal([0.0, 0.0], cov, size=3000).reshape(1000, 3, 2)
    results = PosteriorResults(posterior, sample_labels=["x", "y"])
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    draws = results.sample_posterior(
        n_samples=3000,
        distribution="normal",
        random_state=4,
    )

    assert draws.shape == (3000, 2)
    assert np.corrcoef(draws, rowvar=False)[0, 1] > 0.75


def test_sample_posterior_uses_positional_specs_and_exports_yaml(
    tmp_path: Path,
) -> None:
    specs = Specs(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "charge B": [-1.0, 1.0],
            },
            "charge_constraints": [
                {
                    "selection": "name A B",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B",
                    "coefficients": {"charge A": 1.0, "charge B": 1.0},
                },
            ],
        }
    )
    specs_path = tmp_path / "specs.yaml"
    specs.write(specs_path)
    priors = Priors(
        [
            Prior("normal", 0.0, 0.2, name="charge A"),
            Prior("normal", -2.0, 0.1, name="log_sigma_rdf"),
        ]
    )
    charge = np.linspace(-0.5, 0.5, 20)
    log_sigma = np.full_like(charge, -2.0)
    posterior = np.column_stack([charge, log_sigma]).reshape(10, 2, 2)
    results = PosteriorResults(posterior, priors, specs_path)
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)
    out = tmp_path / "draws.yaml"

    draws = results.sample_posterior(
        n_samples=25,
        distribution="uniform",
        fn_out=out,
        random_state=5,
    )

    assert draws.shape == (25, 1)
    assert results.specs == specs
    assert results.sample_labels == ["charge A", "$\\sigma_{\\mathrm{rdf}}$"]
    assert out.read_text().startswith("charge A:")


def test_sample_posterior_can_return_implicit_charges_satisfying_specs() -> None:
    specs = Specs(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "charge B": [-1.0, 1.0],
                "charge C": [0.0, 2.0],
            },
            "charge_constraints": [
                {
                    "selection": "name A B C",
                    "target": 1.0,
                    "scope": "residue",
                    "implicit": "charge C",
                    "coefficients": {
                        "charge A": 1.0,
                        "charge B": 1.0,
                        "charge C": 1.0,
                    },
                },
                {
                    "selection": "name A B",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B",
                    "coefficients": {"charge A": 1.0, "charge B": 1.0},
                },
            ],
        }
    )
    priors = Priors(
        [
            Prior("normal", 0.0, 0.2, name="charge A"),
            Prior("normal", -2.0, 0.1, name="log_sigma_rdf"),
        ]
    )
    charge = np.linspace(-0.5, 0.5, 30)
    log_sigma = np.full_like(charge, -2.0)
    posterior = np.column_stack([charge, log_sigma]).reshape(10, 3, 2)
    results = PosteriorResults(posterior, priors=priors, specs=specs)
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    draws = results.sample_posterior(
        n_samples=40,
        distribution="uniform",
        random_state=8,
        include_implicit_charge=True,
    )

    assert draws.shape == (40, 3)
    np.testing.assert_allclose(
        draws @ specs.constraint_matrix.T,
        np.tile(specs.constraint_targets, (len(draws), 1)),
    )
    lower, upper = specs.bounds.array.T
    assert np.logical_and(draws >= lower, draws <= upper).all()
