from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from bff.bayes.priors import Prior, Priors
from bff.bayes.results import Results, marginal_mode
from bff.domain.specs import Specs

SPECS = {
    "bounds": {
        "charge A": [-1.0, 1.0],
        "charge B": [-0.5, 0.5],
        "sigma C": [0.1, 2.0],
    },
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


def make_results(n_saved: int = 400, n_walkers: int = 4, seed: int = 0) -> Results:
    """Posterior of charge A ~ N(0.2, 0.05) and sigma C ~ N(1, 0.1), with the
    noise of one QoI at about exp(-3); the MAP is the most probable state."""
    rng = np.random.default_rng(seed)
    chain = np.stack(
        [
            rng.normal(0.2, 0.05, (n_saved, n_walkers)),
            rng.normal(1.0, 0.1, (n_saved, n_walkers)),
            rng.normal(-3.0, 0.1, (n_saved, n_walkers)),
        ],
        axis=-1,
    )
    log_prob = -(
        ((chain[..., 0] - 0.2) / 0.05) ** 2 + ((chain[..., 1] - 1.0) / 0.1) ** 2
    )
    prior = Priors(
        [
            Prior("normal", 0.0, 0.5, "charge A"),
            Prior("normal", 1.0, 0.5, "sigma C"),
            Prior("normal", -2.0, 2.0, "log noise rdf"),
        ]
    )
    return Results(
        chain,
        log_prob,
        Specs(SPECS),
        prior=prior,
        nuisances=["rdf"],
        qoi={"rdf": {"n_eff": 5.0, "tolerance": 0.1, "nuisance": None}},
        info={"mcmc": {"converged": True}},
    )


def test_results_validates_shapes() -> None:
    results = make_results()
    with pytest.raises(ValueError, match="chain must have shape"):
        Results(results.chain[..., :1], results.log_prob, Specs(SPECS))
    with pytest.raises(ValueError, match="log_prob must have shape"):
        Results(results.chain, results.log_prob[:-1], Specs(SPECS), nuisances=["rdf"])
    with pytest.raises(ValueError, match="prior has 1 entries"):
        Results(
            results.chain,
            results.log_prob,
            Specs(SPECS),
            prior=Priors([Prior("normal", 0, 1)]),
            nuisances=["rdf"],
        )


def test_samples_are_in_physical_units_with_implicit_charges() -> None:
    results = make_results()

    assert results.names == ("charge A", "charge B", "sigma C", "noise rdf")
    assert results.explicit_names == ("charge A", "sigma C")
    assert results.implicit_names == ("charge B",)
    assert results.nuisance_names == ("noise rdf",)
    assert results.samples.shape == (1600, 4)
    np.testing.assert_allclose(results["charge B"], -results["charge A"])
    np.testing.assert_allclose(
        results["noise rdf"], np.exp(results.chain[..., 2]).ravel()
    )
    with pytest.raises(KeyError, match="not one of"):
        results["charge Z"]


def test_map_is_the_state_of_highest_log_probability() -> None:
    results = make_results()
    flat = results.log_prob.reshape(-1)

    best = results.samples[int(np.argmax(flat))]

    assert results.map == dict(zip(results.names, best))
    assert results.map["charge B"] == pytest.approx(-results.map["charge A"])
    assert abs(results.map["charge A"] - 0.2) < 0.02
    assert abs(results.map["sigma C"] - 1.0) < 0.05


def test_summary_reports_mean_quantiles_mode_and_map() -> None:
    results = make_results()

    summary = results.summary()

    assert list(summary) == list(results.names)
    entry = summary["charge A"]
    assert set(entry) == {"mean", "std", "median", "q16", "q84", "mode", "map"}
    assert entry["mean"] == pytest.approx(0.2, abs=0.01)
    assert entry["std"] == pytest.approx(0.05, abs=0.01)
    assert entry["q16"] < entry["median"] < entry["q84"]
    assert entry["mode"] == pytest.approx(0.2, abs=0.03)
    assert entry["map"] == results.map["charge A"]
    assert summary["noise rdf"]["mean"] == pytest.approx(np.exp(-3.0), rel=0.1)


def test_write_summary_round_trips_through_yaml(tmp_path: Path) -> None:
    results = make_results()
    results.write_summary(tmp_path / "summary.yaml")

    loaded = yaml.safe_load((tmp_path / "summary.yaml").read_text())

    assert loaded["charge A"]["map"] == pytest.approx(results.map["charge A"])


def test_marginal_mode_handles_bounds_and_constant_samples() -> None:
    samples = np.random.default_rng(0).normal(0.9, 0.3, 2000)
    assert marginal_mode(samples, 0.0, 1.0) <= 1.0
    assert marginal_mode(np.full(100, 0.7), 0.0, 1.0) == pytest.approx(0.7)


def test_diagnostics_cover_every_chain_column() -> None:
    diagnostics = make_results().diagnostics()

    assert list(diagnostics) == ["charge A", "sigma C", "log noise rdf", "log_prob"]
    assert all(
        entry["rhat"] == pytest.approx(1.0, abs=0.05)
        for entry in diagnostics.values()
    )
    assert all(entry["ess_bulk"] > 100 for entry in diagnostics.values())
    assert all(entry["ess_tail"] > 100 for entry in diagnostics.values())


def test_prior_density_of_sampled_parameters_only() -> None:
    results = make_results()
    grid = np.linspace(-1, 1, 5)

    density = results.prior_density("charge A", grid)

    assert density.shape == (5,)
    assert density.argmax() == 2
    assert results.prior_density("charge B", grid) is None
    assert Results(
        results.chain, results.log_prob, Specs(SPECS), nuisances=["rdf"]
    ).prior_density("charge A", grid) is None


@pytest.mark.parametrize("distribution", ["empirical", "kde", "normal", "uniform"])
def test_draw_returns_named_valid_parameter_sets(distribution: str) -> None:
    results = make_results()

    draws = results.draw(25, distribution=distribution, seed=3)

    assert list(draws) == ["charge A", "sigma C"]
    assert all(len(values) == 25 for values in draws.values())
    assert results.specs.is_valid(np.column_stack(list(draws.values()))).all()
    again = results.draw(25, distribution=distribution, seed=3)
    np.testing.assert_allclose(draws["charge A"], again["charge A"])


def test_draw_prepends_mean_and_map_and_adds_implicit_charges() -> None:
    results = make_results()

    draws = results.draw(
        3, seed=1, include_mean=True, include_map=True, implicit=True
    )

    assert list(draws) == ["charge A", "charge B", "sigma C"]
    assert draws["charge A"].shape == (5,)
    assert draws["charge A"][0] == pytest.approx(results["charge A"].mean())
    assert draws["charge A"][1] == pytest.approx(results.map["charge A"])
    np.testing.assert_allclose(draws["charge B"], -draws["charge A"])
    only_special = results.draw(0, include_map=True)
    assert only_special["sigma C"].tolist() == [results.map["sigma C"]]


def test_draw_rejects_empty_requests_and_bad_settings() -> None:
    results = make_results()
    with pytest.raises(ValueError, match="Nothing to draw"):
        results.draw(0)
    with pytest.raises(ValueError, match="confidence"):
        results.draw(1, confidence=1.0)
    with pytest.raises(ValueError, match="distribution"):
        results.draw(1, distribution="beta")


def test_draw_redraws_values_outside_the_bounds() -> None:
    # charge B = -charge A must stay within +-0.5, so |charge A| <= 0.5.
    results = make_results()
    chain = results.chain.copy()
    chain[..., 0] += 0.3
    wide = Results(chain, results.log_prob, Specs(SPECS), nuisances=["rdf"])

    draws = wide.draw(200, distribution="normal", seed=0)

    assert np.abs(draws["charge A"]).max() <= 0.5


def test_draw_rejects_a_mean_outside_the_bounds() -> None:
    results = make_results()
    chain = results.chain.copy()
    chain[..., 0] = 0.95
    outside = Results(chain, results.log_prob, Specs(SPECS), nuisances=["rdf"])
    with pytest.raises(ValueError, match="mean violates"):
        outside.draw(0, include_mean=True)


def test_draw_exports_yaml_readable_by_explicit_validation(tmp_path: Path) -> None:
    results = make_results()
    fn_out = tmp_path / "draws.yaml"

    draws = results.draw(4, seed=2, fn_out=fn_out)

    loaded = yaml.safe_load(fn_out.read_text())
    assert loaded["charge A"] == pytest.approx(draws["charge A"].tolist())
    with pytest.raises(FileExistsError):
        results.draw(4, fn_out=fn_out)
    results.draw(4, fn_out=fn_out, overwrite=True)


def test_results_file_round_trip_keeps_everything(tmp_path: Path) -> None:
    results = make_results()
    results = Results(
        results.chain,
        results.log_prob,
        results.specs,
        prior=results.prior,
        nuisances=results.nuisances,
        qoi=results.qoi,
        qoi_index=np.array([0, 10, 20]),
        qoi_log_likelihood={"rdf": np.array([-1.0, -2.0, -3.0])},
        info=results.info,
    )
    fn_results = tmp_path / "results.pt"
    results.save(fn_results)

    loaded = Results.load(fn_results)

    np.testing.assert_allclose(loaded.samples, results.samples, rtol=1e-5)
    assert loaded.map == pytest.approx(results.map, rel=1e-5)
    assert loaded.specs == results.specs
    assert loaded.prior.to_dicts() == results.prior.to_dicts()
    assert loaded.nuisances == ("rdf",)
    assert loaded.qoi == results.qoi
    assert loaded.qoi_index.tolist() == [0, 10, 20]
    assert loaded.qoi_log_likelihood["rdf"].tolist() == [-1.0, -2.0, -3.0]
    assert loaded.info == {"mcmc": {"converged": True}}


def test_results_rejects_files_of_other_formats(tmp_path: Path) -> None:
    torch.save({"posterior": torch.zeros(3, 2, 1)}, tmp_path / "posterior.pt")

    with pytest.raises(ValueError, match="not a BFF results file"):
        Results.load(tmp_path / "posterior.pt")
