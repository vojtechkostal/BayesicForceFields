from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

import bff.workflows.validate.main as validate_module
from bff.bayes.results import PosteriorResults
from bff.domain.specs import Specs


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _write_config(tmp_path: Path, posterior: Path) -> Path:
    inputs = {
        "topology": str(_write(tmp_path / "system.top")),
        "coordinates": str(_write(tmp_path / "system.gro")),
        "mdp_production": str(_write(tmp_path / "production.mdp")),
        "index": str(_write(tmp_path / "index.ndx")),
    }
    path = tmp_path / "validate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "campaign_dir": "./campaign",
                "posterior": {
                    "file": str(posterior),
                    "n_samples": 2,
                    "include_mean": True,
                    "distribution": "empirical",
                    "confidence": 0.9,
                    "seed": 7,
                },
                "systems": [
                    {
                        "system_id": "acetate",
                        "inputs": inputs,
                        "n_steps": 10,
                    }
                ],
                "dispatch": False,
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
            }
        )
    )
    return path


def test_validate_stages_posterior_mean_draws_and_embedded_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    posterior_file = _write(tmp_path / "posterior.pt")
    specs = Specs(
        {
            "bounds": {"x": [-1.0, 1.0]},
            "charge_constraints": [],
        }
    )
    results = PosteriorResults(
        np.array([[[-0.4]], [[0.2]], [[0.5]]]),
        sample_labels=["x"],
        specs=specs,
    )
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)
    monkeypatch.setattr(
        PosteriorResults,
        "load",
        classmethod(lambda cls, path: results),
    )
    captured: dict[str, object] = {}

    def fake_stage(config, *, fn_specs):
        captured["staged_specs"] = fn_specs
        return fn_specs, []

    def fake_run_campaign(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(validate_module, "stage_campaign", fake_stage)
    monkeypatch.setattr(validate_module, "run_campaign", fake_run_campaign)

    validate_module.main(str(_write_config(tmp_path, posterior_file)))

    campaign_specs = tmp_path / "campaign" / "specs.yaml"
    assert campaign_specs.is_file()
    assert Specs.load(campaign_specs) == specs
    assert captured["staged_specs"] == campaign_specs
    samples = captured["parameter_samples"]
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (3, 1)
    np.testing.assert_allclose(samples[0], [0.1])
    assert (tmp_path / "campaign" / "validate.log").is_file()


def test_validate_rejects_posterior_without_embedded_specs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    posterior_file = _write(tmp_path / "posterior.pt")
    results = PosteriorResults(
        np.zeros((3, 2, 1)),
        sample_labels=["x"],
    )
    monkeypatch.setattr(
        PosteriorResults,
        "load",
        classmethod(lambda cls, path: results),
    )

    try:
        validate_module.main(str(_write_config(tmp_path, posterior_file)))
    except ValueError as error:
        assert "embedded" in str(error)
    else:
        raise AssertionError("Expected missing embedded specifications to fail.")
