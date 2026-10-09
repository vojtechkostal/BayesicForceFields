from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import bff.workflows.validate.main as validate_module
from bff.bayes.results import Results
from bff.domain.specs import Specs
from bff.qoi.dataset import QoIDataset
from bff.workflows.build_qoi_datasets.main import main as build_qoi_datasets_main
from bff.workflows.campaign import run as run_module

ROOT = Path(__file__).parents[2]
ACE_TOP = ROOT / "examples/acetate/inputs/topol.top"
SPECS = {"bounds": {"charge C2": [0.0, 1.0]}, "charge_constraints": []}


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _write_config(tmp_path: Path, **mode) -> Path:
    inputs = {
        "topology": str(ACE_TOP),
        "coordinates": str(_write(tmp_path / "system.gro")),
        "mdp_production": str(_write(tmp_path / "production.mdp")),
        "index": str(_write(tmp_path / "index.ndx")),
    }
    path = tmp_path / "validate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "campaign_dir": "./campaign",
                "systems": [{"system_id": "acetate", "inputs": inputs, "n_steps": 10}],
                "store": ["pmf"],
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
                **mode,
            }
        )
    )
    return path


def _explicit_config(tmp_path: Path, **options) -> Path:
    _write(tmp_path / "specs.yaml", yaml.safe_dump(SPECS))
    _write(
        tmp_path / "parameters.yaml", yaml.safe_dump({"charge C2": [0.2, 0.5, 0.8]})
    )
    return _write_config(
        tmp_path, specs="specs.yaml", parameters="parameters.yaml", **options
    )


def _fake_md(monkeypatch) -> list[str]:
    """Replace `bff md` by a job writing one PMF value (the parameter) per sample."""
    runs: list[str] = []

    def fake_run(command, **kwargs):
        campaign_dir, sample_id = Path(command[-2]).parent, command[-1]
        params = yaml.safe_load((campaign_dir / "samples.yaml").read_text())[
            "samples"
        ][sample_id]["params"]
        sample_dir = campaign_dir / "samples" / sample_id
        runs.append(sample_id)
        pmf = _write(sample_dir / "acetate" / "production.pmf", f"{params[0]}\n")
        _write(
            sample_dir / "result.yaml",
            yaml.safe_dump(
                {
                    "status": "completed",
                    "outputs": {
                        "acetate": {"pmf": str(pmf.relative_to(campaign_dir))}
                    },
                }
            ),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)
    return runs


def test_validation_campaign_can_be_analyzed_by_build_qoi_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_md(monkeypatch)
    validate_module.main(_explicit_config(tmp_path))

    campaign = tmp_path / "campaign"
    assert Specs(campaign / "specs.yaml").to_dict() == Specs(SPECS).to_dict()
    manifest = yaml.safe_load((campaign / "samples.yaml").read_text())
    assert manifest["provenance"]["source"] == "parameters"
    assert len(manifest["provenance"]["parameters_sha256"]) == 64

    routine = _write(
        tmp_path / "pmf.py",
        "from bff.qoi import QoI\n"
        "def load(*, inputs, options):\n"
        "    return QoI('pmf', [float(inputs['pmf'].read_text())])\n",
    )
    qoi_config = tmp_path / "qoi.yaml"
    qoi_config.write_text(
        yaml.safe_dump(
            {
                "training_samples": {
                    "manifest": str(campaign / "samples.yaml"),
                    "systems": [{"system_id": "acetate"}],
                    "workers": 1,
                },
                "reference": {
                    "systems": [
                        {
                            "system_id": "acetate",
                            "inputs": {"pmf": str(_write(tmp_path / "ref.pmf", "0.5"))},
                        }
                    ]
                },
                "routines": [
                    {
                        "name": "pmf",
                        "callable": f"{routine}:load",
                        "systems": ["acetate"],
                        "inputs": ["pmf"],
                    }
                ],
            }
        )
    )
    build_qoi_datasets_main(qoi_config)

    dataset = QoIDataset.load(tmp_path / "qoi" / "pmf.pt")
    np.testing.assert_allclose(np.asarray(dataset.X).ravel(), [0.2, 0.5, 0.8])
    np.testing.assert_allclose(np.asarray(dataset.y).ravel(), [0.2, 0.5, 0.8])


def test_rerun_requires_resume_or_overwrite(tmp_path: Path, monkeypatch) -> None:
    runs = _fake_md(monkeypatch)
    validate_module.main(_explicit_config(tmp_path))
    assert runs == ["0", "1", "2"]

    with pytest.raises(FileExistsError, match="resume: true .* overwrite: true"):
        validate_module.main(_explicit_config(tmp_path))

    # Resume reruns only samples that did not complete.
    manifest_path = tmp_path / "campaign" / "samples.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["samples"]["1"]["status"] = "incomplete"
    manifest_path.write_text(yaml.safe_dump(manifest))
    runs.clear()
    validate_module.main(_explicit_config(tmp_path, resume=True))
    assert runs == ["1"]
    manifest = yaml.safe_load(manifest_path.read_text())
    assert {record["status"] for record in manifest["samples"].values()} == {
        "completed"
    }
    assert manifest["samples"]["0"]["outputs"]["acetate"]["pmf"].endswith(".pmf")

    runs.clear()
    validate_module.main(_explicit_config(tmp_path, overwrite=True))
    assert runs == ["0", "1", "2"]


def test_explicit_parameters_are_checked_before_staging(
    tmp_path: Path, monkeypatch
) -> None:
    runs = _fake_md(monkeypatch)
    config = _explicit_config(tmp_path)
    _write(tmp_path / "parameters.yaml", yaml.safe_dump({"charge C2": [0.5, 1.5]}))
    with pytest.raises(ValueError, match="1 of 2 parameter samples"):
        validate_module.main(config)
    assert runs == []
    assert not (tmp_path / "campaign" / "samples").exists()

    _write(tmp_path / "parameters.yaml", yaml.safe_dump({"charge C1": [0.5]}))
    with pytest.raises(ValueError, match=r"missing \['charge C2'\].*'charge C1'"):
        validate_module.main(config)


def _save_results(tmp_path: Path) -> Path:
    """Posterior of charge C2 with samples 0.2, 0.4, 0.6; the MAP is 0.2."""
    results = Results(
        np.array([[[0.2]], [[0.4]], [[0.6]]]),
        np.array([[0.0], [-1.0], [-2.0]]),
        Specs(SPECS),
    )
    fn_results = tmp_path / "results.pt"
    results.save(fn_results)
    return fn_results


def test_posterior_draws_round_trip_through_a_parameter_file(tmp_path: Path) -> None:
    results = Results.load(_save_results(tmp_path))
    fn_out = tmp_path / "draws.yaml"
    draws = results.draw(4, distribution="empirical", seed=1, fn_out=fn_out)

    loaded = validate_module.load_parameter_samples(fn_out, Specs(SPECS))

    np.testing.assert_allclose(loaded[:, 0], draws["charge C2"])


def test_posterior_mode_records_its_seed_and_source(
    tmp_path: Path, monkeypatch
) -> None:
    captured = {}

    def fake_run_campaign(config, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(validate_module, "run_campaign", fake_run_campaign)
    fn_results = _save_results(tmp_path)
    validate_module.main(
        _write_config(
            tmp_path,
            posterior={
                "file": str(fn_results),
                "n_samples": 2,
                "include_mean": True,
                "include_map": True,
                "distribution": "empirical",
            },
        )
    )

    samples, provenance = captured["draw"]()
    assert samples.shape == (4, 1)
    np.testing.assert_allclose(samples[:2, 0], [0.4, 0.2])  # mean, then MAP
    assert provenance["source"] == "posterior"
    assert provenance["include_map"] is True
    assert len(provenance["results_sha256"]) == 64
    assert isinstance(provenance["seed"], int)
    assert captured["specs"].to_dict() == Specs(SPECS).to_dict()


def test_validate_rejects_files_that_are_not_results(
    tmp_path: Path, monkeypatch
) -> None:
    fn_old = tmp_path / "posterior.pt"
    torch.save({"posterior": torch.zeros(3, 2, 1)}, fn_old)
    config = _write_config(tmp_path, posterior={"file": str(fn_old)})

    with pytest.raises(ValueError, match="not a BFF results file"):
        validate_module.main(config)


def test_invalid_parameters_do_not_overwrite_a_campaign(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_md(monkeypatch)
    validate_module.main(_explicit_config(tmp_path))
    config = _explicit_config(tmp_path, overwrite=True)
    _write(tmp_path / "parameters.yaml", yaml.safe_dump({"charge C2": [1.5]}))

    with pytest.raises(ValueError, match="violate the parameter bounds"):
        validate_module.main(config)

    samples = sorted(p.name for p in (tmp_path / "campaign" / "samples").iterdir())
    assert samples == ["0", "1", "2"]
