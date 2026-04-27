from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from bff.workflows.analyze.config import AnalyzeConfig
from bff.workflows.build.config import BuildConfig
from bff.workflows.fit.config import FitConfig
from bff.workflows.learn.config import LearnConfig
from bff.workflows.md.config import MDJobConfig
from bff.workflows.sample.config import SampleConfig
from bff.workflows.validate.config import ValidateConfig


def _write(path: Path, text: str = "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_prepared_system(asset_dir: Path, stem: str = "window-000") -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    _write(asset_dir / f"{stem}.top", "; topology\n")
    _write(asset_dir / f"{stem}.gro", "dummy gro\n")
    _write(asset_dir / f"{stem}.em.mdp", "integrator = steep\n")
    _write(asset_dir / f"{stem}.mdp", "integrator = md\n")
    _write(asset_dir / f"{stem}.ndx", "[ System ]\n1\n")
    return asset_dir


def test_build_config_loads_minimal_config(tmp_path: Path) -> None:
    topology = _write(tmp_path / "system.top")
    template = _write(tmp_path / "template.gro")
    mdp_em = _write(tmp_path / "em.mdp")
    mdp_npt = _write(tmp_path / "npt.mdp")
    mdp_prod = _write(tmp_path / "prod.mdp")

    fn_config = tmp_path / "build.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "project": "./project",
                "gromacs": {"command": "gmx"},
                "systems": [
                    {
                        "topology": str(topology),
                        "templates": {"ACE": str(template)},
                        "charge": -1,
                        "multiplicity": 1,
                        "mdp": {
                            "em": str(mdp_em),
                            "npt": str(mdp_npt),
                            "prod": str(mdp_prod),
                        },
                    }
                ],
            }
        )
    )

    config = BuildConfig.load(fn_config)

    assert config.gmx_cmd == "gmx"
    assert config.project_dir == (tmp_path / "project").resolve()
    assert len(config.systems) == 1


def test_analyze_config_loads_minimal_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    coord = _write(tmp_path / "system.gro")
    topol = _write(tmp_path / "system.top")
    trj = _write(tmp_path / "traj.xtc")

    monkeypatch.setattr(
        "bff.workflows.analyze.config.normalize_routine_list",
        lambda routines, base_dir=None: ("dummy-routine",),
    )
    monkeypatch.setattr(
        "bff.workflows.analyze.config.normalize_analysis_runtime_config",
        lambda raw: SimpleNamespace(in_memory=False, gc_collect=False),
    )

    fn_config = tmp_path / "analyze.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "sample": {"dir": str(sample_dir)},
                "reference": {
                    "systems": [
                        {
                            "coordinates": str(coord),
                            "topology": str(topol),
                            "trajectory": str(trj),
                            "routines": [{"name": "dummy"}],
                        }
                    ]
                },
            }
        )
    )

    config = AnalyzeConfig.load(fn_config)

    assert config.sample.dir == sample_dir.resolve()
    assert len(config.reference.systems) == 1
    assert config.reference.systems[0].routines == ("dummy-routine",)


def test_fit_config_loads_minimal_config(tmp_path: Path) -> None:
    data = _write(tmp_path / "dataset.pt")

    fn_config = tmp_path / "fit.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {"rdf": {"data": str(data)}},
                "fit": {"model_dir": "./models", "device": "cpu"},
            }
        )
    )

    config = FitConfig.load(fn_config)

    assert config.fit.model_dir == (tmp_path / "models").resolve()
    assert config.datasets[0].name == "rdf"
    assert config.datasets[0].fn_model == (tmp_path / "models" / "rdf.lgp").resolve()


def test_learn_config_loads_minimal_config(tmp_path: Path) -> None:
    specs = _write(tmp_path / "specs.yaml")
    model = _write(tmp_path / "model.lgp")

    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {"rdf": str(model)},
                "mcmc": {"posterior": "./posterior.pt", "device": "cpu"},
            }
        )
    )

    config = LearnConfig.load(fn_config)

    assert config.specs == specs.resolve()
    assert config.models["rdf"] == model.resolve()
    assert config.mcmc.posterior == (tmp_path / "posterior.pt").resolve()


def test_md_job_config_loads_minimal_config(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    specs = _write(tmp_path / "specs.yaml")
    topol = _write(tmp_path / "system.top")
    coord = _write(tmp_path / "system.gro")
    mdp_prod = _write(tmp_path / "prod.mdp")
    ndx = _write(tmp_path / "system.ndx")

    fn_config = tmp_path / "md.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "sample_id": "00",
                "params": [0.1, 0.2],
                "campaign_dir": str(campaign_dir),
                "fn_specs": str(specs),
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
                "systems": [
                    {
                        "topology": str(topol),
                        "coordinates": str(coord),
                        "mdp": {"prod": str(mdp_prod)},
                        "index": str(ndx),
                        "n_steps": 1000,
                    }
                ],
            }
        )
    )

    config = MDJobConfig.load(fn_config)

    assert config.sample_id == "00"
    assert config.store == ("xtc",)
    assert len(config.systems) == 1


def test_sample_config_loads_prepared_assets(tmp_path: Path) -> None:
    assets = _make_prepared_system(tmp_path / "assets")

    fn_config = tmp_path / "sample.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "mol_resname": "ACE",
                "campaign_dir": "./campaign",
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
                "systems": [{"assets": str(assets), "n_steps": 1000}],
                "bounds": {"charge C1": [-1.0, 1.0]},
                "total_charge": 0.0,
                "implicit_atoms": ["C1"],
                "n_samples": 4,
            }
        )
    )

    config = SampleConfig.load(fn_config)

    assert config.mol_resname == "ACE"
    assert config.n_samples == 4
    assert len(config.systems) == 1


def test_validate_config_loads_prepared_assets(tmp_path: Path) -> None:
    assets = _make_prepared_system(tmp_path / "assets")
    specs = _write(tmp_path / "specs.yaml")
    params = _write(tmp_path / "posterior.pt")

    fn_config = tmp_path / "validate.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "campaign_dir": "./campaign",
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
                "systems": [{"assets": str(assets), "n_steps": 1000}],
                "specs": str(specs),
                "parameters": str(params),
            }
        )
    )

    config = ValidateConfig.load(fn_config)

    assert config.specs == specs.resolve()
    assert config.parameters == params.resolve()
    assert len(config.systems) == 1
