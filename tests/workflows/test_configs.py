from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bff.workflows.analyze.config import AnalyzeConfig
from bff.workflows.build.config import BuildConfig
from bff.workflows.learn.config import LearnConfig
from bff.workflows.lgpfit.config import LGPFitConfig
from bff.workflows.md.config import MDJobConfig
from bff.workflows.prepare_reference.config import PrepareReferenceConfig
from bff.workflows.sample.config import SampleConfig
from bff.workflows.validate.config import ValidateConfig


def _write(path: Path, text: str = "\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _make_prepared_system(asset_dir: Path, stem: str = "system-000") -> Path:
    asset_dir.mkdir(parents=True, exist_ok=True)
    _write(asset_dir / f"{stem}.top", "; topology\n")
    _write(asset_dir / f"{stem}.gro", "dummy gro\n")
    _write(asset_dir / f"{stem}.em.mdp", "integrator = steep\n")
    _write(asset_dir / f"{stem}.mdp", "integrator = md\n")
    _write(asset_dir / f"{stem}.ndx", "[ System ]\n1\n")
    return asset_dir


def _make_build_stage(project_dir: Path) -> Path:
    for system_id in ("000", "001"):
        system_dir = project_dir / "systems" / system_id
        for name in (
            "topology.top",
            "coordinates.gro",
            "index.ndx",
            "em.mdp",
            "npt.mdp",
            "production.mdp",
            "production.gro",
            "production.xtc",
        ):
            _write(system_dir / name, f"{name}\n")
        _write(
            system_dir / "system.yaml",
            yaml.safe_dump(
            {
                "system_name": None,
                "charge": -1,
                "multiplicity": 1,
                "box": [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
                "maxwarn": 0,
                "production_steps": 1000,
            }
            ),
        )
    return project_dir


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
                        "system_id": "acetate",
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


def test_build_config_defaults_missing_templates_to_empty_mapping(
    tmp_path: Path,
) -> None:
    topology = _write(tmp_path / "system.top")
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
                        "system_id": "acetate",
                        "topology": str(topology),
                        "charge": 0,
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

    assert config.systems[0].templates == {}


def test_prepare_reference_config_loads_build_stage(tmp_path: Path) -> None:
    source = _make_build_stage(tmp_path / "project")
    fn_config = tmp_path / "prepare-reference.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "source": str(source),
                "output": "./reference",
                "systems": ["000", "001"],
            }
        )
    )

    config = PrepareReferenceConfig.load(fn_config)

    assert config.source == source.resolve()
    assert config.output_dir == (tmp_path / "reference").resolve()
    assert len(config.systems) == 2
    assert config.systems[0].production_coordinates_path.name == "production.gro"


def test_prepare_reference_config_selects_systems(tmp_path: Path) -> None:
    source = _make_build_stage(tmp_path / "project")
    fn_config = tmp_path / "prepare-reference.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "source": str(source),
                "output": "./reference",
                "n_single_point_snapshots": 25,
                "systems": ["001"],
            }
        )
    )

    config = PrepareReferenceConfig.load(fn_config)

    assert config.n_single_point_snapshots == 25
    assert [system.system_id for system in config.systems] == ["001"]


def test_analyze_config_loads_minimal_config(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    sample_manifest = _write(
        sample_dir / "samples.yaml", "systems: []\nsamples: {}\n"
    )
    coord = _write(tmp_path / "system.gro")
    topol = _write(tmp_path / "system.top")
    trj = _write(tmp_path / "traj.xtc")

    fn_config = tmp_path / "analyze.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "training_samples": {
                    "manifest": str(sample_manifest),
                    "systems": [{"system_id": "acetate"}],
                },
                "reference": {
                    "systems": [
                        {
                            "system_id": "acetate",
                            "inputs": {
                                "coordinates": str(coord),
                                "topology": str(topol),
                                "trajectory": str(trj),
                            },
                        }
                    ]
                },
                "routines": [
                    {
                        "name": "rdf",
                        "type": "rdf",
                        "systems": ["acetate"],
                        "selections": {
                            "group_a": "name A",
                            "group_b": "name B",
                        },
                    }
                ],
            }
        )
    )

    config = AnalyzeConfig.load(fn_config)

    assert config.training_samples.manifest == sample_manifest.resolve()
    assert len(config.reference.systems) == 1
    assert config.routines[0].name == "rdf"


def test_lgpfit_config_loads_minimal_config(tmp_path: Path) -> None:
    data = _write(tmp_path / "dataset.pt")

    fn_config = tmp_path / "lgpfit.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {"rdf": {"data": str(data)}},
                "lgpfit": {"model_dir": "./models", "device": "cpu"},
            }
        )
    )

    config = LGPFitConfig.load(fn_config)

    assert config.lgpfit.model_dir == (tmp_path / "models").resolve()
    assert config.datasets[0].name == "rdf"
    assert config.datasets[0].fn_model == (tmp_path / "models" / "rdf.lgp").resolve()


def test_lgpfit_config_rejects_observation_scale(tmp_path: Path) -> None:
    data = _write(tmp_path / "dataset.pt")

    fn_config = tmp_path / "lgpfit.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "rdf": {
                        "data": str(data),
                        "observation_scale": 2.0,
                    }
                },
                "lgpfit": {"model_dir": "./models", "device": "cpu"},
            }
        )
    )

    with pytest.raises(ValueError, match="observation_scale"):
        LGPFitConfig.load(fn_config)


def test_learn_config_loads_effective_observation_modes(tmp_path: Path) -> None:
    specs = _write(tmp_path / "specs.yaml")
    model = _write(tmp_path / "model.lgp")

    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {
                    "rdf": {
                        "model_path": str(model),
                        "tolerance": 0.1,
                    },
                    "density": {
                        "model_path": str(model),
                        "independent_observations": True,
                    },
                    "pmf": {
                        "model_path": str(model),
                        "n_eff": 2.5,
                    },
                },
                "mcmc": {"device": "cpu"},
                "output": {"directory": "./learn-output"},
            }
        )
    )

    config = LearnConfig.load(fn_config)

    assert config.specs == specs.resolve()
    assert config.models["rdf"].model_path == model.resolve()
    assert config.models["rdf"].independent_observations is False
    assert config.models["rdf"].n_eff is None
    assert config.models["rdf"].tolerance == 0.1
    assert config.models["density"].independent_observations is True
    assert config.models["density"].n_eff is None
    assert config.models["density"].tolerance is None
    assert config.models["pmf"].n_eff == 2.5
    assert config.output.posterior == (
        tmp_path / "learn-output" / "output" / "posterior.pt"
    ).resolve()


def test_learn_config_rejects_obsolete_effective_observation_keys(
    tmp_path: Path,
) -> None:
    specs = _write(tmp_path / "specs.yaml")
    model = _write(tmp_path / "model.lgp")
    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {
                    "rdf": {
                        "model_path": str(model),
                        "infer_effective_observations": True,
                        "tolerance": 0.1,
                    }
                },
                "mcmc": {},
            }
        )
    )

    with pytest.raises(ValueError, match="unsupported key"):
        LearnConfig.load(fn_config)


def test_learn_config_rejects_ambiguous_effective_observations(
    tmp_path: Path,
) -> None:
    specs = _write(tmp_path / "specs.yaml")
    model = _write(tmp_path / "model.lgp")
    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {
                    "density": {
                        "model_path": str(model),
                        "independent_observations": True,
                        "n_eff": 4,
                    }
                },
                "mcmc": {},
            }
        )
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        LearnConfig.load(fn_config)


def test_learn_config_requires_tolerance_for_curve_model(
    tmp_path: Path,
) -> None:
    specs = _write(tmp_path / "specs.yaml")
    model = _write(tmp_path / "model.lgp")
    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {
                    "rdf": {
                        "model_path": str(model),
                    }
                },
                "mcmc": {},
            }
        )
    )

    with pytest.raises(ValueError, match="requires a positive finite tolerance"):
        LearnConfig.load(fn_config)


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (
            {
                1: {
                    "model_path": "model.lgp",
                    "independent_observations": True,
                }
            },
            "non-empty strings",
        ),
        (
            {
                "density": {
                    "model_path": "model.lgp",
                    "independent_observations": "false",
                }
            },
            "must be true or false",
        ),
    ],
)
def test_learn_config_validates_model_names_and_booleans(
    tmp_path: Path,
    models: dict,
    message: str,
) -> None:
    _write(tmp_path / "specs.yaml")
    _write(tmp_path / "model.lgp")
    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": "./specs.yaml",
                "models": models,
                "mcmc": {},
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        LearnConfig.load(fn_config)


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
                        "system_id": "acetate",
                        "inputs": {
                            "topology": str(topol),
                            "coordinates": str(coord),
                            "mdp_production": str(mdp_prod),
                            "index": str(ndx),
                        },
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
                "campaign_dir": "./campaign",
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
                "systems": [
                    {
                        "system_id": "acetate",
                        "inputs": {
                            "topology": str(assets / "system-000.top"),
                            "coordinates": str(assets / "system-000.gro"),
                            "mdp_em": str(assets / "system-000.em.mdp"),
                            "mdp_production": str(assets / "system-000.mdp"),
                            "index": str(assets / "system-000.ndx"),
                        },
                        "n_steps": 1000,
                    }
                ],
                "bounds": {"charge C1": [-1.0, 1.0]},
                "charge_constraints": [
                    {
                        "selection": "resname ACE",
                        "target": 0.0,
                        "scope": "residue",
                        "implicit": "charge C1",
                    }
                ],
                "n_samples": 4,
            }
        )
    )

    config = SampleConfig.load(fn_config)

    assert config.charge_constraints[0].implicit == "charge C1"
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
                "systems": [
                    {
                        "system_id": "acetate",
                        "inputs": {
                            "topology": str(assets / "system-000.top"),
                            "coordinates": str(assets / "system-000.gro"),
                            "mdp_em": str(assets / "system-000.em.mdp"),
                            "mdp_production": str(assets / "system-000.mdp"),
                            "index": str(assets / "system-000.ndx"),
                        },
                        "n_steps": 1000,
                    }
                ],
                "specs": str(specs),
                "parameters": str(params),
            }
        )
    )

    config = ValidateConfig.load(fn_config)

    assert config.specs == specs.resolve()
    assert config.parameters == params.resolve()
    assert len(config.systems) == 1
