from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bff.workflows.build.config import BuildConfig
from bff.workflows.build_qoi_datasets.config import BuildQoIDatasetsConfig
from bff.workflows.fit_lgp.config import FitLGPConfig
from bff.workflows.label_snapshots.config import LabelSnapshotsConfig
from bff.workflows.learn.config import LearnConfig
from bff.workflows.md.config import MDJobConfig
from bff.workflows.sample_parameters.config import SampleParametersConfig
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
                        "box": [10, 10, 10],
                        "nsteps": {"npt": 250, "prod": 5000},
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
    assert config.systems[0].nsteps_npt == 250
    assert config.systems[0].nsteps_prod == 5000


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
                        "nsteps": {"npt": 0, "prod": 1000},
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


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ({"defaults": {"nsteps": {"npt": 0, "prod": 1000}}}, "unsupported key"),
        ({}, "missing required key 'nsteps'"),
    ],
)
def test_build_config_requires_steps_per_system(
    tmp_path: Path,
    invalid: dict,
    message: str,
) -> None:
    topology = _write(tmp_path / "system.top")
    mdp_em = _write(tmp_path / "em.mdp")
    mdp_npt = _write(tmp_path / "npt.mdp")
    mdp_prod = _write(tmp_path / "prod.mdp")
    config = {
        "project": "./project",
        "gromacs": {"command": "gmx"},
        "systems": [
            {
                "system_id": "acetate",
                "topology": str(topology),
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
    config.update(invalid)
    fn_config = tmp_path / "build.yaml"
    fn_config.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match=message):
        BuildConfig.load(fn_config)


def test_label_snapshots_config_loads_explicit_system(tmp_path: Path) -> None:
    topology = _write(tmp_path / "system.gro")
    trajectory = _write(tmp_path / "trajectory.xtc")
    md_input = _write(tmp_path / "md.inp")
    sp_input = _write(tmp_path / "sp.inp")
    hydrogen_input = _write(tmp_path / "h.inp")
    fn_config = tmp_path / "label-snapshots.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "output_dir": "./labels",
                "cp2k_cmd": "cp2k.psmp",
                "job_scheduler": "local",
                "systems": [
                    {
                        "system_id": "acetate",
                        "topology": str(topology),
                        "trajectory": str(trajectory),
                        "md_input": str(md_input),
                        "sp_input": str(sp_input),
                        "single_atom_inputs": {"h": str(hydrogen_input)},
                        "n_snapshots": 25,
                    }
                ],
            }
        )
    )

    config = LabelSnapshotsConfig.load(fn_config)

    assert config.output_dir == (tmp_path / "labels").resolve()
    assert config.systems[0].system_id == "acetate"
    assert config.systems[0].n_snapshots == 25
    assert config.systems[0].trajectory_path == trajectory.resolve()
    assert config.systems[0].single_atom_input_paths == {
        "H": hydrogen_input.resolve()
    }


def test_label_snapshots_config_rejects_incomplete_system(tmp_path: Path) -> None:
    fn_config = tmp_path / "label-snapshots.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "cp2k_cmd": "cp2k.psmp",
                "job_scheduler": "local",
                "systems": [{"system_id": "acetate"}],
            }
        )
    )

    with pytest.raises(ValueError, match="missing"):
        LabelSnapshotsConfig.load(fn_config)


def test_label_snapshots_requires_user_single_atom_inputs(tmp_path: Path) -> None:
    fn_config = tmp_path / "label-snapshots.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "cp2k_cmd": "cp2k.psmp",
                "job_scheduler": "local",
                "single_atoms": True,
                "systems": [
                    {
                        "system_id": "acetate",
                        "topology": str(_write(tmp_path / "system.gro")),
                        "trajectory": str(_write(tmp_path / "trajectory.xtc")),
                        "md_input": str(_write(tmp_path / "md.inp")),
                        "sp_input": str(_write(tmp_path / "sp.inp")),
                        "n_snapshots": 2,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="single_atom_inputs"):
        LabelSnapshotsConfig.load(fn_config)


def test_label_snapshots_config_rejects_duplicate_ids_and_invalid_split(
    tmp_path: Path,
) -> None:
    system = {
        "system_id": "acetate",
        "topology": str(_write(tmp_path / "system.gro")),
        "trajectory": str(_write(tmp_path / "trajectory.xtc")),
        "md_input": str(_write(tmp_path / "md.inp")),
        "sp_input": str(_write(tmp_path / "sp.inp")),
        "n_snapshots": 2,
    }
    fn_config = tmp_path / "label-snapshots.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "cp2k_cmd": "cp2k.psmp",
                "job_scheduler": "local",
                "single_atoms": False,
                "systems": [system, system],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        LabelSnapshotsConfig.load(fn_config)

    fn_config.write_text(
        yaml.safe_dump(
            {
                "cp2k_cmd": "cp2k.psmp",
                "job_scheduler": "local",
                "single_atoms": False,
                "train_fraction": 1.0,
                "systems": [system],
            }
        )
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        LabelSnapshotsConfig.load(fn_config)


def test_build_qoi_datasets_config_loads_minimal_config(
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

    fn_config = tmp_path / "build-qoi-datasets.yaml"
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

    config = BuildQoIDatasetsConfig.load(fn_config)

    assert config.training_samples.manifest == sample_manifest.resolve()
    assert len(config.reference.systems) == 1
    assert config.routines[0].name == "rdf"

    raw = yaml.safe_load(fn_config.read_text())
    for removed_option in ("gc_collect", "maxtasksperchild"):
        raw["run"] = {removed_option: False}
        fn_config.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="run contains unsupported"):
            BuildQoIDatasetsConfig.load(fn_config)


def test_fit_lgp_config_loads_minimal_config(tmp_path: Path) -> None:
    data = _write(tmp_path / "dataset.pt")

    fn_config = tmp_path / "fit-lgp.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {"rdf": {"data": str(data)}},
                "fit": {"model_dir": "./models", "device": "cpu"},
            }
        )
    )

    config = FitLGPConfig.load(fn_config)

    assert config.fit.model_dir == (tmp_path / "models").resolve()
    assert config.datasets[0].name == "rdf"
    assert config.datasets[0].fn_model == (tmp_path / "models" / "rdf.lgp").resolve()


def test_fit_lgp_config_rejects_observation_scale(tmp_path: Path) -> None:
    data = _write(tmp_path / "dataset.pt")

    fn_config = tmp_path / "fit-lgp.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "rdf": {
                        "data": str(data),
                        "observation_scale": 2.0,
                    }
                },
                "fit": {"model_dir": "./models", "device": "cpu"},
            }
        )
    )

    with pytest.raises(ValueError, match="observation_scale"):
        FitLGPConfig.load(fn_config)


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
        tmp_path / "learn-output" / "outputs" / "posterior.pt"
    ).resolve()
    assert config.output.specs == (
        tmp_path / "learn-output" / "outputs" / "specs.yaml"
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


def test_sample_parameters_config_loads_prepared_assets(tmp_path: Path) -> None:
    assets = _make_prepared_system(tmp_path / "assets")

    fn_config = tmp_path / "sample-parameters.yaml"
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

    config = SampleParametersConfig.load(fn_config)

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


def test_validate_config_loads_posterior_source(tmp_path: Path) -> None:
    assets = _make_prepared_system(tmp_path / "assets")
    posterior = _write(tmp_path / "posterior.pt")
    fn_config = tmp_path / "validate-posterior.yaml"
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
                "posterior": {
                    "file": str(posterior),
                    "n_samples": 0,
                    "include_mean": True,
                    "distribution": "empirical",
                    "confidence": 0.8,
                    "seed": 17,
                },
            }
        )
    )

    config = ValidateConfig.load(fn_config)

    assert config.parameters is None
    assert config.specs is None
    assert config.posterior is not None
    assert config.posterior.file == posterior.resolve()
    assert config.posterior.n_samples == 0
    assert config.posterior.include_mean is True
    assert config.posterior.distribution == "empirical"
    assert config.posterior.confidence == 0.8
    assert config.posterior.seed == 17


@pytest.mark.parametrize(
    ("posterior_update", "message"),
    [
        ({"n_samples": -1}, "n_samples"),
        ({"n_samples": 0, "include_mean": False}, "at least one sample"),
        ({"distribution": "gamma"}, "distribution"),
        ({"confidence": 1.0}, "confidence"),
        ({"include_mean": 1}, "include_mean"),
        ({"seed": True}, "seed"),
        ({"unexpected": 1}, "unsupported key"),
    ],
)
def test_validate_config_rejects_invalid_posterior_options(
    tmp_path: Path,
    posterior_update: dict,
    message: str,
) -> None:
    assets = _make_prepared_system(tmp_path / "assets")
    posterior_file = _write(tmp_path / "posterior.pt")
    posterior = {"file": str(posterior_file), "n_samples": 1}
    posterior.update(posterior_update)
    fn_config = tmp_path / "validate-posterior.yaml"
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
                            "mdp_production": str(assets / "system-000.mdp"),
                            "index": str(assets / "system-000.ndx"),
                        },
                        "n_steps": 1000,
                    }
                ],
                "posterior": posterior,
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        ValidateConfig.load(fn_config)


@pytest.mark.parametrize(
    "extra",
    [
        {"parameters": "parameters.yaml"},
        {"specs": "specs.yaml"},
    ],
)
def test_validate_config_rejects_conflicting_posterior_sources(
    tmp_path: Path,
    extra: dict,
) -> None:
    assets = _make_prepared_system(tmp_path / "assets")
    _write(tmp_path / "posterior.pt")
    for path in extra.values():
        _write(tmp_path / path)
    fn_config = tmp_path / "validate-posterior.yaml"
    config = {
        "campaign_dir": "./campaign",
        "gmx_cmd": "gmx",
        "job_scheduler": "local",
        "systems": [
            {
                "system_id": "acetate",
                "inputs": {
                    "topology": str(assets / "system-000.top"),
                    "coordinates": str(assets / "system-000.gro"),
                    "mdp_production": str(assets / "system-000.mdp"),
                    "index": str(assets / "system-000.ndx"),
                },
                "n_steps": 1000,
            }
        ],
        "posterior": {"file": "posterior.pt"},
        **extra,
    }
    fn_config.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError):
        ValidateConfig.load(fn_config)
