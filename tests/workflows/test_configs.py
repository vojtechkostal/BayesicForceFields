from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bff.workflows.build.config import BuildConfig
from bff.workflows.build_qoi_datasets.config import BuildQoIDatasetsConfig
from bff.workflows.campaign.job import MDJobConfig
from bff.workflows.fit_lgp.config import FitLGPConfig
from bff.workflows.learn.config import LearnConfig
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
    """A build directory with only the files a campaign uses: no trajectory,
    NpT input, or system.yaml."""
    for system_id in ("000", "001"):
        system_dir = project_dir / "systems" / system_id
        for name in (
            "topology.top",
            "production.gro",
            "index.ndx",
            "em.mdp",
            "production.mdp",
        ):
            _write(system_dir / name, f"{name}\n")
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
                "project": {"directory": "./project"},
                "gromacs": {"command": "gmx"},
                "systems": [
                    {
                        "system_id": "acetate",
                        "topology": str(topology),
                        "templates": {"ACE": str(template)},
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
                "project": {"directory": "./project"},
                "gromacs": {"command": "gmx"},
                "systems": [
                    {
                        "system_id": "acetate",
                        "topology": str(topology),
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
        ({}, "missing required key.*'nsteps'"),
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
        "project": {"directory": "./project"},
        "gromacs": {"command": "gmx"},
        "systems": [
            {
                "system_id": "acetate",
                "topology": str(topology),
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
                "fit": {"model_dir": "./models"},
            }
        )
    )

    config = FitLGPConfig.load(fn_config)

    assert config.fit.model_dir == (tmp_path / "models").resolve()
    assert config.datasets[0].name == "rdf"
    assert config.datasets[0].fn_model == (tmp_path / "models" / "rdf.lgp").resolve()


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
                    "density": {"model_path": str(model)},
                },
                "mcmc": {"device": "cpu"},
                "plots": {
                    "max_marginal_samples": -1,
                    "plot_metadata": {
                        "define VSA": {
                            "xlabel": "O-VS",
                            "ylabel": "angle [degree]",
                        }
                    }
                },
                "output": {"directory": "./learn-output"},
            }
        )
    )

    config = LearnConfig.load(fn_config)

    assert config.specs == specs.resolve()
    assert config.models["rdf"].model_path == model.resolve()
    assert config.models["rdf"].tolerance == 0.1
    assert config.models["density"].tolerance == 0.0
    assert config.plots.max_marginal_samples is None
    assert config.plots.plot_metadata == {
        "define VSA": {"xlabel": "O-VS", "ylabel": "angle [degree]"}
    }
    assert config.output.results == (
        tmp_path / "learn-output" / "outputs" / "results.pt"
    ).resolve()
    assert config.output.specs == (
        tmp_path / "learn-output" / "outputs" / "specs.yaml"
    ).resolve()


def test_learn_config_accepts_unlimited_marginal_samples(tmp_path: Path) -> None:
    unlimited = -1
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
                "plots": {"max_marginal_samples": unlimited},
            }
        )
    )

    assert LearnConfig.load(fn_config).plots.max_marginal_samples is None


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ({1: {"model_path": "model.lgp"}}, "non-empty strings"),
        (
            {"density": {"model_path": "model.lgp", "tolerance": -0.1}},
            "tolerance must be a finite number >= 0",
        ),
        ({"density": {"model_path": "model.lgp", "n_eff": 4}}, "unsupported key"),
    ],
)
def test_learn_config_validates_model_names_and_tolerances(
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


def test_md_job_reads_campaign_and_its_sample(tmp_path: Path) -> None:
    from bff.domain.samples import write_sample_manifest

    campaign_dir = tmp_path / "campaign"
    for name in ("topology.top", "coordinates.gro", "production.mdp", "index.ndx"):
        _write(campaign_dir / "systems" / "acetate" / name)
    write_sample_manifest(
        campaign_dir / "samples.yaml",
        parameter_names=["charge A", "charge B"],
        systems={"acetate": 1000},
        samples={"00": {"params": [0.1, 0.2], "status": "staged"}},
    )
    fn_campaign = campaign_dir / "campaign.yaml"
    staged = "systems/acetate"
    fn_campaign.write_text(
        yaml.safe_dump(
            {
                "gmx_cmd": "gmx",
                "systems": [
                    {
                        "system_id": "acetate",
                        "inputs": {
                            "topology": f"{staged}/topology.top",
                            "coordinates": f"{staged}/coordinates.gro",
                            "mdp_production": f"{staged}/production.mdp",
                            "index": f"{staged}/index.ndx",
                        },
                        "n_steps": 1000,
                    }
                ],
            }
        )
    )

    config = MDJobConfig.load(fn_campaign, "00")

    assert config.params == [0.1, 0.2]
    assert config.fn_specs == campaign_dir / "specs.yaml"
    assert config.store == ("xtc",)
    assert config.systems[0].topology_path == campaign_dir / staged / "topology.top"
    with pytest.raises(ValueError, match="'01' is not in"):
        MDJobConfig.load(fn_campaign, "01")

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
                        "implicit": "C1",
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
    params = _write(tmp_path / "results.pt")

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
    posterior = _write(tmp_path / "results.pt")
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


def test_config_section_reports_keys_types_and_ranges(tmp_path: Path) -> None:
    from bff.workflows.config import ConfigSection

    def section(raw):
        return ConfigSection(
            raw, "stage", base_dir=tmp_path, allowed=("a", "b"), required=("b",)
        )

    with pytest.raises(ValueError, match="stage contains unsupported key.*legacy"):
        section({"b": 1, "legacy": 2})
    with pytest.raises(ValueError, match="stage is missing required key.*'b'"):
        section({"a": 1})
    with pytest.raises(ValueError, match="stage is missing required key.*'b'"):
        section({"b": None})
    with pytest.raises(ValueError, match="stage must be a mapping"):
        section([])

    config = section({"a": "1e-3", "b": 2.0})
    assert config.number("a", minimum=0, exclusive=True) == pytest.approx(1e-3)
    assert config.integer("b", minimum=1) == 2
    with pytest.raises(ValueError, match=r"stage.b must be an integer >= 3, got 2"):
        config.integer("b", minimum=3)
    with pytest.raises(ValueError, match="stage.a must be an integer"):
        config.integer("a")
    with pytest.raises(ValueError, match="stage.b must be true or false"):
        config.boolean("b")
    with pytest.raises(ValueError, match="stage.b must be a non-empty string"):
        config.string("b")
    with pytest.raises(FileNotFoundError, match="stage.a: .* does not exist"):
        config.path("a")
    assert section({"b": True}).integer("a", 7) == 7
    with pytest.raises(ValueError, match="must be an integer"):
        section({"b": True}).integer("b")


def _constraint(implicit: str = "C1") -> dict:
    return {
        "selection": "resname ACE",
        "target": -1.0,
        "scope": "residue",
        "implicit": implicit,
    }


def _build_config(tmp_path: Path, **system_overrides) -> Path:
    system = {
        "system_id": "acetate",
        "topology": str(_write(tmp_path / "system.top")),
        "nsteps": {"npt": 0, "prod": 1000},
        "mdp": {
            "em": str(_write(tmp_path / "em.mdp")),
            "npt": str(_write(tmp_path / "npt.mdp")),
            "prod": str(_write(tmp_path / "prod.mdp")),
        },
        **system_overrides,
    }
    fn_config = tmp_path / "build.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "project": {"directory": "./project"},
                "gromacs": {"command": "gmx"},
                "systems": [system],
            }
        )
    )
    return fn_config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"nsteps": {"npt": 0, "prod": 0}}, r"systems\[0\].nsteps.prod must be"),
        ({"nsteps": {"npt": 1.5, "prod": 10}}, "nsteps.npt must be an integer"),
        ({"box": [10, -1, 10]}, "box must be 3 or 6 positive numbers"),
        (
            {"bias": {"colvars_file": "a.colvars.dat", "plumed_file": "b.dat"}},
            "colvars_file or plumed_file, not both",
        ),
        ({"bias": {"kind": "colvars"}}, "unsupported key"),
        ({"system_name": 3}, "system_name must be a non-empty string"),
    ],
)
def test_build_config_rejects_implausible_systems(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BuildConfig.load(_build_config(tmp_path, **overrides))


def test_build_config_rejects_project_shorthand(tmp_path: Path) -> None:
    fn_config = _build_config(tmp_path)
    config = yaml.safe_load(fn_config.read_text())
    config["project"] = "./project"
    fn_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="project must be a mapping"):
        BuildConfig.load(fn_config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bounds": {"charge C1": [1.0, -1.0]}}, "finite lower < upper"),
        ({"store": "xtc"}, "store must be a list of strings"),
        ({"job_scheduler": "pbs"}, "job_scheduler must be one of"),
        ({"n_samples": 0}, "n_samples must be an integer >= 1"),
        ({"max_restarts": -1}, "max_restarts must be an integer >= 0"),
        ({"max_restarts": 2}, "max_restarts applies only to job_scheduler: slurm"),
        ({"overwrite": True, "resume": True}, "cannot both be true"),
        ({"seed": -1}, "seed must be an integer >= 0"),
        ({"local": {"max_parallel_jobs": 0}}, "local.max_parallel_jobs must be"),
        (
            {"charge_constraints": [_constraint(implicit="C9")]},
            "implicit must be an atom name or type of exactly one",
        ),
        (
            {
                "bounds": {"charge C1": [-1.0, 1.0], "charge C1 C2": [-1.0, 1.0]},
                "charge_constraints": [_constraint(implicit="C1")],
            },
            "exactly one 'charge ...' parameter",
        ),
        (
            {"charge_constraints": [_constraint(), _constraint()]},
            "must each solve a different charge parameter",
        ),
    ],
)
def test_sample_parameters_config_rejects_implausible_values(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    assets = _make_prepared_system(tmp_path / "assets")
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
        "bounds": {"charge C1": [-1.0, 1.0]},
        "n_samples": 4,
        **overrides,
    }
    fn_config = tmp_path / "sample-parameters.yaml"
    fn_config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match=message):
        SampleParametersConfig.load(fn_config)


@pytest.mark.parametrize(
    ("fit", "dataset", "message"),
    [
        ({"max_iter": 0}, {}, "max_iter must be an integer >= 1"),
        ({"test_fraction": 1.0}, {}, "test_fraction must be a finite number > 0"),
        ({"tol_grad": "tight"}, {}, "fit.tol_grad must be a finite number"),
        ({}, {"mean": "linear"}, "mean must be 'data', 'sigmoid', a number"),
        ({}, {"nuisance": 0}, "nuisance must be a finite number > 0"),
    ],
)
def test_fit_lgp_config_rejects_implausible_values(
    tmp_path: Path, fit: dict, dataset: dict, message: str
) -> None:
    data = _write(tmp_path / "dataset.pt")
    fn_config = tmp_path / "fit-lgp.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {"datasets": {"rdf": {"data": str(data), **dataset}}, "fit": fit}
        )
    )
    with pytest.raises(ValueError, match=message):
        FitLGPConfig.load(fn_config)


def test_fit_lgp_config_reads_exponent_strings(tmp_path: Path) -> None:
    data = _write(tmp_path / "dataset.pt")
    fn_config = tmp_path / "fit-lgp.yaml"
    fn_config.write_text(
        f"datasets: {{rdf: {{data: {data}}}}}\nfit: {{tol_grad: 1e-3}}\n"
    )
    assert FitLGPConfig.load(fn_config).fit.opt_kwargs == {"tol_grad": 1e-3}


@pytest.mark.parametrize(
    ("mcmc", "message"),
    [
        ({"warmup": 1500}, "smaller than mcmc.total_steps"),
        ({"priors_disttype": "cauchy"}, "priors_disttype must be one of"),
        ({"rhat_tol": 1.0}, "rhat_tol must be a finite number > 1"),
        ({"n_walkers": 1}, "n_walkers must be an integer >= 2"),
    ],
)
def test_learn_config_rejects_implausible_mcmc(
    tmp_path: Path, mcmc: dict, message: str
) -> None:
    specs = _write(tmp_path / "specs.yaml")
    model = _write(tmp_path / "model.lgp")
    fn_config = tmp_path / "learn.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {"rdf": {"model_path": str(model), "tolerance": 0.1}},
                "mcmc": mcmc,
            }
        )
    )
    with pytest.raises(ValueError, match=message):
        LearnConfig.load(fn_config)


def test_campaign_from_build_needs_only_the_files_it_uses(tmp_path: Path) -> None:
    source = _make_build_stage(tmp_path / "01-build")
    fn_config = tmp_path / "sample-parameters.yaml"
    fn_config.write_text(
        yaml.safe_dump(
            {
                "campaign_dir": "./campaign",
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
                "source": str(source),
                "systems": [
                    {"system_id": "000", "n_steps": 100},
                    {"system_id": "001", "n_steps": 100},
                ],
                "bounds": {"charge C1": [-1.0, 1.0]},
                "n_samples": 2,
            }
        )
    )

    config = SampleParametersConfig.load(fn_config)

    system = config.systems[0]
    assert system.coordinates_path == source / "systems/000/production.gro"
    assert system.mdp_production_path == source / "systems/000/production.mdp"

    (source / "systems/001/index.ndx").unlink()
    with pytest.raises(FileNotFoundError, match="'001'.*index.ndx"):
        SampleParametersConfig.load(fn_config)


def test_fit_lgp_custom_mean_resolves_relative_to_the_config(tmp_path: Path) -> None:
    data = str(_write(tmp_path / "dataset.pt"))
    _write(tmp_path / "means.py", "def flat(X):\n    return X[:, :1] * 0\n")

    def load(**dataset) -> FitLGPConfig:
        fn_config = tmp_path / "fit-lgp.yaml"
        config = {"datasets": {"rdf": {"data": data, **dataset}}}
        fn_config.write_text(yaml.safe_dump(config))
        return FitLGPConfig.load(fn_config)

    custom = load(mean="means.py:flat").datasets[0].mean
    assert custom == f"{(tmp_path / 'means.py').resolve()}:flat"
    assert load().datasets[0].mean == "data"
