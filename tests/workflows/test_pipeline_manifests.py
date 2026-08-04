from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import torch
import yaml
from typer.testing import CliRunner

from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.learning import fit_surrogates
from bff.cli import app
from bff.domain.sample import SampleSet
from bff.domain.systems import (
    BuildSystemMetadata,
    ReferenceSystemMetadata,
    load_build_system_metadata,
    load_reference_system_metadata,
    write_build_system_metadata,
    write_reference_system_metadata,
)
from bff.io.logs import Logger
from bff.io.utils import save_json
from bff.qoi.data import QoIDataset
from bff.qoi.hbonds import compute_hydrogen_bond_qoi
from bff.qoi.rdf import compute_rdf_qoi
from bff.qoi.routines import normalize_routine_list, run_analysis_routine
from bff.workflows.analyze.main import main as analyze_main
from bff.workflows.evaluate_snapshots.config import EvaluateSnapshotsConfig
from bff.workflows.evaluate_snapshots.main import _run_cp2k
from bff.workflows.learn.config import LearnConfig
from bff.workflows.learn.main import _prepare_output
from bff.workflows.learn.main import main as learn_main
from bff.workflows.lgpfit.main import main as lgpfit_main
from bff.workflows.prepare_reference.main import main as prepare_reference_main
from bff.workflows.sample.config import SampleConfig
from bff.workflows.validate.config import ValidateConfig


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_system_metadata_round_trips_without_paths_or_versions(tmp_path: Path) -> None:
    build_path = write_build_system_metadata(
        tmp_path,
        "acetate",
        BuildSystemMetadata(
            system_name="Aqueous acetate",
            charge=-1,
            multiplicity=1,
            box=(10.0, 11.0, 12.0, 90.0, 90.0, 90.0),
            maxwarn=0,
            production_steps=1000,
        ),
    )
    raw = yaml.safe_load(build_path.read_text())
    assert not ({"schema_version", "version", "paths", "inputs"} & set(raw))
    loaded = load_build_system_metadata(tmp_path, "acetate")
    assert loaded.system_name == "Aqueous acetate"
    assert loaded.box[:3] == (10.0, 11.0, 12.0)

    reference_path = write_reference_system_metadata(
        tmp_path / "reference",
        "acetate",
        ReferenceSystemMetadata(
            system_name="Aqueous acetate",
            charge=-1,
            multiplicity=1,
            box=loaded.box,
            snapshot_count=3,
            elements=("C", "H", "O"),
        ),
    )
    reference_raw = yaml.safe_load(reference_path.read_text())
    assert not ({"schema_version", "version", "paths", "inputs"} & set(reference_raw))
    reference = load_reference_system_metadata(tmp_path / "reference", "acetate")
    assert reference.snapshot_count == 3
    assert reference.elements == ("C", "H", "O")


def _build_stage(root: Path, *, both_biases: bool = False) -> Path:
    system_dir = root / "systems" / "acetate"
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
        _write(system_dir / name)
    write_build_system_metadata(
        root,
        "acetate",
        BuildSystemMetadata(
            system_name=None,
            charge=-1,
            multiplicity=1,
            box=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
            maxwarn=0,
            production_steps=100,
        ),
    )
    if both_biases:
        _write(system_dir / "bias.colvars.dat")
        _write(system_dir / "bias.plumed.dat")
    return root


def test_sample_uses_build_directory_contract_and_rejects_ambiguous_bias(
    tmp_path: Path,
) -> None:
    source = _build_stage(tmp_path / "build")
    config_path = tmp_path / "sample.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "campaign_dir": "campaign",
                "source": str(source),
                "systems": [{"system_id": "acetate", "n_steps": 10}],
                "bounds": {"sigma A": [0.1, 1.0]},
                "charge_constraints": [],
                "n_samples": 2,
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
            }
        )
    )
    config = SampleConfig.load(config_path)
    assert config.systems[0].coordinates_path == (
        source / "systems" / "acetate" / "production.gro"
    ).resolve()

    _write(source / "systems" / "acetate" / "bias.colvars.dat")
    _write(source / "systems" / "acetate" / "bias.plumed.dat")
    with pytest.raises(ValueError, match="both reserved bias files"):
        SampleConfig.load(config_path)


def test_validate_uses_build_directory_contract(tmp_path: Path) -> None:
    source = _build_stage(tmp_path / "build")
    _write(tmp_path / "specs.yaml")
    _write(tmp_path / "parameters.yaml")
    path = tmp_path / "validate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "campaign_dir": "validation",
                "source": str(source),
                "systems": [{"system_id": "acetate", "n_steps": 10}],
                "specs": "specs.yaml",
                "parameters": "parameters.yaml",
                "gmx_cmd": "gmx",
                "job_scheduler": "local",
            }
        )
    )
    config = ValidateConfig.load(path)
    assert config.systems[0].topology_path == (
        source / "systems/acetate/topology.top"
    ).resolve()


def test_prepare_reference_writes_local_metadata_without_path_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    source = _build_stage(tmp_path / "build")
    config = tmp_path / "prepare-reference.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "source": str(source),
                "output": "reference",
                "systems": ["acetate"],
                "n_single_point_snapshots": 2,
            }
        )
    )
    monkeypatch.setattr(
        "bff.workflows.prepare_reference.main.write_reference_inputs",
        lambda *args, **kwargs: (2, ("C", "H", "O")),
    )
    prepare_reference_main(config)
    metadata_path = tmp_path / "reference/systems/acetate/system.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    assert metadata["snapshot_count"] == 2
    assert metadata["elements"] == ["C", "H", "O"]
    assert not ({"paths", "inputs", "version", "schema_version"} & set(metadata))


def _reference_stage(root: Path, *, snapshot_count: int = 2) -> Path:
    system_dir = root / "systems" / "acetate"
    for name in (
        "system.top",
        "system.gro",
        "system.xyz",
        "snapshots/md.inp",
        "snapshots/sp.inp",
        "single-atoms/carbon/input.inp",
        "single-atoms/carbon/pos.xyz",
    ):
        _write(system_dir / name)
    for index in range(snapshot_count):
        _write(system_dir / "snapshots" / "xyz" / f"snapshot-{index:04d}.xyz")
    write_reference_system_metadata(
        root,
        "acetate",
        ReferenceSystemMetadata(
            system_name=None,
            charge=-1,
            multiplicity=1,
            box=(10.0, 10.0, 10.0, 90.0, 90.0, 90.0),
            snapshot_count=snapshot_count,
            elements=("C",),
        ),
    )
    return root


def _evaluate_config(tmp_path: Path, source: Path, systems: list[object]) -> Path:
    path = tmp_path / "evaluate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": str(source),
                "systems": systems,
                "output_dir": "evaluated",
                "job_scheduler": "local",
                "cp2k_cmd": "cp2k",
            }
        )
    )
    return path


def test_evaluate_resolves_reference_contract_and_rejects_stale_snapshots(
    tmp_path: Path,
) -> None:
    source = _reference_stage(tmp_path / "reference")
    path = _evaluate_config(tmp_path, source, ["acetate"])
    config = EvaluateSnapshotsConfig.load(path)
    assert [item.name for item in config.systems[0].snapshot_files] == [
        "snapshot-0000.xyz",
        "snapshot-0001.xyz",
    ]

    _write(source / "systems/acetate/snapshots/xyz/snapshot-0002.xyz")
    with pytest.raises(ValueError, match="beyond snapshot_count"):
        EvaluateSnapshotsConfig.load(path)


def test_reference_contract_rejects_missing_snapshot_unknown_element_and_mixing(
    tmp_path: Path,
) -> None:
    source = _reference_stage(tmp_path / "reference")
    path = _evaluate_config(tmp_path, source, ["acetate"])
    (source / "systems/acetate/snapshots/xyz/snapshot-0001.xyz").unlink()
    with pytest.raises(FileNotFoundError, match="snapshot-0001.xyz"):
        EvaluateSnapshotsConfig.load(path)

    metadata_path = source / "systems/acetate/system.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    metadata["elements"] = ["Xx"]
    metadata_path.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="unknown element"):
        EvaluateSnapshotsConfig.load(path)

    mixed = _evaluate_config(
        tmp_path,
        source,
        [{"system_id": "acetate", "inputs": {"topology": "other.top"}}],
    )
    with pytest.raises(ValueError, match="mixes source"):
        EvaluateSnapshotsConfig.load(mixed)


def _sample_campaign(tmp_path: Path, outputs: list[dict]) -> Path:
    for system_id in ("a", "b"):
        _write(tmp_path / "systems" / system_id / "topology.top")
        _write(tmp_path / "systems" / system_id / "coordinates.gro")
    for output in outputs:
        _write(tmp_path / output["trajectory"])
    (tmp_path / "specs.yaml").write_text(
        yaml.safe_dump(
            {"bounds": {"sigma A": [0.1, 1.0]}, "charge_constraints": []}
        )
    )
    (tmp_path / "samples.yaml").write_text(
        yaml.safe_dump(
            {
                "systems": [
                    {
                        "system_id": system_id,
                        "topology": f"systems/{system_id}/topology.top",
                        "coordinates": f"systems/{system_id}/coordinates.gro",
                    }
                    for system_id in ("a", "b")
                ],
                "samples": {
                    "sample-0": {
                        "params": [0.4],
                        "status": "completed",
                        "outputs": outputs,
                    }
                },
            }
        )
    )
    return tmp_path


def test_sample_outputs_are_paired_by_system_id_not_position(tmp_path: Path) -> None:
    campaign = _sample_campaign(
        tmp_path,
        [
            {"system_id": "b", "trajectory": "samples/sample-0/b/traj.xtc"},
            {"system_id": "a", "trajectory": "samples/sample-0/a/traj.xtc"},
        ],
    )
    samples = SampleSet.from_dir(campaign)
    sample = samples.samples[0]
    assert sample.system_ids == ("a", "b")
    assert [path.parts[-2] for path in sample.trajectory_paths] == ["a", "b"]


@pytest.mark.parametrize(
    "outputs, expected",
    [
        (
            [
                {"system_id": "a", "trajectory": "samples/s/a/one.xtc"},
                {"system_id": "a", "trajectory": "samples/s/a/two.xtc"},
            ],
            "duplicate a",
        ),
        (
            [{"system_id": "a", "trajectory": "samples/s/a/one.xtc"}],
            "missing b",
        ),
    ],
)
def test_sample_manifest_rejects_duplicate_or_missing_system_outputs(
    tmp_path: Path,
    outputs: list[dict],
    expected: str,
) -> None:
    campaign = _sample_campaign(tmp_path, outputs)
    with pytest.raises(ValueError, match=expected):
        SampleSet.from_dir(campaign)


def test_custom_file_routine_receives_roles_and_authoritative_name(
    tmp_path: Path,
) -> None:
    profile = _write(tmp_path / "profile.pmf", "0 1\n")
    module = _write(
        tmp_path / "routine.py",
        "from bff.qoi.data import QoI\n"
        "def load_profile(*, inputs, system_id, sample_id, options):\n"
        "    assert inputs['pmf'].name == 'profile.pmf'\n"
        "    assert system_id == 'contact' and sample_id == 'reference'\n"
        "    assert options == {'scale': 2}\n"
        "    return QoI('ignored', [1.0, 2.0])\n",
    )
    routine = normalize_routine_list(
        [
            {
                "name": "contact-pmf",
                "callable": f"{module}:load_profile",
                "loader": "files",
                "systems": ["contact"],
                "inputs": ["pmf"],
                "options": {"scale": 2},
            }
        ]
    )[0]
    result = run_analysis_routine(
        routine,
        universe=None,
        inputs={"pmf": profile},
        system_id="contact",
        sample_id="reference",
        start=0,
        stop=None,
        step=1,
    )
    assert result.name == "contact-pmf"


def _trajectory_universe() -> mda.Universe:
    universe = mda.Universe.empty(3, trajectory=True)
    universe.add_TopologyAttr("names", ["O1", "H1", "OW"])
    universe.add_TopologyAttr("types", ["O", "H", "O"])
    universe.add_TopologyAttr("resnames", ["ACE"])
    universe.add_TopologyAttr("resids", [1])
    universe.add_TopologyAttr("bonds", [(0, 1)])
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.7, 0.0, 0.0]],
        ]
    )
    dimensions = np.asarray(
        [
            [10.0, 11.0, 12.0, 80.0, 95.0, 105.0],
            [12.0, 13.0, 14.0, 75.0, 90.0, 110.0],
        ]
    )
    universe.load_new(coordinates, format="MEMORY", dimensions=dimensions)
    return universe


def test_rdf_handles_changing_triclinic_boxes_and_excludes_self_pairs() -> None:
    universe = _trajectory_universe()
    qoi = compute_rdf_qoi(
        universe,
        group_a="name O1 OW",
        group_b="name O1 OW",
        range=(0.0, 5.0),
        bins=20,
        pbc=True,
        smooth=False,
    )
    assert qoi.values.shape == (20,)
    assert np.all(np.isfinite(qoi.values))


def test_hydrogen_bonds_use_explicit_selections_and_bonds() -> None:
    universe = _trajectory_universe()
    qoi = compute_hydrogen_bond_qoi(
        universe,
        donors="name O1",
        hydrogens="name H1",
        acceptors="name OW",
        pbc=True,
    )
    assert qoi.labels == ("ACE(O) to ACE(O)",)
    assert qoi.values.tolist() == [1.0]


def test_lgp_cache_rejects_same_shape_different_data(tmp_path: Path) -> None:
    first = QoIDataset("pmf", [[0.0], [1.0]], [[1.0], [2.0]], [1.5])
    second = QoIDataset("pmf", [[0.0], [1.0]], [[1.0], [3.0]], [1.5])
    lgp = LocalGaussianProcess(
        torch.tensor(first.inputs, dtype=torch.float32),
        torch.tensor(first.outputs, dtype=torch.float32),
        0.0,
        torch.ones(1),
        1.0,
        0.1,
        "cpu",
    )
    model = LGPCommittee(
        [lgp],
        first.outputs_ref,
        n_curves=1,
        dataset_fingerprint=first.fingerprint(),
    )
    fn_model = tmp_path / "pmf.lgp"
    model.write(fn_model)
    with pytest.raises(ValueError, match="different QoI data"):
        fit_surrogates(
            [second],
            model_paths={"pmf": fn_model},
            reuse_models=True,
            device="cpu",
        )


def _learn_config(tmp_path: Path, *, resume: bool = False, overwrite: bool = False):
    _write(tmp_path / "specs.yaml")
    _write(tmp_path / "model.lgp")
    config_path = tmp_path / "learn.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "specs": "specs.yaml",
                "models": {
                    "pmf": {
                        "model_path": "model.lgp",
                        "independent_observations": True,
                    }
                },
                "mcmc": {"resume": resume},
                "output": {"directory": "run", "overwrite": overwrite},
            }
        )
    )
    return LearnConfig.load(config_path)


def test_learning_collision_policy_only_removes_owned_files(tmp_path: Path) -> None:
    config = _learn_config(tmp_path)
    _write(config.output.log)
    with pytest.raises(ValueError, match="already exist"):
        _prepare_output(config)

    config = _learn_config(tmp_path, overwrite=True)
    unknown = _write(config.output.directory / "keep-me.txt")
    _prepare_output(config)
    assert not config.output.log.exists()
    assert unknown.exists()


def test_cli_exposes_lgpfit_and_rejects_fit() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert "lgpfit" in help_result.stdout
    assert "prepare-reference" in help_result.stdout
    assert runner.invoke(app, ["fit", "missing.yaml"]).exit_code != 0
    assert runner.invoke(app, ["prepare-assets", "missing.yaml"]).exit_code != 0


def test_raw_json_and_non_tty_progress_are_lossless(
    tmp_path: Path, monkeypatch
) -> None:
    value = 0.1234567890123456
    output = tmp_path / "raw.json"
    save_json({"value": np.float64(value)}, output)
    assert json.loads(output.read_text())["value"] == value

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    logger = Logger("sample")
    logger.status("Work", "1/2", overwrite=True)
    logger.done("Work")
    assert "\r" not in stream.getvalue()


def test_local_cp2k_subprocess_output_does_not_interleave(
    tmp_path: Path, capsys
) -> None:
    executable = _write(
        tmp_path / "fake-cp2k",
        "#!/bin/sh\nprintf 'launcher noise\\n'\nexit 0\n",
    )
    executable.chmod(0o755)
    _write(tmp_path / "input.inp")

    _run_cp2k(
        cp2k_cmd=str(executable),
        fn_input="input.inp",
        fn_output="output.out",
        cwd=tmp_path,
    )

    assert "launcher noise" not in capsys.readouterr().out


def test_small_cpu_learning_run_writes_fixed_artifacts_and_resumes(
    tmp_path: Path,
) -> None:
    specs = tmp_path / "specs.yaml"
    specs.write_text(
        yaml.safe_dump(
            {"bounds": {"sigma A": [0.5, 1.5]}, "charge_constraints": []}
        )
    )
    x_train = torch.tensor([[0.5], [0.8], [1.2], [1.5]])
    y_train = torch.tensor([[0.6], [0.9], [1.1], [1.4]])
    lgp = LocalGaussianProcess(
        x_train,
        y_train,
        0.0,
        torch.ones(1),
        1.0,
        0.1,
        "cpu",
    )
    model = LGPCommittee(
        [lgp],
        np.asarray([1.0]),
        n_curves=1,
        nuisance=0.1,
        dataset_fingerprint="fixture",
    )
    model_path = tmp_path / "pmf.lgp"
    model.write(model_path)

    def write_config(*, steps: int, resume: bool) -> Path:
        path = tmp_path / "learn.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "specs": str(specs),
                    "models": {
                        "pmf": {"model_path": str(model_path), "n_eff": 1.0}
                    },
                    "mcmc": {
                        "total_steps": steps,
                        "warmup": 2,
                        "n_walkers": 4,
                        "progress_stride": 2,
                        "resume": resume,
                        "device": "cpu",
                    },
                    "output": {"directory": "run"},
                }
            )
        )
        return path

    learn_main(write_config(steps=8, resume=False))
    output = tmp_path / "run"
    mandatory = [
        output / "learn.log",
        output / "output" / "prior.pt",
        output / "output" / "posterior.pt",
        output / "output" / "mcmc.ckpt",
        output / "plots" / "marginals.pdf",
        output / "plots" / "qoi-marginals.pdf",
        output / "plots" / "corner.pdf",
    ]
    assert all(path.is_file() for path in mandatory)
    posterior = torch.load(output / "output" / "posterior.pt", weights_only=False)
    assert posterior["metadata"]["specifications_fingerprint"]
    assert posterior["metadata"]["model_fingerprints"]["pmf"]

    learn_main(write_config(steps=10, resume=True))
    assert "Posterior Learning (resumed)" in (output / "learn.log").read_text()


def test_real_cpu_file_pipeline_analyze_lgpfit_learn(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    topology = _write(sample_dir / "systems" / "acetate" / "topology.top")
    coordinates = _write(
        sample_dir / "systems" / "acetate" / "coordinates.gro"
    )
    specs = sample_dir / "specs.yaml"
    specs.write_text(
        yaml.safe_dump(
            {"bounds": {"sigma A": [0.5, 1.5]}, "charge_constraints": []}
        )
    )
    sample_records = {}
    for index, (parameter, value) in enumerate(
        [(0.55, 0.62), (0.8, 0.84), (1.2, 1.18), (1.45, 1.39)]
    ):
        sample_id = f"sample-{index}"
        trajectory = _write(
            sample_dir / "samples" / sample_id / "acetate" / "trajectory.xtc"
        )
        profile = _write(
            sample_dir / "samples" / sample_id / "acetate" / "profile.pmf",
            f"{value}\n",
        )
        sample_records[sample_id] = {
            "params": [parameter],
            "status": "completed",
            "outputs": [
                {
                    "system_id": "acetate",
                    "trajectory": str(trajectory.relative_to(sample_dir)),
                    "inputs": {"pmf": str(profile.relative_to(sample_dir))},
                }
            ],
        }
    (sample_dir / "samples.yaml").write_text(
        yaml.safe_dump(
            {
                "systems": [
                    {
                        "system_id": "acetate",
                        "topology": str(topology.relative_to(sample_dir)),
                        "coordinates": str(coordinates.relative_to(sample_dir)),
                    }
                ],
                "samples": sample_records,
            }
        )
    )

    reference_profile = _write(tmp_path / "reference.pmf", "1.0\n")
    routine_module = _write(
        tmp_path / "pmf.py",
        "from bff.qoi.data import QoI\n"
        "def load_profile(*, inputs, system_id, sample_id, options):\n"
        "    value = float(inputs['pmf'].read_text())\n"
        "    return QoI('ignored', [value], labels=('profile',))\n",
    )
    analyze_config = tmp_path / "analyze.yaml"
    analyze_config.write_text(
        yaml.safe_dump(
            {
                "training_samples": {
                    "manifest": str(sample_dir / "samples.yaml"),
                    "systems": [{"system_id": "acetate"}],
                    "workers": 1,
                },
                "reference": {
                    "systems": [
                        {
                            "system_id": "acetate",
                            "inputs": {"pmf": str(reference_profile)},
                        }
                    ]
                },
                "routines": [
                    {
                        "name": "pmf",
                        "callable": f"{routine_module}:load_profile",
                        "loader": "files",
                        "systems": ["acetate"],
                        "inputs": ["pmf"],
                    }
                ],
                "output": {"directory": "qoi"},
            }
        )
    )
    analyze_main(analyze_config)

    lgpfit_config = tmp_path / "lgpfit.yaml"
    lgpfit_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "pmf": {"data": "qoi/pmf.pt", "nuisance": 0.1}
                },
                "lgpfit": {
                    "model_dir": "models",
                    "reuse_models": False,
                    "n_hyper_max": 3,
                    "committee_size": 1,
                    "test_fraction": 0.25,
                    "device": "cpu",
                    "lr": 0.001,
                    "max_iter": 20,
                },
            }
        )
    )
    lgpfit_main(lgpfit_config)

    learn_config = tmp_path / "pipeline-learn.yaml"
    learn_config.write_text(
        yaml.safe_dump(
            {
                "specs": str(specs),
                "models": {
                    "pmf": {"model_path": "models/pmf.lgp", "n_eff": 1.0}
                },
                "mcmc": {
                    "total_steps": 8,
                    "warmup": 2,
                    "n_walkers": 4,
                    "progress_stride": 2,
                    "device": "cpu",
                },
                "output": {"directory": "pipeline-learn"},
            }
        )
    )
    learn_main(learn_config)

    assert (tmp_path / "qoi" / "pmf.pt").is_file()
    assert (tmp_path / "models" / "pmf.lgp").is_file()
    assert (tmp_path / "pipeline-learn" / "output" / "posterior.pt").is_file()
