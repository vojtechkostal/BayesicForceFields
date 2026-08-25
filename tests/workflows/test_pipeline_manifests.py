from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest
import torch
import yaml
from typer.testing import CliRunner

import bff
from bff.bayes.gaussian_process import LGPCommittee, LocalGaussianProcess
from bff.bayes.learning import fit_surrogates
from bff.cli import app
from bff.domain.bias import BiasSpec
from bff.domain.sample import SampleSet
from bff.domain.systems import (
    BuildSystemMetadata,
    load_build_system_metadata,
    write_build_system_metadata,
)
from bff.io.extxyz import write_extxyz_frames
from bff.io.logs import Logger
from bff.io.mdp import read_mdp
from bff.io.utils import save_json
from bff.qoi.data import QoI, QoIDataset
from bff.qoi.hbonds import compute_hydrogen_bond_qoi
from bff.qoi.rdf import compute_rdf_qoi
from bff.qoi.routines import normalize_routine_list, run_analysis_routine
from bff.workflows._shared.campaign import (
    build_submission_script,
    collect_campaign_metadata,
    write_sample_job_config,
)
from bff.workflows._shared.config import SimulationSystemConfig, SlurmConfig
from bff.workflows.build_qoi_datasets.main import (
    _shared_block_metadata,
)
from bff.workflows.build_qoi_datasets.main import (
    main as build_qoi_datasets_main,
)
from bff.workflows.fit_lgp.main import main as fit_lgp_main
from bff.workflows.label_snapshots.config import (
    LabelSnapshotsConfig,
    SnapshotSystemConfig,
)
from bff.workflows.label_snapshots.main import (
    _run_cp2k,
    collect_snapshot_splits,
    count_scheduler_jobs,
    process_system,
    run_snapshot_job,
    stage_system,
)
from bff.workflows.label_snapshots.main import (
    main as label_snapshots_main,
)
from bff.workflows.learn.config import LearnConfig
from bff.workflows.learn.main import _prepare_output
from bff.workflows.learn.main import main as learn_main
from bff.workflows.md.main import _collect_working_outputs
from bff.workflows.md.main import main as md_main
from bff.workflows.sample_parameters.config import SampleParametersConfig
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
    config = SampleParametersConfig.load(config_path)
    assert config.systems[0].coordinates_path == (
        source / "systems" / "acetate" / "production.gro"
    ).resolve()

    _write(source / "systems" / "acetate" / "bias.colvars.dat")
    _write(source / "systems" / "acetate" / "bias.plumed.dat")
    with pytest.raises(ValueError, match="both reserved bias files"):
        SampleParametersConfig.load(config_path)


def test_md_job_writes_colvars_path_from_campaign_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    inputs = tmp_path / "inputs"
    topology = _write(inputs / "topology.top")
    coordinates = _write(inputs / "coordinates.gro")
    production_mdp = _write(inputs / "production.mdp", "integrator = md\n")
    index = _write(inputs / "index.ndx")
    bias = _write(inputs / "restraint.colvars.dat", "colvar {}\n")
    specs = _write(tmp_path / "specs.yaml")
    system = SimpleNamespace(
        system_id="acetate",
        topology_path=topology,
        coordinates_path=coordinates,
        mdp_em_path=None,
        mdp_production_path=production_mdp,
        index_path=index,
        bias=BiasSpec(kind="colvars", colvars_file=bias),
        n_steps=10,
    )
    config = SimpleNamespace(
        fn_specs=specs,
        sample_id="000",
        params=[0.0],
        campaign_dir=campaign_dir,
        gmx_cmd="gmx",
        run=True,
        job_scheduler="local",
        store=("xtc", "pmf", "mdp", "dat"),
        cleanup=False,
        systems=[system],
    )
    monkeypatch.setattr(
        "bff.workflows.md.main.MDJobConfig.load",
        lambda _: config,
    )
    monkeypatch.setattr(
        "bff.workflows.md.main.check_gmx_available",
        lambda _: None,
    )
    monkeypatch.setattr("bff.workflows.md.main.Specs", lambda _: object())

    def fake_modify_topology(fn_topol, specs, params, implicit, fn_out):
        Path(fn_out).write_text(Path(fn_topol).read_text())

    monkeypatch.setattr(
        "bff.workflows.md.main.modify_topology",
        fake_modify_topology,
    )

    def fake_run(command, **kwargs):
        if "mdrun" in command:
            deffnm = Path(command[command.index("-deffnm") + 1])
            deffnm.with_suffix(".xtc").write_text("trajectory\n")
            Path(kwargs["cwd"], "production.pmf").write_text("profile\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("bff.workflows.md.main.subprocess.run", fake_run)
    monkeypatch.setattr("bff.workflows.md.main.check_success", lambda *_: True)

    md_main(tmp_path / "job.yaml")

    run_system_dir = campaign_dir / "samples/000/acetate"
    generated_mdp = read_mdp(run_system_dir / "production-colvars.mdp")
    assert generated_mdp["colvars-configfile"] == (
        "../../samples/000/acetate/restraint.colvars.dat"
    )
    assert (run_system_dir / "restraint.colvars.dat").is_file()
    assert (campaign_dir / "outputs/000/gmx.log").is_file()
    assert not list(campaign_dir.glob("gmx-*.log"))
    pmf = campaign_dir / "samples/000/acetate/production.pmf"
    assert pmf.read_text() == "profile\n"
    result = yaml.safe_load((campaign_dir / "outputs/000/result.yaml").read_text())
    assert result["outputs"][0]["inputs"]["pmf"] == (
        "samples/000/acetate/production.pmf"
    )


def test_md_job_collects_requested_files_created_in_working_directory(
    tmp_path: Path,
) -> None:
    working_dir = tmp_path / "outputs" / "000"
    system_dir = tmp_path / "samples" / "000" / "acetate-calcium"
    working_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    unchanged = _write(working_dir / "existing.pmf", "old\n")
    state_before = {
        unchanged: (unchanged.stat().st_mtime_ns, unchanged.stat().st_size)
    }
    generated = _write(working_dir / "production.pmf", "new profile\n")
    _write(working_dir / "unrequested.dat", "diagnostic\n")

    collected = _collect_working_outputs(
        working_dir,
        system_dir,
        state_before,
        ("xtc", "pmf"),
        True,
    )

    assert collected == [system_dir / "production.pmf"]
    assert (system_dir / "production.pmf").read_text() == "new profile\n"
    assert unchanged.is_file()
    assert not generated.exists()
    assert (working_dir / "unrequested.dat").is_file()


def test_md_job_collects_all_working_files_without_cleanup(tmp_path: Path) -> None:
    working_dir = tmp_path / "outputs" / "000"
    system_dir = tmp_path / "samples" / "000" / "acetate-calcium"
    working_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    generated = _write(working_dir / "production.colvars.traj")

    collected = _collect_working_outputs(
        working_dir,
        system_dir,
        {},
        ("xtc",),
        False,
    )

    assert collected == [system_dir / "production.colvars.traj"]
    assert not generated.exists()


def test_sample_operational_files_are_staged_under_outputs(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    specs = _write(tmp_path / "specs.yaml")
    system = SimpleNamespace(to_dict=lambda: {"system_id": "acetate"})

    job_config = write_sample_job_config(
        sample_id="000",
        sample=np.asarray([0.5]),
        campaign_dir=campaign_dir,
        fn_specs=specs,
        gmx_cmd="gmx",
        job_scheduler="slurm",
        store=("xtc",),
        cleanup=False,
        systems=[system],
    )
    config = SimpleNamespace(
        job_scheduler="slurm",
        campaign_dir=campaign_dir,
        slurm=SlurmConfig(sbatch={"job_name": "sample"}),
    )
    submission = build_submission_script(
        sample_id="000",
        fn_config_md=job_config,
        config=config,
    )
    assert submission is not None
    run_script = campaign_dir / "outputs/000/run.sh"
    submission.save(run_script)

    assert job_config == campaign_dir / "outputs/000/config.yaml"
    assert run_script.is_file()
    assert (
        f"#SBATCH --output={campaign_dir / 'outputs/000/run.out'}"
        in run_script.read_text()
    )
    assert not list(campaign_dir.glob("config-*.yaml"))
    assert not list(campaign_dir.glob("run-*.sh"))


def test_campaign_collects_transient_results_from_outputs(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    system_dir = campaign_dir / "systems/acetate"
    system = SimulationSystemConfig(
        system_id="acetate",
        topology_path=_write(system_dir / "topology.top"),
        coordinates_path=_write(system_dir / "coordinates.gro"),
        mdp_em_path=None,
        mdp_production_path=_write(system_dir / "production.mdp"),
        index_path=_write(system_dir / "index.ndx"),
        bias=BiasSpec(),
        n_steps=10,
    )
    result = campaign_dir / "outputs/000/result.yaml"
    result.parent.mkdir(parents=True)
    result.write_text(
        yaml.safe_dump(
            {
                "sample_id": "000",
                "status": "completed",
                "outputs": [
                    {
                        "system_id": "acetate",
                        "trajectory": "samples/000/acetate/production.xtc",
                    }
                ],
            }
        )
    )

    collect_campaign_metadata(
        samples={"000": {"params": [0.5], "status": "failed"}},
        systems=[system],
        campaign_dir=campaign_dir,
    )

    manifest = yaml.safe_load((campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["000"]["status"] == "completed"
    assert not result.exists()


def test_campaign_cleanup_keeps_only_requested_system_files(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    systems = []
    for system_id in ("acetate", "acetate-calcium"):
        source_dir = campaign_dir / "systems" / system_id
        systems.append(
            SimulationSystemConfig(
                system_id=system_id,
                topology_path=_write(source_dir / "topology.top"),
                coordinates_path=_write(source_dir / "coordinates.gro"),
                mdp_em_path=None,
                mdp_production_path=_write(source_dir / "production.mdp"),
                index_path=_write(source_dir / "index.ndx"),
                bias=BiasSpec(),
                n_steps=10,
            )
        )
        sample_system_dir = campaign_dir / "samples" / "000" / system_id
        _write(sample_system_dir / "production.xtc")
        _write(sample_system_dir / "production.log")
        _write(sample_system_dir / "topology.top")
        if system_id == "acetate-calcium":
            _write(sample_system_dir / "production.pmf")

    output_dir = campaign_dir / "outputs" / "000"
    for name in ("config.yaml", "run.sh", "run.out", "gmx.log"):
        _write(output_dir / name)
    _write(
        output_dir / "result.yaml",
        yaml.safe_dump(
            {
                "sample_id": "000",
                "status": "completed",
                "outputs": [
                    {
                        "system_id": "acetate",
                        "trajectory": "samples/000/acetate/production.xtc",
                        "inputs": {},
                    },
                    {
                        "system_id": "acetate-calcium",
                        "trajectory": (
                            "samples/000/acetate-calcium/production.xtc"
                        ),
                        "inputs": {
                            "pmf": (
                                "samples/000/acetate-calcium/production.pmf"
                            )
                        },
                    },
                ],
            }
        ),
    )

    collect_campaign_metadata(
        samples={"000": {"params": [0.5], "status": "failed"}},
        systems=systems,
        campaign_dir=campaign_dir,
        store=("xtc", "pmf"),
        remove=True,
    )

    assert not (campaign_dir / "outputs").exists()
    sample_dir = campaign_dir / "samples" / "000"
    assert {path.name for path in sample_dir.iterdir()} == {
        "acetate",
        "acetate-calcium",
        "config.yaml",
        "run.sh",
        "run.out",
    }
    assert {path.name for path in (sample_dir / "acetate").iterdir()} == {
        "production.xtc"
    }
    assert {
        path.name for path in (sample_dir / "acetate-calcium").iterdir()
    } == {"production.xtc", "production.pmf"}


def test_campaign_without_cleanup_keeps_generated_files(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    source_dir = campaign_dir / "systems" / "acetate"
    system = SimulationSystemConfig(
        system_id="acetate",
        topology_path=_write(source_dir / "topology.top"),
        coordinates_path=_write(source_dir / "coordinates.gro"),
        mdp_em_path=None,
        mdp_production_path=_write(source_dir / "production.mdp"),
        index_path=_write(source_dir / "index.ndx"),
        bias=BiasSpec(),
        n_steps=10,
    )
    sample_system_dir = campaign_dir / "samples" / "000" / "acetate"
    generated = {
        _write(sample_system_dir / "production.xtc"),
        _write(sample_system_dir / "production.log"),
        _write(sample_system_dir / "topology.top"),
    }
    output_dir = campaign_dir / "outputs" / "000"
    config = _write(output_dir / "config.yaml")
    run_script = _write(output_dir / "run.sh")
    run_output = _write(output_dir / "run.out")

    collect_campaign_metadata(
        samples={"000": {"params": [0.5], "status": "staged"}},
        systems=[system],
        campaign_dir=campaign_dir,
        store=("xtc",),
        remove=False,
    )

    assert generated == set(sample_system_dir.iterdir())
    assert output_dir.is_dir()
    sample_dir = campaign_dir / "samples" / "000"
    for source in (config, run_script, run_output):
        assert (sample_dir / source.name).read_text() == source.read_text()


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


def test_snapshot_collection_contributes_one_frame_per_job(tmp_path: Path) -> None:
    run_dirs = [tmp_path / "snapshots" / f"snapshot-{index:04d}" for index in range(2)]
    for index, run_dir in enumerate(run_dirs):
        run_dir.mkdir(parents=True)
        write_extxyz_frames(
            [
                {
                    "atoms": ["H"],
                    "positions": [[float(index), 0.0, 0.0]],
                    "forces": [[0.0, 0.0, 0.0]],
                    "energy": -float(index + 1),
                }
            ],
            run_dir / "sp.extxyz",
        )

    config = SimpleNamespace(
        collection_wait_seconds=0.0,
        seed=2026,
        train_fraction=0.5,
    )
    collected, n_train, n_test = collect_snapshot_splits(
        run_dirs,
        tmp_path,
        config,
        Logger("label-snapshots", verbose=False),
    )

    assert collected == run_dirs
    assert (n_train, n_test) == (1, 1)

    (run_dirs[1] / "sp.extxyz").unlink()
    collected, n_train, n_test = collect_snapshot_splits(
        run_dirs,
        tmp_path,
        config,
        Logger("label-snapshots", verbose=False),
    )
    assert collected == run_dirs[:1]
    assert (n_train, n_test) == (1, 0)


def test_snapshot_job_does_not_modify_user_cp2k_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "snapshot-0000"
    run_dir.mkdir()
    md_input = _write(run_dir / "md.inp", "GFN_TYPE GFN1\n")
    _write(run_dir / "sp.inp", "SP INPUT\n")
    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main._run_cp2k",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main.write_cp2k_snapshot_extxyz",
        lambda _: None,
    )

    run_snapshot_job("snapshot", run_dir, "cp2k")

    assert md_input.read_text() == "GFN_TYPE GFN1\n"


def test_local_labeling_collects_successes_after_a_failed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_dir = tmp_path / "systems" / "water"
    run_dirs = [
        system_dir / "snapshots" / f"snapshot-{index:04d}"
        for index in range(2)
    ]
    for run_dir in run_dirs:
        run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main.stage_system",
        lambda system, config: (system_dir, run_dirs, [], (0, 1)),
    )

    def fake_run(kind, run_dir, cp2k_cmd):
        if run_dir == run_dirs[0]:
            raise RuntimeError("CP2K failed")
        write_extxyz_frames(
            [
                {
                    "atoms": ["H"],
                    "positions": [[0.0, 0.0, 0.0]],
                    "forces": [[0.0, 0.0, 0.0]],
                    "energy": -1.0,
                }
            ],
            run_dir / "sp.extxyz",
        )

    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main.run_snapshot_job", fake_run
    )
    config = SimpleNamespace(
        job_scheduler="local",
        single_atoms=False,
        cp2k_cmd="cp2k",
        collection_wait_seconds=0.0,
        seed=2026,
        train_fraction=0.8,
        cleanup_snapshots=False,
    )
    system = SimpleNamespace(system_id="water")

    _, _, results = process_system(
        system,
        config,
        Logger("label-snapshots", verbose=False),
    )

    assert results is not None
    assert (results["train_count"], results["test_count"]) == (1, 0)


def test_label_snapshots_extracts_exact_frames_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    universe = mda.Universe.empty(
        2,
        n_residues=1,
        atom_resindex=np.zeros(2, dtype=int),
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["H", "O"])
    universe.add_TopologyAttr("resnames", ["SOL"])
    universe.add_TopologyAttr("resids", [1])
    universe.dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    topology = tmp_path / "system.gro"
    with mda.Writer(topology, n_atoms=2) as writer:
        writer.write(universe.atoms)
    trajectory = tmp_path / "trajectory.xtc"
    with mda.Writer(str(trajectory), n_atoms=2) as writer:
        for frame in range(3):
            universe.atoms.positions = [[frame, 0.0, 0.0], [1.0, frame, 0.0]]
            writer.write(universe.atoms)

    md_input = _write(tmp_path / "md.inp", "MD INPUT\n")
    sp_input = _write(tmp_path / "sp.inp", "SP INPUT\n")
    hydrogen_input = _write(tmp_path / "h.inp", "H INPUT\n")
    oxygen_input = _write(tmp_path / "o.inp", "O INPUT\n")
    system = SnapshotSystemConfig(
        system_id="water",
        topology_path=topology,
        trajectory_path=trajectory,
        md_input_path=md_input,
        sp_input_path=sp_input,
        n_snapshots=2,
        single_atom_input_paths={
            "H": hydrogen_input,
            "O": oxygen_input,
        },
    )
    config = SimpleNamespace(output_dir=tmp_path / "labels", single_atoms=True)

    _, run_dirs, atom_dirs, indices = stage_system(system, config)

    assert indices == (0, 2)
    assert len(run_dirs) == 2
    assert {run_dir.name for run_dir in atom_dirs} == {"h", "o"}
    assert (tmp_path / "labels/systems/water/single-atoms/h/input.inp").read_text() == (
        "H INPUT\n"
    )
    assert (tmp_path / "labels/systems/water/single-atoms/o/input.inp").read_text() == (
        "O INPUT\n"
    )
    assert all(
        (run_dir / "md.inp").read_text() == "MD INPUT\n" for run_dir in run_dirs
    )
    assert all(
        (run_dir / "sp.inp").read_text() == "SP INPUT\n" for run_dir in run_dirs
    )
    assert count_scheduler_jobs(
        SimpleNamespace(systems=[system], single_atoms=True)
    ) == 4

    missing_atom_input = SnapshotSystemConfig(
        system_id="water",
        topology_path=topology,
        trajectory_path=trajectory,
        md_input_path=md_input,
        sp_input_path=sp_input,
        n_snapshots=2,
        single_atom_input_paths={"H": hydrogen_input},
    )
    with pytest.raises(ValueError, match="missing O"):
        stage_system(missing_atom_input, config)

    too_many = SnapshotSystemConfig(
        system_id="water",
        topology_path=topology,
        trajectory_path=trajectory,
        md_input_path=md_input,
        sp_input_path=sp_input,
        n_snapshots=4,
        single_atom_input_paths=system.single_atom_input_paths,
    )
    with pytest.raises(ValueError, match="only 3 frames"):
        stage_system(too_many, config)


def test_label_snapshots_writes_provenance_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {
        name: _write(tmp_path / name, name)
        for name in (
            "system.gro",
            "trajectory.xtc",
            "md.inp",
            "sp.inp",
            "atom-h.inp",
        )
    }
    system = SnapshotSystemConfig(
        system_id="acetate",
        topology_path=sources["system.gro"],
        trajectory_path=sources["trajectory.xtc"],
        md_input_path=sources["md.inp"],
        sp_input_path=sources["sp.inp"],
        n_snapshots=2,
        single_atom_input_paths={"H": sources["atom-h.inp"]},
    )
    output_dir = tmp_path / "labels"
    config = LabelSnapshotsConfig(
        fn_config=tmp_path / "config.yaml",
        output_dir=output_dir,
        log=output_dir / "label-snapshots.log",
        results_manifest=output_dir / "label-results.yaml",
        systems=[system],
        cp2k_cmd="cp2k",
        job_scheduler="local",
        train_fraction=0.5,
        seed=2026,
        single_atoms=True,
        cleanup_snapshots=False,
        collection_wait_seconds=0.0,
        slurm=None,
    )
    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main.LabelSnapshotsConfig.load",
        lambda _: config,
    )
    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main.check_cp2k_available",
        lambda command: command,
    )

    def fake_process(*args, **kwargs):
        system_dir = output_dir / "systems" / "acetate"
        system_dir.mkdir(parents=True)
        _write(system_dir / "train.extxyz")
        _write(system_dir / "test.extxyz")
        return (
            system_dir,
            (0, 2),
            {
                "train_count": 1,
                "test_count": 1,
                "single_atom_energies": {},
                "single_atom_failures": {},
            },
        )

    monkeypatch.setattr(
        "bff.workflows.label_snapshots.main.process_system", fake_process
    )

    label_snapshots_main(config.fn_config)

    manifest = yaml.safe_load(config.results_manifest.read_text())
    record = manifest["systems"]["acetate"]
    assert manifest["stage"] == "label-snapshots"
    assert record["selected_frame_indices"] == [0, 2]
    assert (record["train_count"], record["test_count"]) == (1, 1)
    assert record["sources"]["trajectory"]["sha256"]
    assert record["sources"]["single_atom_inputs"]["H"]["sha256"]


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


def test_custom_trajectory_routine_is_inferred_without_inputs(
    tmp_path: Path,
) -> None:
    module = _write(
        tmp_path / "trajectory_routine.py",
        "from bff.qoi.data import QoI\n"
        "def calculate(*, universe, frames, system_id, sample_id, options):\n"
        "    assert universe == 'universe'\n"
        "    assert frames == slice(2, 8, 2)\n"
        "    assert system_id == 'contact' and sample_id == 'sample-0'\n"
        "    return QoI('ignored', [options['value']])\n",
    )
    routine = normalize_routine_list(
        [
            {
                "name": "distance",
                "callable": f"{module}:calculate",
                "systems": ["contact"],
                "options": {"value": 3.0},
            }
        ]
    )[0]

    result = run_analysis_routine(
        routine,
        universe="universe",
        inputs={},
        system_id="contact",
        sample_id="sample-0",
        start=2,
        stop=8,
        step=2,
    )

    assert routine.uses_trajectory
    assert result.name == "distance"
    assert result.values.tolist() == [3.0]


def test_shared_qoi_metadata_does_not_contain_recursive_references() -> None:
    blocks = [
        QoI("rdf", [1.0], settings={"bins": 200}),
        QoI("rdf", [2.0], settings={"bins": 200}),
    ]

    settings, metadata = _shared_block_metadata(blocks)

    assert settings == {"bins": 200}
    assert metadata == {}
    json.dumps(metadata)


def test_routine_loader_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported key.*loader"):
        normalize_routine_list(
            [
                {
                    "name": "rdf",
                    "type": "rdf",
                    "loader": "mdanalysis",
                    "systems": ["acetate"],
                    "selections": {"group_a": "name A", "group_b": "name B"},
                }
            ]
        )


def _trajectory_universe() -> mda.Universe:
    universe = mda.Universe.empty(
        3,
        n_residues=2,
        atom_resindex=[0, 0, 1],
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["O1", "H1", "OW"])
    universe.add_TopologyAttr("types", ["O", "H", "O"])
    universe.add_TopologyAttr("elements", ["O", "H", "O"])
    universe.add_TopologyAttr("masses", [16.0, 1.0, 16.0])
    universe.add_TopologyAttr("resnames", ["ACE", "SOL"])
    universe.add_TopologyAttr("resids", [1, 2])
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


def test_rdf_expands_group_a_into_one_curve_per_atom_type() -> None:
    universe = _trajectory_universe()
    qoi = compute_rdf_qoi(
        universe,
        group_a="resname ACE and name O1 H1",
        group_b="resname SOL and name OW",
        range=(0.0, 5.0),
        bins=20,
        pbc=True,
        smooth=False,
    )
    assert qoi.labels == ("H", "O")
    assert qoi.values_per_label == 20
    assert qoi.values.shape == (40,)
    assert np.all(np.isfinite(qoi.values))


def test_hydrogen_bonds_discover_donors_and_hydrogens_from_bonds() -> None:
    universe = _trajectory_universe()
    qoi = compute_hydrogen_bond_qoi(
        universe,
        selection="resname ACE and name O1",
        water_selection="resname SOL",
        pbc=True,
    )
    assert qoi.labels == ("ACE(O) to SOL(O)",)
    assert qoi.values.tolist() == [1.0]

    dynamic_qoi = compute_hydrogen_bond_qoi(
        universe,
        selection="resname ACE and name O1",
        water_selection="resname SOL",
        pbc=True,
        update_selections=True,
    )
    assert dynamic_qoi.labels == qoi.labels
    assert dynamic_qoi.values.tolist() == qoi.values.tolist()


def test_hydrogen_bonds_detect_ambident_nitrogen_in_both_directions() -> None:
    universe = mda.Universe.empty(
        6,
        n_residues=2,
        atom_resindex=[0, 0, 0, 1, 1, 1],
        trajectory=True,
    )
    universe.add_TopologyAttr("names", ["N", "HN1", "HN2", "OW", "HW1", "HW2"])
    universe.add_TopologyAttr("types", ["N", "H", "H", "O", "H", "H"])
    universe.add_TopologyAttr("elements", ["N", "H", "H", "O", "H", "H"])
    universe.add_TopologyAttr("masses", [14.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    universe.add_TopologyAttr("resnames", ["AMN", "SOL"])
    universe.add_TopologyAttr("resids", [1, 2])
    universe.add_TopologyAttr("bonds", [(0, 1), (0, 2), (3, 4), (3, 5)])
    universe.load_new(
        np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
              [2.5, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0]]]
        ),
        format="MEMORY",
        dimensions=np.asarray([[10.0, 10.0, 10.0, 90.0, 90.0, 90.0]]),
    )

    qoi = compute_hydrogen_bond_qoi(
        universe,
        selection="resname AMN and name N",
        water_selection="resname SOL",
    )

    assert qoi.labels == ("AMN(N) to SOL(O)", "SOL(O) to AMN(N)")
    assert qoi.values.tolist() == [1.0, 1.0]


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


def test_learning_resume_requires_matching_copied_specs(tmp_path: Path) -> None:
    config = _learn_config(tmp_path)
    _prepare_output(config)
    _write(config.output.checkpoint)

    config.output.specs.unlink()
    with pytest.raises(ValueError, match="requires the copied specifications"):
        _prepare_output(_learn_config(tmp_path, resume=True))

    shutil.copy2(config.specs, config.output.specs)
    config.output.specs.write_text("bounds: {}\n")
    with pytest.raises(ValueError, match="do not match"):
        _prepare_output(_learn_config(tmp_path, resume=True))


def test_sample_set_rejects_legacy_tarball_input(tmp_path: Path) -> None:
    archive = _write(tmp_path / "campaign.tar.gz")
    with pytest.raises(ValueError, match="must be an existing directory"):
        SampleSet.from_dir(archive)


def test_cli_exposes_refactored_workflow_names() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    for command in (
        "label-snapshots",
        "sample-parameters",
        "build-qoi-datasets",
        "fit-lgp",
    ):
        assert command in help_result.stdout
    for removed in (
        "prepare-reference",
        "evaluate-snapshots",
        "sample",
        "analyze",
        "lgpfit",
    ):
        result = runner.invoke(app, [removed, "missing.yaml"])
        assert "No such command" in result.stdout + result.stderr


def test_python_api_exposes_only_refactored_workflow_names(tmp_path: Path) -> None:
    names = (
        "label_snapshots",
        "sample_parameters",
        "build_qoi_datasets",
        "fit_lgp",
    )
    project = bff.Project(tmp_path)
    assert all(callable(getattr(bff, name)) for name in names)
    assert all(callable(getattr(project, name)) for name in names)
    assert not any(
        hasattr(bff, name)
        for name in (
            "prepare_reference",
            "evaluate_snapshots",
            "sample",
            "analyze",
            "lgpfit",
        )
    )


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
        output / "outputs" / "specs.yaml",
        output / "outputs" / "prior.pt",
        output / "outputs" / "posterior.pt",
        output / "outputs" / "mcmc.ckpt",
        output / "plots" / "marginals.pdf",
        output / "plots" / "qoi-marginals.pdf",
        output / "plots" / "corner.pdf",
    ]
    assert all(path.is_file() for path in mandatory)
    assert (output / "outputs" / "specs.yaml").read_bytes() == specs.read_bytes()
    posterior = torch.load(output / "outputs" / "posterior.pt", weights_only=False)
    assert posterior["metadata"]["specifications_fingerprint"]
    assert posterior["metadata"]["model_fingerprints"]["pmf"]
    learn_log = (output / "learn.log").read_text()
    assert "effective observations: 1.000\n\n" in learn_log
    assert "corner.pdf\n\n> Posterior learning" in learn_log

    learn_main(write_config(steps=10, resume=True))
    assert "Posterior Learning (resumed)" in (output / "learn.log").read_text()


def test_real_cpu_file_pipeline_build_qoi_fit_lgp_learn(tmp_path: Path) -> None:
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
    qoi_config = tmp_path / "build-qoi-datasets.yaml"
    qoi_config.write_text(
        yaml.safe_dump(
            {
                "training_samples": {
                    "manifest": str(sample_dir / "samples.yaml"),
                    "systems": [{"system_id": "acetate"}],
                    "workers": 2,
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
                        "systems": ["acetate"],
                        "inputs": ["pmf"],
                    }
                ],
                "output": {"directory": "qoi"},
            }
        )
    )
    build_qoi_datasets_main(qoi_config)
    qoi_log = (tmp_path / "build-qoi-datasets.log").read_text()
    reference_start = qoi_log.index("Reference QoI: in progress")
    reference_end = qoi_log.index("Done. Finished", reference_start)
    training_start = qoi_log.index("Training QoI: in progress")
    training_end = qoi_log.index("Done. Finished", training_start)
    dataset_start = qoi_log.index("QoI dataset: Done.")
    assert qoi_log[reference_end:training_start].count("\n") >= 2
    assert qoi_log[training_end:dataset_start].count("\n") >= 2

    fit_config = tmp_path / "fit-lgp.yaml"
    fit_config.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "pmf": {"data": "qoi/pmf.pt", "nuisance": 0.1}
                },
                "fit": {
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
    fit_lgp_main(fit_config)

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
    assert (tmp_path / "pipeline-learn" / "outputs" / "posterior.pt").is_file()
