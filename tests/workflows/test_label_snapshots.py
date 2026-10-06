"""Snapshot labeling: staging, CP2K jobs, collection, and Slurm arrays."""

from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest
import yaml

from bff.io.extxyz import write_extxyz_frames
from bff.io.logs import Logger
from bff.workflows.label_snapshots import job as job_module
from bff.workflows.label_snapshots import main as main_module
from bff.workflows.label_snapshots.config import (
    LabelSnapshotsConfig,
    SnapshotSystemConfig,
)
from bff.workflows.label_snapshots.main import collect_snapshot_splits, stage_system


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _frame(run_dir: Path, energy: float = -1.0) -> None:
    write_extxyz_frames(
        [
            {
                "atoms": ["H"],
                "positions": [[0.0, 0.0, 0.0]],
                "forces": [[0.0, 0.0, 0.0]],
                "energy": energy,
            }
        ],
        run_dir / "sp.extxyz",
    )


def _config(tmp_path: Path, **overrides) -> LabelSnapshotsConfig:
    sources = {
        name: _write(tmp_path / "sources" / name, name)
        for name in ("system.gro", "trajectory.xtc", "md.inp", "sp.inp", "atom-h.inp")
    }
    system = SnapshotSystemConfig(
        system_id="acetate",
        topology_path=sources["system.gro"],
        trajectory_path=sources["trajectory.xtc"],
        md_input_path=sources["md.inp"],
        sp_input_path=sources["sp.inp"],
        n_snapshots=3,
        single_atom_input_paths={"H": sources["atom-h.inp"]},
    )
    output_dir = tmp_path / "labels"
    options = dict(
        fn_config=tmp_path / "config.yaml",
        output_dir=output_dir,
        log=output_dir / "label-snapshots.log",
        results_manifest=output_dir / "label-results.yaml",
        systems=[system],
        cp2k_cmd="cp2k",
        job_scheduler="local",
        train_fraction=0.5,
        seed=2026,
        single_atoms=False,
        cleanup_snapshots=False,
        collection_wait_seconds=0.0,
        slurm=None,
    )
    return LabelSnapshotsConfig(**(options | overrides))


def _stage_without_trajectory(monkeypatch, config: LabelSnapshotsConfig) -> list[Path]:
    system_dir = config.output_dir / "systems" / "acetate"
    run_dirs = [
        system_dir / "snapshots" / f"snapshot-{index:04d}" for index in range(3)
    ]
    for run_dir in run_dirs:
        run_dir.mkdir(parents=True)
    monkeypatch.setattr(LabelSnapshotsConfig, "load", lambda _: config)
    monkeypatch.setattr(main_module, "resolve_cp2k_command", lambda cmd, **_: cmd)
    monkeypatch.setattr(
        main_module,
        "stage_system",
        lambda system, cfg: (system_dir, run_dirs, [], (0, 5, 9)),
    )
    return run_dirs


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
    assert all((run_dir / "md.inp").read_text() == "MD INPUT\n" for run_dir in run_dirs)
    assert all((run_dir / "sp.inp").read_text() == "SP INPUT\n" for run_dir in run_dirs)

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


def test_snapshot_job_does_not_modify_user_cp2k_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "snapshot-0000"
    md_input = _write(run_dir / "md.inp", "GFN_TYPE GFN1\n")
    _write(run_dir / "sp.inp", "SP INPUT\n")
    calls = []
    monkeypatch.setattr(job_module, "run_cp2k", lambda *args: calls.append(args[1:3]))
    monkeypatch.setattr(job_module, "write_cp2k_snapshot_extxyz", lambda _: None)

    job_module.run_snapshot_job("snapshot", run_dir, "cp2k")

    assert calls == [("md.inp", "md.out"), ("sp.inp", "sp.out")]
    assert md_input.read_text() == "GFN_TYPE GFN1\n"


def test_local_cp2k_subprocess_output_does_not_interleave(
    tmp_path: Path, capsys
) -> None:
    executable = _write(
        tmp_path / "fake-cp2k", "#!/bin/sh\nprintf 'launcher noise\\n'\nexit 0\n"
    )
    executable.chmod(0o755)
    _write(tmp_path / "input.inp")

    job_module.run_cp2k(str(executable), "input.inp", "output.out", tmp_path)

    assert "launcher noise" not in capsys.readouterr().out


def test_cp2k_command_must_be_one_executable() -> None:
    with pytest.raises(ValueError, match="single executable"):
        job_module.resolve_cp2k_command("mpirun -np 4 cp2k.psmp")
    with pytest.raises(FileNotFoundError, match="not found"):
        job_module.resolve_cp2k_command("./missing-cp2k")
    assert job_module.resolve_cp2k_command("cp2k.psmp", must_exist=False) == "cp2k.psmp"


def test_local_labeling_continues_after_a_failed_job_and_writes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    run_dirs = _stage_without_trajectory(monkeypatch, config)

    def fake_run(kind, run_dir, cp2k_cmd):
        if run_dir == run_dirs[0]:
            raise RuntimeError("CP2K failed")
        _frame(run_dir)

    monkeypatch.setattr(main_module, "run_snapshot_job", fake_run)

    main_module.main(config.fn_config)

    manifest = yaml.safe_load(config.results_manifest.read_text())
    record = manifest["systems"]["acetate"]
    assert manifest["stage"] == "label-snapshots"
    assert record["selected_frame_indices"] == [0, 5, 9]
    assert (record["train_count"], record["test_count"]) == (1, 1)
    assert record["train"] == "systems/acetate/train.extxyz"
    assert record["sources"]["trajectory"]["sha256"]
    assert record["sources"]["single_atom_inputs"]["H"]["sha256"]
    assert "snapshot-0000 failed: CP2K failed" in config.log.read_text()


def test_slurm_labeling_submits_one_job_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bff.slurm import SlurmConfig

    config = _config(
        tmp_path,
        job_scheduler="slurm",
        slurm=SlurmConfig(max_parallel_jobs=-1, sbatch={"partition": "cpu"}),
    )
    run_dirs = _stage_without_trajectory(monkeypatch, config)
    monkeypatch.setattr(main_module.slurm, "submit", lambda script, **kwargs: 42)

    def fake_wait(job_id, n_tasks, **kwargs):
        assert (job_id, n_tasks) == (42, 3)
        for run_dir in run_dirs:
            _frame(run_dir)

    monkeypatch.setattr(main_module.slurm, "wait_for_array", fake_wait)

    main_module.main(config.fn_config)

    jobs = (config.output_dir / "jobs.txt").read_text().splitlines()
    assert jobs == [str(run_dir / ".bff-job.yaml") for run_dir in run_dirs]
    assert yaml.safe_load(Path(jobs[1]).read_text()) == {
        "kind": "snapshot",
        "run_dir": str(run_dirs[1]),
        "cp2k_cmd": "cp2k",
    }
    script = (config.output_dir / "run.sh").read_text()
    assert "#SBATCH --partition=cpu" in script
    assert f"#SBATCH --output={config.output_dir}/slurm/%A_%a.out" in script
    assert 'sed -n "$((TASK_ID + 1))p"' in script
    assert 'label-snapshot-job "$JOB_CONFIG"' in script
    record = yaml.safe_load(config.results_manifest.read_text())["systems"]["acetate"]
    assert record["train_count"] + record["test_count"] == 3
