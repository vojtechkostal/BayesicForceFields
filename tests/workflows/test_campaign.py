"""Simulation campaigns shared by sample-parameters and validate."""

import os
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bff.domain.bias import BiasSpec
from bff.domain.specs import Specs
from bff.io.mdp import read_mdp
from bff.slurm import SlurmConfig
from bff.workflows.campaign import job as job_module
from bff.workflows.campaign import run as run_module
from bff.workflows.campaign.config import (
    SimulationCampaignConfig,
    SimulationSystemConfig,
)
from bff.workflows.campaign.job import trajectory_is_complete
from bff.workflows.campaign.run import run_campaign
from bff.workflows.sample_parameters.config import SampleParametersConfig
from bff.workflows.validate.config import ValidateConfig

ROOT = Path(__file__).parents[2]
ACE_TOP = ROOT / "examples/acetate/inputs/acetate.top"


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _system(directory: Path, system_id: str = "acetate") -> SimulationSystemConfig:
    return SimulationSystemConfig(
        system_id=system_id,
        topology_path=ACE_TOP,
        coordinates_path=_write(directory / "coordinates.gro"),
        mdp_em_path=None,
        mdp_production_path=_write(
            directory / "production.mdp", "nstxout-compressed = 500\n"
        ),
        index_path=_write(directory / "index.ndx"),
        bias=BiasSpec(),
        n_steps=10,
    )


def _campaign(tmp_path: Path, **overrides) -> SimulationCampaignConfig:
    _write(
        tmp_path / "specs.yaml",
        yaml.safe_dump({"bounds": {}, "charge_constraints": []}),
    )
    options = dict(
        fn_config=tmp_path / "config.yaml",
        campaign_dir=tmp_path / "campaign",
        log=tmp_path / "campaign.log",
        gmx_cmd="gmx",
        job_scheduler="local",
        systems=[_system(tmp_path / "inputs")],
    )
    return SimulationCampaignConfig(**(options | overrides))


def _empty_draw(n_samples: int) -> dict:
    """run_campaign arguments for samples of a specification without parameters."""
    return dict(
        stage="sample-parameters",
        title="Campaign",
        specs=Specs({"bounds": {}, "charge_constraints": []}),
        draw=lambda: (np.zeros((n_samples, 0)), {"source": "test"}),
    )


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
    assert (
        config.systems[0].coordinates_path
        == (source / "systems" / "acetate" / "production.gro").resolve()
    )

    _write(source / "systems" / "acetate" / "bias.colvars.dat")
    _write(source / "systems" / "acetate" / "bias.plumed.dat")
    with pytest.raises(ValueError, match="both reserved bias files"):
        SampleParametersConfig.load(config_path)


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
    assert (
        config.systems[0].topology_path
        == (source / "systems/acetate/topology.top").resolve()
    )


def _md_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides
) -> tuple[Path, list[list[str]]]:
    """Run the MD job with fake GROMACS; return the sample dir and commands."""
    campaign_dir = tmp_path / "campaign"
    inputs = tmp_path / "inputs"
    bias = _write(inputs / "restraint.colvars.dat", "colvar {}\n")
    system = SimulationSystemConfig(
        system_id="acetate",
        topology_path=_write(inputs / "topology.top"),
        coordinates_path=_write(inputs / "coordinates.gro"),
        mdp_em_path=_write(inputs / "em.mdp", "integrator = steep\n"),
        mdp_production_path=_write(inputs / "production.mdp", "integrator = md\n"),
        index_path=_write(inputs / "index.ndx"),
        bias=BiasSpec(kind="colvars", colvars_file=bias),
        n_steps=10,
    )
    options = dict(
        sample_id="000",
        params=[0.0],
        campaign_dir=campaign_dir,
        fn_specs=tmp_path / "specs.yaml",
        gmx_cmd="gmx",
        store=("xtc", "pmf"),
        cleanup=False,
        systems=[system],
    )
    config = job_module.MDJobConfig(**(options | overrides))
    monkeypatch.setattr(job_module.MDJobConfig, "load", lambda *_: config)
    monkeypatch.setattr(job_module, "check_gmx_available", lambda _: None)
    monkeypatch.setattr(job_module, "Specs", lambda _: None)
    monkeypatch.setattr(
        job_module,
        "write_sample_topology",
        lambda fn_topol, specs, params, fn_out: Path(fn_out).write_text("top\n"),
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append([str(part) for part in command] + [str(kwargs["cwd"])])
        if command[1] == "mdrun":
            deffnm = Path(kwargs["cwd"], command[command.index("-deffnm") + 1])
            deffnm.with_suffix(".gro").write_text("gro\n")
            if deffnm.name != "production":
                return
            # Like mdrun: a final checkpoint at step 10 (nsteps), or earlier
            # when -maxh stops the run.
            step = 4 if "-maxh" in command else 10
            with deffnm.with_suffix(".log").open("a") as log:
                log.write(f"Writing checkpoint, step {step} at Mon Jan  1\n\n")
            if "xtc" in config.store:
                deffnm.with_suffix(".xtc").write_text("frames\n")
            deffnm.with_suffix(".cpt").write_text("checkpoint\n")
            Path(kwargs["cwd"], "production.pmf").write_text("profile\n")

    monkeypatch.setattr("bff.gromacs.subprocess.run", fake_run)
    job_module.main(tmp_path / "campaign.yaml", "000")
    return campaign_dir / "samples" / "000", commands


def test_md_job_runs_gromacs_inside_the_sample_system_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_dir, commands = _md_job(tmp_path, monkeypatch)

    system_dir = sample_dir / "acetate"
    assert {command[-1] for command in commands} == {str(system_dir)}
    assert [command[1] for command in commands] == ["grompp", "mdrun"] * 2
    run_mdp = read_mdp(system_dir / "production-run.mdp")
    assert run_mdp["colvars-configfile"] == "./restraint.colvars.dat"
    # The total step count is in the .tpr, so a restart cannot overshoot it.
    assert run_mdp["nsteps"] == "10"
    assert "-nsteps" not in commands[-1]
    assert (system_dir / "restraint.colvars.dat").is_file()
    assert (sample_dir / "gmx.log").is_file()
    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "completed"
    assert result["outputs"]["acetate"] == {
        "topology": "samples/000/acetate/topology.top",
        "trajectory": "samples/000/acetate/production.xtc",
        "pmf": "samples/000/acetate/production.pmf",
    }


def test_md_job_in_scratch_copies_back_only_stored_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    sample_dir, commands = _md_job(
        tmp_path, monkeypatch, scratch_dir=str(scratch), cleanup=True
    )

    assert all(command[-1].startswith(str(scratch)) for command in commands)
    assert list(scratch.iterdir()) == []
    assert sorted(path.name for path in (sample_dir / "acetate").iterdir()) == [
        "production.done",
        "production.pmf",
        "production.xtc",
        "topology.top",
    ]
    assert (sample_dir / "gmx.log").is_file()
    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "completed"


def test_md_job_with_undefined_scratch_variable_runs_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BFF_UNDEFINED_SCRATCH", raising=False)
    sample_dir, commands = _md_job(
        tmp_path, monkeypatch, scratch_dir="$BFF_UNDEFINED_SCRATCH/bff"
    )
    assert {command[-1] for command in commands} == {str(sample_dir / "acetate")}
    assert not (tmp_path / "$BFF_UNDEFINED_SCRATCH").exists()


def test_md_job_stopped_by_time_limit_is_incomplete_and_restartable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    sample_dir, commands = _md_job(
        tmp_path, monkeypatch, scratch_dir=str(scratch), cleanup=True, max_hours=1.0
    )

    production = commands[-1]
    assert production[1] == "mdrun" and "-maxh" in production
    system_dir = sample_dir / "acetate"
    assert (system_dir / "production.cpt").is_file()
    assert (system_dir / "production-run.mdp").is_file()
    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "incomplete"

    # The rerun continues from the checkpoint without grompp or minimization.
    _, commands = _md_job(tmp_path, monkeypatch, scratch_dir=str(scratch))
    assert [command[1] for command in commands] == ["mdrun"]
    assert "-cpi" in commands[0]
    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "completed"

    # A complete system is skipped.
    _, commands = _md_job(tmp_path, monkeypatch)
    assert commands == []


def test_md_job_stops_after_sigterm_and_is_restartable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slurm's SIGTERM at the time limit must not lose the sample: the job
    starts no further run and records it as incomplete."""
    run_md = job_module.run_md

    def terminated_during_minimization(deffnm, **kwargs):
        if Path(deffnm).name == "em":
            os.kill(os.getpid(), signal.SIGTERM)
        return run_md(deffnm, **kwargs)

    monkeypatch.setattr(job_module, "run_md", terminated_during_minimization)
    sample_dir, commands = _md_job(
        tmp_path, monkeypatch, scratch_dir=str(tmp_path / "scratch")
    )

    assert [Path(command[-1]).name for command in commands] == ["acetate"] * 2
    assert "em" in commands[0][commands[0].index("-o") + 1]
    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "incomplete"
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_md_job_completes_without_a_stored_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    sample_dir, _ = _md_job(
        tmp_path, monkeypatch, scratch_dir=str(scratch), cleanup=True, store=("pmf",)
    )

    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "completed"
    assert "trajectory" not in result["outputs"]["acetate"]
    assert sorted(path.name for path in (sample_dir / "acetate").iterdir()) == [
        "production.done",
        "production.pmf",
        "topology.top",
    ]

    # Rerunning the sample skips the finished system despite the pruned files.
    _, commands = _md_job(
        tmp_path, monkeypatch, scratch_dir=str(scratch), cleanup=True, store=("pmf",)
    )
    assert commands == []


@pytest.mark.parametrize(("n_frames", "complete"), [(101, True), (100, False)])
def test_trajectory_is_complete_counts_saved_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_frames: int, complete: bool
) -> None:
    class Reader:
        def __init__(self, _):
            self.n_frames = n_frames

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(job_module, "XTCReader", Reader)
    trajectory = _write(tmp_path / "production.xtc")
    mdp = _write(tmp_path / "production.mdp", "nsteps = -1\nnstxout-compressed = 500\n")
    assert trajectory_is_complete(trajectory, mdp, 50000) is complete


def test_trajectory_without_compressed_output_is_incomplete(tmp_path: Path) -> None:
    trajectory = _write(tmp_path / "production.xtc")
    mdp = _write(tmp_path / "production.mdp", "nstxout-compressed = 0\n")
    assert trajectory_is_complete(trajectory, mdp, 1000) is False


def test_cleanup_leaves_unfinished_systems_untouched(tmp_path: Path) -> None:
    """Regression test: an incomplete system's checkpoint, log, and .tpr are
    what a later ``resume: true`` needs; cleanup must not prune them just
    because the sample as a whole isn't marked completed yet."""
    system = SimulationSystemConfig(
        system_id="acetate",
        topology_path=Path("topology.top"),
        coordinates_path=Path("coordinates.gro"),
        mdp_em_path=None,
        mdp_production_path=Path("production.mdp"),
        index_path=Path("index.ndx"),
        bias=BiasSpec(),
        n_steps=10,
    )
    system_dir = tmp_path / "samples" / "0" / "acetate"
    for name in (
        "production.xtc",
        "production.cpt",
        "production.log",
        "production.tpr",
        "topology.top",
    ):
        _write(system_dir / name)

    run_module._cleanup(tmp_path, {"0": {}}, [system], store=("xtc",))

    assert {p.name for p in system_dir.iterdir()} == {
        "production.xtc",
        "production.cpt",
        "production.log",
        "production.tpr",
        "topology.top",
    }

    (system_dir / "production.done").write_text("10\n")
    run_module._cleanup(tmp_path, {"0": {}}, [system], store=("xtc",))

    assert {p.name for p in system_dir.iterdir()} == {
        "production.xtc",
        "topology.top",
        "production.done",
    }


@pytest.mark.parametrize("cleanup", [False, True])
def test_campaign_merges_job_results_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: bool
) -> None:
    config = _campaign(tmp_path, cleanup=cleanup, store=("xtc",))

    def fake_md(command, **kwargs):
        fn_campaign, sample_id = Path(command[-2]), command[-1]
        system_dir = fn_campaign.parent / "samples" / sample_id / "acetate"
        for name in (
            "production.xtc",
            "production.log",
            "topology.top",
            "production.done",
        ):
            _write(system_dir / name)
        _write(
            system_dir.parent / "result.yaml",
            yaml.safe_dump(
                {
                    "status": "completed",
                    "outputs": {
                        "acetate": {
                            "topology": f"samples/{sample_id}/acetate/topology.top",
                            "trajectory": f"samples/{sample_id}/acetate/production.xtc",
                        }
                    },
                }
            ),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module.subprocess, "run", fake_md)
    run_campaign(config, **_empty_draw(1))

    campaign_dir = config.campaign_dir
    manifest = yaml.safe_load((campaign_dir / "samples.yaml").read_text())
    record = manifest["samples"]["0"]
    assert record["status"] == "completed"
    assert record["outputs"]["acetate"]["trajectory"].endswith("production.xtc")
    assert not (campaign_dir / "samples" / "0" / "result.yaml").exists()
    expected = {"production.xtc", "topology.top", "production.done"}
    if not cleanup:
        expected.add("production.log")
    assert {p.name for p in (campaign_dir / "samples/0/acetate").iterdir()} == expected


def test_local_campaign_records_failures_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = _campaign(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        sample_id = command[-1]
        calls.append(sample_id)
        if sample_id == "0":
            return SimpleNamespace(returncode=1)
        _write(
            config.campaign_dir / "samples" / sample_id / "result.yaml",
            yaml.safe_dump({"status": "completed", "outputs": {}}),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)
    run_campaign(config, **_empty_draw(2))

    assert calls == ["0", "1"]
    samples = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert samples["samples"]["0"]["status"] == "failed"
    assert samples["samples"]["1"]["status"] == "completed"
    console = capsys.readouterr().out
    assert "Running MD: 0/2" in console
    assert "Sample 0 failed (exit code 1)" in console
    log = config.log.read_text()
    assert "Running MD: 0/2" not in log
    assert "Campaign: Done. | 1 completed, 1 failed" in log


def test_job_that_writes_no_result_is_failed_not_a_stale_earlier_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    run_campaign(_campaign(tmp_path), **_empty_draw(1))
    # Left by an earlier job whose result was never collected.
    config = _campaign(tmp_path, resume=True)
    stale = _write(
        config.campaign_dir / "samples" / "0" / "result.yaml",
        yaml.safe_dump({"status": "incomplete", "outputs": {}}),
    )

    run_campaign(config, **_empty_draw(1))

    assert not stale.exists()
    samples = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert samples["samples"]["0"]["status"] == "failed"


def test_slurm_campaign_runs_samples_as_job_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _campaign(
        tmp_path,
        job_scheduler="slurm",
        slurm=SlurmConfig(
            max_parallel_jobs=2, max_array_size=5, sbatch={"time": "01:00:00"}
        ),
    )
    submitted = []

    def fake_submit(script, *, array, offset):
        submitted.append((array, offset))
        return 100 + len(submitted)

    monkeypatch.setattr(run_module.slurm, "submit", fake_submit)
    monkeypatch.setattr(run_module.slurm, "wait_for_array", lambda *a, **k: None)

    run_campaign(config, **_empty_draw(12))

    assert submitted == [("0-4%2", 0), ("0-4%2", 5), ("0-1%2", 10)]
    script = (config.campaign_dir / "run.sh").read_text()
    assert "#SBATCH --time=01:00:00" in script
    assert f"#SBATCH --output={config.campaign_dir}/slurm/%A_%a.out" in script
    assert "TASK_ID=$((SLURM_ARRAY_TASK_ID + ${BFF_TASK_OFFSET:-0}))" in script
    assert 'SAMPLE_ID=$(sed -n "$((TASK_ID + 1))p"' in script
    assert 'exec >>"$SAMPLE_DIR/run.out" 2>&1' in script
    tasks = (config.campaign_dir / "tasks.txt").read_text().split()
    assert tasks == [f"{index:02d}" for index in range(12)]
    assert script.rstrip().endswith('campaign.yaml "$SAMPLE_ID"')
    assert (config.campaign_dir / "campaign.yaml").is_file()
    manifest = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["03"]["job_id"] == "101_3"
    assert manifest["samples"]["11"]["job_id"] == "103_1"


def test_slurm_campaign_resubmits_samples_stopped_by_the_time_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _campaign(
        tmp_path,
        job_scheduler="slurm",
        max_restarts=2,
        slurm=SlurmConfig(max_parallel_jobs=-1, sbatch={"time": "04:00:00"}),
    )
    rounds = []

    def fake_run_tasks(script, n_tasks, *, config, logger):
        tasks = (script.parent / "tasks.txt").read_text().split()
        assert len(tasks) == n_tasks
        rounds.append(tasks)
        for sample_id in tasks:
            status = (
                "incomplete" if sample_id == "2" and len(rounds) < 3 else "completed"
            )
            _write(
                script.parent / "samples" / sample_id / "result.yaml",
                yaml.safe_dump({"status": status, "outputs": {}}),
            )
        return [f"{len(rounds)}_{index}" for index in range(n_tasks)]

    monkeypatch.setattr(run_module.slurm, "run_tasks", fake_run_tasks)
    run_campaign(config, **_empty_draw(4))

    assert rounds == [["0", "1", "2", "3"], ["2"], ["2"]]
    job = yaml.safe_load((config.campaign_dir / "campaign.yaml").read_text())
    assert job["max_hours"] == pytest.approx(3.6)
    manifest = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["2"] == {
        "params": [],
        "job_id": "3_0",
        "status": "completed",
    }


def test_staged_campaign_writes_sample_topologies_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _campaign(tmp_path, dispatch=False)
    monkeypatch.setattr(
        run_module.subprocess, "run", lambda *a, **k: pytest.fail("MD was run")
    )

    run_campaign(config, **_empty_draw(2))

    for sample_id in ("0", "1"):
        assert (
            config.campaign_dir / "samples" / sample_id / "acetate" / "topology.top"
        ).is_file()
    manifest = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["0"]["status"] == "staged"


def test_local_campaign_runs_samples_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _campaign(tmp_path, local_max_parallel_jobs=2)
    # Both samples must be running at once to pass the barrier.
    barrier = threading.Barrier(2, timeout=10)

    def fake_run(command, **kwargs):
        barrier.wait()
        _write(
            config.campaign_dir / "samples" / command[-1] / "result.yaml",
            yaml.safe_dump({"status": "completed", "outputs": {}}),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)
    run_campaign(config, **_empty_draw(2))

    manifest = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert {record["status"] for record in manifest["samples"].values()} == {
        "completed"
    }


def test_overwrite_replaces_only_campaign_files(tmp_path: Path) -> None:
    config = _campaign(tmp_path, dispatch=False)
    run_campaign(config, **_empty_draw(3))
    notes = _write(config.campaign_dir / "config.yaml", "kept\n")

    with pytest.raises(FileExistsError, match="already contains a campaign"):
        run_campaign(config, **_empty_draw(1))
    overwrite = _campaign(tmp_path, dispatch=False, overwrite=True)
    run_campaign(overwrite, **_empty_draw(1))

    assert notes.read_text() == "kept\n"
    assert sorted(p.name for p in (config.campaign_dir / "samples").iterdir()) == [
        "0"
    ]
