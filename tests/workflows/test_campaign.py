"""Simulation campaigns shared by sample-parameters and validate."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from bff.domain.bias import BiasSpec
from bff.domain.systems import BuildSystemMetadata, write_build_system_metadata
from bff.io.logs import Logger
from bff.io.mdp import read_mdp
from bff.slurm import SlurmConfig
from bff.workflows.campaign import job as job_module
from bff.workflows.campaign import run as run_module
from bff.workflows.campaign.config import (
    SimulationCampaignConfig,
    SimulationSystemConfig,
)
from bff.workflows.campaign.job import trajectory_is_complete
from bff.workflows.campaign.run import collect_campaign, run_campaign
from bff.workflows.sample_parameters.config import SampleParametersConfig
from bff.workflows.validate.config import ValidateConfig

ROOT = Path(__file__).parents[2]
ACE_TOP = ROOT / "examples/acetate/inputs/common/topol.top"


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


def test_md_job_runs_gromacs_inside_the_sample_system_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    config = job_module.MDJobConfig(
        sample_id="000",
        params=[0.0],
        campaign_dir=campaign_dir,
        fn_specs=tmp_path / "specs.yaml",
        gmx_cmd="gmx",
        store=("xtc", "pmf"),
        cleanup=False,
        systems=[system],
    )
    monkeypatch.setattr(job_module.MDJobConfig, "load", lambda _: config)
    monkeypatch.setattr(job_module, "check_gmx_available", lambda _: None)
    monkeypatch.setattr(job_module, "Specs", lambda _: None)
    monkeypatch.setattr(
        job_module,
        "write_sample_topology",
        lambda fn_topol, specs, params, fn_out: Path(fn_out).write_text("top\n"),
    )
    monkeypatch.setattr(job_module, "trajectory_is_complete", lambda *_: True)
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command[1], Path(kwargs["cwd"])))
        if command[1] == "mdrun":
            deffnm = Path(command[command.index("-deffnm") + 1])
            deffnm.with_suffix(".gro").write_text("gro\n")
            deffnm.with_suffix(".xtc").write_text("xtc\n")
            Path(kwargs["cwd"], "production.pmf").write_text("profile\n")

    monkeypatch.setattr("bff.gromacs.subprocess.run", fake_run)

    job_module.main(tmp_path / "config.yaml")

    sample_dir = campaign_dir / "samples" / "000"
    system_dir = sample_dir / "acetate"
    assert {cwd for _, cwd in commands} == {system_dir}
    assert [name for name, _ in commands] == ["grompp", "mdrun"] * 2
    colvars_mdp = read_mdp(system_dir / "production-colvars.mdp")
    assert colvars_mdp["colvars-configfile"] == "./restraint.colvars.dat"
    assert (system_dir / "restraint.colvars.dat").is_file()
    assert (sample_dir / "gmx.log").is_file()
    result = yaml.safe_load((sample_dir / "result.yaml").read_text())
    assert result["status"] == "completed"
    assert result["outputs"][0]["trajectory"] == "samples/000/acetate/production.xtc"
    assert result["outputs"][0]["inputs"] == {
        "pmf": "samples/000/acetate/production.pmf"
    }


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


def _sample_files(campaign_dir: Path, system_ids: tuple[str, ...]) -> None:
    sample_dir = campaign_dir / "samples" / "0"
    for name in ("config.yaml", "run.out", "gmx.log"):
        _write(sample_dir / name)
    outputs = []
    for system_id in system_ids:
        for name in ("production.xtc", "production.log", "topology.top"):
            _write(sample_dir / system_id / name)
        outputs.append(
            {
                "system_id": system_id,
                "trajectory": f"samples/0/{system_id}/production.xtc",
                "inputs": {},
            }
        )
    _write(
        sample_dir / "result.yaml",
        yaml.safe_dump({"sample_id": "0", "status": "completed", "outputs": outputs}),
    )


@pytest.mark.parametrize("cleanup", [False, True])
def test_collect_campaign_merges_results_and_cleans_up(
    tmp_path: Path, cleanup: bool
) -> None:
    campaign_dir = tmp_path / "campaign"
    systems = [
        _system(campaign_dir / "systems" / system_id, system_id)
        for system_id in ("acetate", "calcium")
    ]
    systems = [
        SimulationSystemConfig(
            **(
                vars(system)
                | {
                    "topology_path": _write(
                        campaign_dir / "systems" / system.system_id / "topology.top"
                    )
                }
            )
        )
        for system in systems
    ]
    _sample_files(campaign_dir, ("acetate", "calcium"))

    collect_campaign(
        samples={"0": {"params": [0.5], "status": "failed"}},
        systems=systems,
        campaign_dir=campaign_dir,
        store=("xtc",),
        cleanup=cleanup,
    )

    manifest = yaml.safe_load((campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["0"]["status"] == "completed"
    sample_dir = campaign_dir / "samples" / "0"
    assert {path.name for path in sample_dir.iterdir()} == {
        "acetate",
        "calcium",
        "config.yaml",
        "run.out",
        "gmx.log",
    }
    expected = (
        {"production.xtc"}
        if cleanup
        else {"production.xtc", "production.log", "topology.top"}
    )
    assert {path.name for path in (sample_dir / "acetate").iterdir()} == expected


def test_local_campaign_records_failures_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = _campaign(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        sample_dir = Path(command[-1]).parent
        calls.append(sample_dir.name)
        if sample_dir.name == "0":
            return SimpleNamespace(returncode=1)
        _write(
            sample_dir / "result.yaml",
            yaml.safe_dump({"sample_id": "1", "status": "completed", "outputs": []}),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module.subprocess, "run", fake_run)
    logger = Logger("sample-parameters", fn_log=config.log, mode="w", color=False)

    run_campaign(
        config,
        fn_specs=tmp_path / "specs.yaml",
        parameter_samples=np.zeros((2, 0)),
        logger=logger,
    )

    assert calls == ["0", "1"]
    samples = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert samples["samples"]["0"]["status"] == "failed"
    assert samples["samples"]["1"]["status"] == "completed"
    console = capsys.readouterr().out
    assert "Running MD: 0/2" in console
    assert "Sample 0 failed with exit code 1" in console
    log = config.log.read_text()
    assert "Running MD: 0/2" not in log
    assert "Running MD: Done. | 2/2" in log


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

    run_campaign(
        config,
        fn_specs=tmp_path / "specs.yaml",
        parameter_samples=np.zeros((12, 0)),
        logger=Logger("validate", verbose=False),
    )

    assert submitted == [("0-4%2", 0), ("0-4%2", 5), ("0-1%2", 10)]
    script = (config.campaign_dir / "run.sh").read_text()
    assert "#SBATCH --time=01:00:00" in script
    assert f"#SBATCH --output={config.campaign_dir}/slurm/%A_%a.out" in script
    assert "TASK_ID=$((SLURM_ARRAY_TASK_ID + ${BFF_TASK_OFFSET:-0}))" in script
    assert '$(printf "%02d" "$TASK_ID")' in script
    assert 'exec >"$SAMPLE_DIR/run.out" 2>&1' in script
    assert ' -m bff.cli md "$SAMPLE_DIR/config.yaml"' in script
    assert (config.campaign_dir / "samples" / "11" / "config.yaml").is_file()
    manifest = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["03"]["job_id"] == "101_3"
    assert manifest["samples"]["11"]["job_id"] == "103_1"


def test_staged_campaign_writes_sample_topologies_without_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _campaign(tmp_path, dispatch=False)
    monkeypatch.setattr(
        run_module.subprocess, "run", lambda *a, **k: pytest.fail("MD was run")
    )

    run_campaign(
        config,
        fn_specs=tmp_path / "specs.yaml",
        parameter_samples=np.zeros((2, 0)),
        logger=Logger("sample-parameters", verbose=False),
    )

    for sample_id in ("0", "1"):
        assert (
            config.campaign_dir / "samples" / sample_id / "acetate" / "topology.top"
        ).is_file()
    manifest = yaml.safe_load((config.campaign_dir / "samples.yaml").read_text())
    assert manifest["samples"]["0"]["status"] == "staged"
