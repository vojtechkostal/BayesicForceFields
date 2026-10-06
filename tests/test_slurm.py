from pathlib import Path
from types import SimpleNamespace

import pytest

from bff import slurm
from bff.io.logs import Logger


def test_slurm_config_rejects_array_and_validates_limits() -> None:
    config = slurm.load_slurm_config(
        {"sbatch": {"time": "1:00:00"}, "max_parallel_jobs": -1, "setup": ["ml gmx"]}
    )
    assert config.max_parallel_jobs == -1
    assert config.setup == ("ml gmx",)
    with pytest.raises(ValueError, match="array is set by BFF"):
        slurm.load_slurm_config({"sbatch": {"array": "0-3"}})
    with pytest.raises(ValueError, match="positive or -1"):
        slurm.load_slurm_config({"sbatch": {}, "max_parallel_jobs": 0})
    with pytest.raises(ValueError, match="unsupported key"):
        slurm.load_slurm_config({"sbatch": {}, "partition": "cpu"})


def test_task_script_maps_array_index_to_global_task(tmp_path: Path) -> None:
    config = slurm.SlurmConfig(
        sbatch={"job_name": "user", "cpus_per_task": 4},
        setup=("module load gromacs",),
        teardown=("echo done",),
    )
    script = slurm.write_task_script(
        tmp_path / "run.sh",
        config=config,
        sbatch={"job_name": "bff"},
        commands=["echo $TASK_ID"],
    ).read_text()
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=bff" in lines
    assert "#SBATCH --cpus-per-task=4" in lines
    assert not any("--array" in line for line in lines)
    assert lines[-5:] == [
        "set -eo pipefail",
        "TASK_ID=$((SLURM_ARRAY_TASK_ID + ${BFF_TASK_OFFSET:-0}))",
        "module load gromacs",
        "echo $TASK_ID",
        "echo done",
    ]


@pytest.mark.parametrize(("limit", "array"), [(3, "0-9%3"), (-1, "0-9")])
def test_array_option_limits_running_tasks(limit, array) -> None:
    assert slurm.array_option(10, limit) == array


def test_run_tasks_submits_chunks_with_global_offsets(tmp_path, monkeypatch) -> None:
    submitted, waited = [], []
    monkeypatch.setattr(
        slurm,
        "submit",
        lambda script, *, array, offset: submitted.append((array, offset)) or 7,
    )
    monkeypatch.setattr(
        slurm,
        "wait_for_array",
        lambda job_id, n, **kwargs: waited.append((n, kwargs["done_before"])),
    )
    config = slurm.SlurmConfig(max_parallel_jobs=50, max_array_size=1000)

    ids = slurm.run_tasks(
        tmp_path / "run.sh", 2500, config=config, logger=Logger("t", verbose=False)
    )

    assert submitted == [("0-999%50", 0), ("0-999%50", 1000), ("0-499%50", 2000)]
    assert waited == [(1000, 0), (1000, 1000), (500, 2000)]
    assert ids[1999] == "7_999" and len(ids) == 2500


def _squeue(monkeypatch, returncode: int, stdout: str = "", stderr: str = "") -> None:
    monkeypatch.setattr(
        slurm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        ),
    )


def test_array_task_counts_parse_squeue(monkeypatch) -> None:
    _squeue(monkeypatch, 0, "PD\nR\nCG\n")
    assert slurm.array_task_counts(7, 5) == {"pending": 1, "running": 2, "finished": 2}


def test_job_that_left_the_queue_is_finished(monkeypatch) -> None:
    _squeue(monkeypatch, 1, stderr="slurm_load_jobs error: Invalid job id specified")
    assert slurm.array_task_counts(7, 5)["finished"] == 5


def test_transient_squeue_failure_is_never_read_as_finished(monkeypatch) -> None:
    _squeue(monkeypatch, 1, stderr="Socket timed out on send/recv operation")
    assert slurm.array_task_counts(7, 5) is None


def test_submit_returns_the_job_id(tmp_path: Path, monkeypatch) -> None:
    _squeue(monkeypatch, 0, "4242;cluster\n")
    assert slurm.submit(tmp_path / "run.sh", array="0-3") == 4242
    _squeue(monkeypatch, 1, stderr="invalid partition")
    with pytest.raises(RuntimeError, match="invalid partition"):
        slurm.submit(tmp_path / "run.sh", array="0-3")


def test_wait_for_array_polls_until_every_task_finished(monkeypatch, capsys) -> None:
    states = iter(
        [
            None,
            {"pending": 1, "running": 1, "finished": 0},
            {"pending": 0, "running": 0, "finished": 2},
        ]
    )
    monkeypatch.setattr(slurm, "array_task_counts", lambda job_id, n: next(states))
    sleeps = []
    monkeypatch.setattr(slurm.time, "sleep", sleeps.append)

    slurm.wait_for_array(9, 2, logger=Logger("test", color=False), poll_interval=1)

    assert sleeps == [1, 1]
    assert "finished 2/2" in capsys.readouterr().out


def test_bff_command_imports_this_checkout() -> None:
    command = slurm.bff_command("md", '"$CONFIG"')
    assert command.startswith("PYTHONPATH=")
    assert command.endswith(' -m bff.cli md "$CONFIG"')
