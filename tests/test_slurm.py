import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bff import slurm
from bff.io.logs import Logger
from bff.workflows.campaign.config import load_slurm_config
from bff.workflows.config import ConfigSection


def _load_slurm(raw: dict) -> slurm.SlurmConfig:
    config = ConfigSection(
        {"slurm": raw}, "", base_dir=Path.cwd(), allowed=("slurm",)
    )
    return load_slurm_config(config)


def test_slurm_config_rejects_array_and_validates_limits() -> None:
    config = _load_slurm(
        {"sbatch": {"time": "1:00:00"}, "max_parallel_jobs": -1, "setup": ["ml gmx"]}
    )
    assert config.max_parallel_jobs == -1
    assert config.setup == ("ml gmx",)
    assert config.submit_pause == 0.0
    # The array chunk size of BFF <= 0.5.2 is accepted and ignored.
    assert _load_slurm({"sbatch": {}, "max_array_size": 1000}).max_parallel_jobs == 1
    with pytest.raises(ValueError, match="submit_pause must be"):
        _load_slurm({"sbatch": {}, "submit_pause": -1})
    with pytest.raises(ValueError, match="array is not supported"):
        _load_slurm({"sbatch": {"array": "0-3"}})
    with pytest.raises(ValueError, match=r"max_parallel_jobs must be .* or -1"):
        _load_slurm({"sbatch": {}, "max_parallel_jobs": 0})
    with pytest.raises(ValueError, match="unsupported key"):
        _load_slurm({"sbatch": {}, "partition": "cpu"})
    with pytest.raises(ValueError, match="missing required key"):
        _load_slurm({"max_parallel_jobs": 4})
    # YAML reads an unquoted 4:00:00 as 14400 (base 60).
    with pytest.raises(ValueError, match="must be quoted"):
        _load_slurm({"sbatch": {"time": 14400}})


def test_task_script_takes_the_task_as_first_argument(tmp_path: Path) -> None:
    config = slurm.SlurmConfig(
        sbatch={"job_name": "user", "cpus_per_task": 4},
        setup=("module load gromacs",),
        teardown=("echo done",),
    )
    script = slurm.write_task_script(
        tmp_path / "run.sh",
        config=config,
        sbatch={"job_name": "bff"},
        commands=["echo $TASK"],
    ).read_text()
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=bff" in lines
    assert "#SBATCH --cpus-per-task=4" in lines
    assert not any("--array" in line for line in lines)
    assert lines[-8:] == [
        "set -eo pipefail",
        'TASK="$1"',
        "teardown() {",
        "    echo done",
        "}",
        "trap teardown EXIT",
        "module load gromacs",
        "echo $TASK",
    ]


def test_task_script_runs_teardown_after_a_failing_task(tmp_path: Path) -> None:
    config = slurm.SlurmConfig(teardown=(f"touch {tmp_path / 'torn-down'}",))
    script = slurm.write_task_script(
        tmp_path / "run.sh", config=config, commands=["false", "echo unreachable"]
    )

    result = subprocess.run(
        ["bash", str(script), "0"],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unreachable" not in result.stdout
    assert (tmp_path / "torn-down").is_file()


class _FakeQueue:
    """Plain jobs; ``per_poll`` of the oldest queued jobs finish at every squeue."""

    def __init__(self, per_poll: int, running: int = 10**9) -> None:
        self.per_poll, self.running = per_poll, running
        self.queue: list[int] = []
        self.submitted: list[tuple[int, str]] = []
        self.max_in_queue = 0

    def submit(self, script, task):
        job_id = 100 + len(self.submitted)
        self.submitted.append((job_id, task))
        self.queue.append(job_id)
        self.max_in_queue = max(self.max_in_queue, len(self.queue))
        return job_id

    def squeue(self):
        del self.queue[: self.per_poll]
        states = {j: "R" if i < self.running else "PD" for i, j in enumerate(self.queue)}
        return states | {9999: "R"}  # another job of the user


def _fake_queue(monkeypatch, per_poll: int, running: int = 10**9) -> _FakeQueue:
    queue = _FakeQueue(per_poll, running)
    monkeypatch.setattr(slurm, "submit", queue.submit)
    monkeypatch.setattr(slurm, "queued_jobs", queue.squeue)
    monkeypatch.setattr(slurm.time, "sleep", lambda seconds: None)
    return queue


def test_run_tasks_refills_every_freed_slot(tmp_path, monkeypatch, capsys) -> None:
    queue = _fake_queue(monkeypatch, per_poll=3, running=4)
    config = slurm.SlurmConfig(max_parallel_jobs=10)
    tasks = [f"s{i}" for i in range(25)]

    ids = slurm.run_tasks(
        tmp_path / "run.sh", tasks, config=config, logger=Logger("t", color=False)
    )

    assert [task for _, task in queue.submitted] == tasks
    assert queue.max_in_queue == 10
    assert ids == [str(job_id) for job_id, _ in queue.submitted]
    out = capsys.readouterr().out
    assert "pending 10 | running  0 | finished  0/25" in out  # just submitted
    assert "pending  6 | running  4 | finished  3/25" in out  # 3 done, refilled
    assert "finished 25/25" in out


def test_run_tasks_without_limit_submits_everything_at_once(
    tmp_path, monkeypatch
) -> None:
    queue = _fake_queue(monkeypatch, per_poll=10**6)
    config = slurm.SlurmConfig(max_parallel_jobs=-1)

    ids = slurm.run_tasks(
        tmp_path / "run.sh", ["a", "b", "c"], config=config, logger=Logger("t", verbose=False)
    )

    assert queue.max_in_queue == 3 and ids == ["100", "101", "102"]


def test_run_tasks_pauses_between_submissions_only_when_asked(
    tmp_path, monkeypatch
) -> None:
    queue = _fake_queue(monkeypatch, per_poll=10**6)
    sleeps = []
    monkeypatch.setattr(slurm.time, "sleep", sleeps.append)
    slurm.run_tasks(
        tmp_path / "run.sh", ["a", "b", "c"],
        config=slurm.SlurmConfig(max_parallel_jobs=-1, submit_pause=0.5),
        logger=Logger("t", verbose=False), poll_interval=60,
    )
    assert sleeps == [0.5, 0.5, 60]
    sleeps.clear()
    slurm.run_tasks(
        tmp_path / "run.sh", ["a", "b", "c"], config=slurm.SlurmConfig(max_parallel_jobs=-1),
        logger=Logger("t", verbose=False), poll_interval=60,
    )
    assert sleeps == [60] and len(queue.submitted) == 6


def test_run_tasks_waits_out_a_failing_squeue(tmp_path, monkeypatch) -> None:
    queue = _fake_queue(monkeypatch, per_poll=10)
    failures = [None]
    monkeypatch.setattr(
        slurm, "queued_jobs", lambda: failures.pop() if failures else queue.squeue()
    )

    ids = slurm.run_tasks(
        tmp_path / "run.sh", [str(i) for i in range(20)],
        config=slurm.SlurmConfig(max_parallel_jobs=10), logger=Logger("t", verbose=False),
    )

    assert len(ids) == 20 and queue.max_in_queue == 10


def _squeue(monkeypatch, returncode: int, stdout: str = "", stderr: str = "") -> None:
    monkeypatch.setattr(
        slurm.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        ),
    )


def test_queued_jobs_parse_squeue(monkeypatch) -> None:
    _squeue(monkeypatch, 0, "11 PD\n12 R\n13_4 R\n")
    assert slurm.queued_jobs() == {11: "PD", 12: "R"}


def test_transient_squeue_failure_is_never_read_as_finished(monkeypatch) -> None:
    _squeue(monkeypatch, 1, stderr="slurm_load_jobs error: timeout")
    assert slurm.queued_jobs() is None


def test_submit_returns_the_job_id(tmp_path: Path, monkeypatch) -> None:
    _squeue(monkeypatch, 0, "4242;cluster\n")
    assert slurm.submit(tmp_path / "run.sh", "s1") == 4242
    _squeue(monkeypatch, 1, stderr="invalid partition")
    with pytest.raises(RuntimeError, match="invalid partition"):
        slurm.submit(tmp_path / "run.sh", "s1")


def test_bff_command_imports_this_checkout() -> None:
    command = slurm.bff_command("md", '"$CONFIG"')
    assert command.startswith("PYTHONPATH=")
    assert command.endswith(' -m bff.cli md "$CONFIG"')


@pytest.mark.parametrize(
    ("value", "hours"),
    [
        ("00:40:00", 2 / 3),
        ("30", 0.5),
        ("30:00", 0.5),
        ("1-12", 36.0),
        ("1-12:30", 36.5),
        ("2-01:00:00", 49.0),
        (None, None),
        ("UNLIMITED", None),
    ],
)
def test_time_limit_hours_reads_sbatch_formats(value, hours) -> None:
    assert slurm.time_limit_hours(value) == pytest.approx(hours)


@pytest.mark.parametrize("value", ["1:2:3:4", "four hours"])
def test_time_limit_hours_rejects_unknown_formats(value) -> None:
    with pytest.raises(ValueError, match="Slurm time limit"):
        slurm.time_limit_hours(value)
