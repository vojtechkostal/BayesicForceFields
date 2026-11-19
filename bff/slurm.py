"""Run many BFF jobs as plain Slurm jobs and follow their progress.

One script runs every task; each job gets its task argument (for a campaign:
the sample ID) as ``$1``. One job per task, rolling: at most
``max_parallel_jobs`` of these jobs are queued or running at a time, and freed
slots are refilled at every poll (one ``squeue --me`` call). No job arrays:
no ``MaxArraySize`` limit, and an idle slot never waits for other tasks.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io.logs import Logger

PENDING_STATES = {"PD", "CF", "CONFIGURING"}
POLL_INTERVAL = 60.0  # s between squeue calls


@dataclass(frozen=True)
class SlurmConfig:
    """Job limits, extra ``sbatch`` options, and shell lines run before
    (``setup``) and after (``teardown``, also when the job fails) each job.

    ``max_parallel_jobs``: jobs queued or running at once (-1: no limit).
    ``submit_pause``: seconds between consecutive ``sbatch`` calls.
    """

    max_parallel_jobs: int = 1
    submit_pause: float = 0.0
    sbatch: dict[str, Any] | None = None
    setup: tuple[str, ...] = ()
    teardown: tuple[str, ...] = ()


def time_limit_hours(value: Any) -> float | None:
    """Hours of an ``sbatch --time`` value, or ``None`` when unset or unlimited.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and
    ``D-HH:MM:SS``.
    """
    if value is None or str(value).strip().lower() in {"", "infinite", "unlimited"}:
        return None
    text = str(value).strip()
    days, _, clock = text.rpartition("-") if "-" in text else ("0", "", text)
    try:
        parts = [int(part) for part in clock.split(":")]
        n_days = int(days)
    except ValueError as exc:
        raise ValueError(f"Cannot parse the Slurm time limit {value!r}.") from exc
    if len(parts) > 3:
        raise ValueError(f"Cannot parse the Slurm time limit {value!r}.")
    if "-" in text:
        hours, minutes, seconds = (parts + [0, 0])[:3]
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, *parts
    else:
        hours, minutes, seconds = 0, parts[0], 0
    return 24 * n_days + hours + minutes / 60 + seconds / 3600


def bff_command(command: str, *args: str) -> str:
    """Shell command running a BFF CLI command with this checkout importable.

    Arguments are inserted verbatim so they may reference shell variables.
    """
    repo_root = Path(__file__).resolve().parents[1]
    return (
        f'PYTHONPATH={shlex.quote(str(repo_root))}${{PYTHONPATH:+":$PYTHONPATH"}} '
        f"{shlex.quote(sys.executable)} -m bff.cli {command} {' '.join(args)}"
    )


def write_task_script(
    fn_script: Path,
    *,
    config: SlurmConfig,
    commands: list[str],
    sbatch: dict[str, Any] | None = None,
) -> Path:
    """Write the sbatch script run by every task; ``sbatch`` overrides config."""
    options = dict(config.sbatch or {}) | dict(sbatch or {})
    lines = ["#!/bin/bash"]
    lines += [
        f"#SBATCH --{key.replace('_', '-')}={value}" for key, value in options.items()
    ]
    lines += [
        "",
        "set -eo pipefail",
        'TASK="$1"',
    ]
    if config.teardown:
        # An EXIT trap also runs after a failing command, which `set -e` ends.
        teardown = [f"    {line}" for line in config.teardown]
        lines += ["teardown() {", *teardown, "}", "trap teardown EXIT"]
    lines += [*config.setup, *commands]
    fn_script.parent.mkdir(parents=True, exist_ok=True)
    fn_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fn_script


def submit(fn_script: Path, task: str) -> int:
    """Submit one job of ``fn_script`` with argument ``task``; return its job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable", str(fn_script), task],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Slurm submission of {fn_script} {task} failed: {message}")
    try:
        return int(result.stdout.strip().split(";")[0])
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse a job ID from sbatch output {result.stdout!r}."
        ) from exc


def queued_jobs() -> dict[int, str] | None:
    """State code of every job of the user in the queue, by job ID.

    Returns ``None`` when ``squeue`` fails, so a transient scheduler error is
    never mistaken for jobs that have finished.
    """
    result = subprocess.run(
        ["squeue", "--me", "--noheader", "--format", "%i %t"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    jobs = {}
    for line in result.stdout.splitlines():
        job_id, _, state = line.strip().partition(" ")
        if job_id.isdigit():  # array tasks (123_4) of other work are not ours
            jobs[int(job_id)] = state.strip()
    return jobs


def run_tasks(
    fn_script: Path,
    tasks: list[str],
    *,
    config: SlurmConfig,
    logger: Logger,
    poll_interval: float = POLL_INTERVAL,
) -> list[str]:
    """Run one job per task (``sbatch fn_script <task>``); return the job IDs.

    At most ``max_parallel_jobs`` of these jobs are queued or running; at
    every poll, as many new jobs are submitted as slots are free. Returns
    when every job has left the queue.
    """
    n_tasks = len(tasks)
    limit = n_tasks if config.max_parallel_jobs == -1 else config.max_parallel_jobs
    width = len(str(n_tasks))
    job_ids: list[str] = []
    active: set[int] = set()
    while True:
        queue = queued_jobs() if active else {}
        if queue is None:
            time.sleep(poll_interval)  # transient squeue error: never guess
            continue
        active &= set(queue)
        while len(job_ids) < n_tasks and len(active) < limit:
            if job_ids and config.submit_pause > 0:
                time.sleep(config.submit_pause)
            job_id = submit(fn_script, tasks[len(job_ids)])
            job_ids.append(str(job_id))
            active.add(job_id)
        # Jobs submitted in this poll are not in the queue snapshot: pending.
        running = sum(queue.get(j, "PD") not in PENDING_STATES for j in active)
        finished = len(job_ids) - len(active)
        complete = finished == n_tasks
        logger.progress_status(
            f"Slurm: pending {len(active) - running:>{width}d} | "
            f"running {running:>{width}d} | "
            f"finished {finished:>{width}d}/{n_tasks}",
            finished,
            n_tasks,
            overwrite=not complete,
            write_file=complete,
        )
        if complete:
            return job_ids
        time.sleep(poll_interval)
