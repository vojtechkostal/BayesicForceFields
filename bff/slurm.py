"""Run many BFF jobs as Slurm job arrays and follow their progress.

One script runs every task; ``$TASK_ID`` (0..n_tasks-1) selects the work.
Tasks are submitted in arrays of at most ``max_array_size`` tasks (Slurm's
``MaxArraySize``), rolling: a new array whenever finished tasks free slots, with
at most ``max_parallel_jobs`` tasks queued or running in all arrays together.
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


@dataclass(frozen=True)
class SlurmConfig:
    """Job-array limits, extra ``sbatch`` options, and shell lines run before
    (``setup``) and after (``teardown``, also when the job fails) each job."""

    max_parallel_jobs: int = 1
    max_array_size: int = 1000
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
        "TASK_ID=$((SLURM_ARRAY_TASK_ID + ${BFF_TASK_OFFSET:-0}))",
    ]
    if config.teardown:
        # An EXIT trap also runs after a failing command, which `set -e` ends.
        teardown = [f"    {line}" for line in config.teardown]
        lines += ["teardown() {", *teardown, "}", "trap teardown EXIT"]
    lines += [*config.setup, *commands]
    fn_script.parent.mkdir(parents=True, exist_ok=True)
    fn_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fn_script


def array_option(n_tasks: int, max_parallel_jobs: int) -> str:
    """Value of ``sbatch --array`` for tasks 0..n_tasks-1."""
    limit = "" if max_parallel_jobs == -1 else f"%{max_parallel_jobs}"
    return f"0-{n_tasks - 1}{limit}"


def submit(fn_script: Path, *, array: str, offset: int = 0) -> int:
    """Submit one job array of ``fn_script`` and return its job ID."""
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--array={array}",
            f"--export=ALL,BFF_TASK_OFFSET={offset}",
            str(fn_script),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Slurm submission of {fn_script} failed: {message}")
    try:
        return int(result.stdout.strip().split(";")[0])
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse a job ID from sbatch output {result.stdout!r}."
        ) from exc


def run_tasks(
    fn_script: Path,
    n_tasks: int,
    *,
    config: SlurmConfig,
    logger: Logger,
    poll_interval: float = 60.0,
) -> list[str]:
    """Run tasks 0..n_tasks-1 and return each task's ``<job>_<index>`` ID.

    Rolling submission: at most ``max_parallel_jobs`` tasks (``-1``: at most
    ``max_array_size``) are queued or running at any time. Whenever finished
    tasks free enough slots, the next tasks are submitted as a new array of at
    most ``max_array_size`` tasks, so the slots are refilled without waiting
    for a whole array to finish. Chunks are at least a tenth of the limit
    (the last one may be smaller), to keep the number of arrays small.
    """
    limit = (
        config.max_array_size
        if config.max_parallel_jobs == -1
        else config.max_parallel_jobs
    )
    min_chunk = max(1, min(limit, config.max_array_size) // 10)
    width = len(str(n_tasks))
    task_ids: list[str] = []
    active: dict[int, int] = {}  # job id -> tasks of arrays still in the queue
    done = 0  # tasks of arrays that have left the queue
    offset = 0  # next task to submit
    while True:
        counts = {job_id: array_task_counts(job_id, size) for job_id, size in active.items()}
        if any(count is None for count in counts.values()):
            time.sleep(poll_interval)  # transient squeue error: never guess
            continue
        for job_id, count in counts.items():
            if count["finished"] == active[job_id]:
                done += active.pop(job_id)
        in_queue = sum(c["pending"] + c["running"] for j, c in counts.items() if j in active)
        finished = done + sum(c["finished"] for j, c in counts.items() if j in active)
        while offset < n_tasks:
            size = min(limit - in_queue, config.max_array_size, n_tasks - offset)
            if size <= 0 or (size < min_chunk and size < n_tasks - offset):
                break
            job_id = submit(fn_script, array=f"0-{size - 1}", offset=offset)
            logger.kv("Slurm job array", f"{job_id}: tasks {offset}-{offset + size - 1}")
            task_ids += [f"{job_id}_{index}" for index in range(size)]
            active[job_id] = size
            in_queue += size
            offset += size
        pending = sum(c["pending"] for j, c in counts.items() if j in active)
        running = sum(c["running"] for j, c in counts.items() if j in active)
        complete = offset == n_tasks and not active
        logger.progress_status(
            f"Slurm: queued {pending:>{width}d} | "
            f"running {running:>{width}d} | "
            f"finished {finished:>{width}d}/{n_tasks}",
            finished,
            n_tasks,
            overwrite=not complete,
            write_file=complete,
        )
        if complete:
            return task_ids
        time.sleep(poll_interval)


def array_task_counts(job_id: int, n_tasks: int) -> dict[str, int] | None:
    """Count pending, running, and finished tasks of a job array.

    Returns ``None`` when ``squeue`` fails for a reason other than the job
    having left the queue, so a transient scheduler error is never mistaken
    for completion.
    """
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-r", "--noheader", "--format", "%t"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and "invalid job id" not in result.stderr.lower():
        return None
    states = result.stdout.split() if result.returncode == 0 else []
    pending = sum(state in PENDING_STATES for state in states)
    running = len(states) - pending
    return {
        "pending": pending,
        "running": running,
        "finished": max(n_tasks - len(states), 0),
    }


def wait_for_array(
    job_id: int,
    n_tasks: int,
    *,
    logger: Logger,
    done_before: int = 0,
    total: int | None = None,
    poll_interval: float = 10.0,
) -> None:
    """Block until every task of the job array has left the queue."""
    total = n_tasks if total is None else total
    width = len(str(total))
    while True:
        counts = array_task_counts(job_id, n_tasks)
        if counts is not None:
            done = counts["finished"] == n_tasks
            finished = done_before + counts["finished"]
            logger.progress_status(
                f"Slurm job {job_id}: pending {counts['pending']:>{width}d} | "
                f"running {counts['running']:>{width}d} | "
                f"finished {finished:>{width}d}/{total}",
                finished,
                total,
                overwrite=not done,
                write_file=done,
            )
            if done:
                return
        time.sleep(poll_interval)
