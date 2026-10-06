"""Run many BFF jobs as Slurm job arrays and follow their progress.

One script runs every task; ``$TASK_ID`` (0..n_tasks-1) selects the work.
Tasks are submitted in arrays of at most ``max_array_size`` tasks (Slurm's
``MaxArraySize`` and per-user submit limits), one array after another, each
running at most ``max_parallel_jobs`` tasks at a time.
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
    max_parallel_jobs: int = 1
    max_array_size: int = 1000
    sbatch: dict[str, Any] | None = None
    setup: tuple[str, ...] = ()
    teardown: tuple[str, ...] = ()


def load_slurm_config(raw: Any) -> SlurmConfig:
    """Parse the ``slurm`` section of a stage configuration."""
    if not isinstance(raw, dict):
        raise ValueError("Missing 'slurm' configuration for slurm scheduler.")
    unknown = set(raw) - {
        "max_parallel_jobs",
        "max_array_size",
        "sbatch",
        "setup",
        "teardown",
    }
    if unknown:
        raise ValueError(
            "slurm contains unsupported key(s): " + ", ".join(sorted(unknown))
        )
    if not isinstance(raw.get("sbatch"), dict):
        raise ValueError("Scheduler 'slurm' must define the 'sbatch' mapping.")
    if "array" in raw["sbatch"]:
        raise ValueError(
            "slurm.sbatch.array is set by BFF; use slurm.max_parallel_jobs to "
            "limit concurrently running tasks."
        )
    for key in ("setup", "teardown"):
        commands = raw.get(key, [])
        if not isinstance(commands, list) or not all(
            isinstance(command, str) for command in commands
        ):
            raise ValueError(f"slurm.{key} must be a list of shell commands.")
    max_parallel_jobs = int(raw.get("max_parallel_jobs", 1))
    if max_parallel_jobs == 0 or max_parallel_jobs < -1:
        raise ValueError("'slurm.max_parallel_jobs' must be positive or -1.")
    max_array_size = int(raw.get("max_array_size", 1000))
    if max_array_size <= 0:
        raise ValueError("'slurm.max_array_size' must be positive.")
    return SlurmConfig(
        max_parallel_jobs=max_parallel_jobs,
        max_array_size=max_array_size,
        sbatch=dict(raw["sbatch"]),
        setup=tuple(raw.get("setup", [])),
        teardown=tuple(raw.get("teardown", [])),
    )


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
        *config.setup,
        *commands,
        *config.teardown,
    ]
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
) -> list[str]:
    """Run tasks 0..n_tasks-1 and return each task's ``<job>_<index>`` ID."""
    task_ids: list[str] = []
    for offset in range(0, n_tasks, config.max_array_size):
        size = min(config.max_array_size, n_tasks - offset)
        job_id = submit(
            fn_script, array=array_option(size, config.max_parallel_jobs), offset=offset
        )
        logger.kv("Slurm job array", f"{job_id}: tasks {offset}-{offset + size - 1}")
        task_ids += [f"{job_id}_{index}" for index in range(size)]
        wait_for_array(job_id, size, logger=logger, done_before=offset, total=n_tasks)
    return task_ids


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
