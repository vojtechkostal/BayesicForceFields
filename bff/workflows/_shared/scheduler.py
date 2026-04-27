from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ...io.schedulers import Slurm

PathLike = str | Path
SCHEDULER_CLASSES = {
    'slurm': Slurm,
}
PENDING_SLURM_STATES = {'PD', 'CF', 'CONFIGURING'}
RUNNING_SLURM_STATES = {'R', 'CG', 'COMPLETING', 'S', 'ST', 'STOPPED'}


def bff_cli_command(command: str, *args: PathLike) -> list[str]:
    """Build a Python-module invocation for one hidden or public BFF command."""
    return [sys.executable, '-m', 'bff.cli', command, *[str(arg) for arg in args]]


def repo_pythonpath_command(command: list[str]) -> str:
    """Prefix a command so scheduled jobs import this checkout first."""
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath_prefix = (
        f'PYTHONPATH={shlex.quote(str(repo_root))}'
        '${PYTHONPATH:+":$PYTHONPATH"}'
    )
    return f'{pythonpath_prefix} {shlex.join(command)}'


def build_slurm_cli_job(
    *,
    command: list[str],
    slurm_config: Any,
    sbatch: dict[str, Any],
    cwd: Path | None = None,
) -> Slurm:
    """Build a Slurm script that runs one BFF CLI command."""
    submit_script = Slurm(**sbatch)
    submit_script.add_command('set -eo pipefail')
    for cmd in slurm_config.setup:
        submit_script.add_command(cmd)
    if cwd is not None:
        submit_script.add_command(f'cd {shlex.quote(str(cwd.resolve()))}')
    submit_script.add_command(repo_pythonpath_command(command))
    for cmd in slurm_config.teardown:
        submit_script.add_command(cmd)
    return submit_script


def get_job_state_counts(
    ids: list[int],
    scheduler: str,
    chunk_size: int = 1000,
) -> dict[str, int]:
    """Count submitted, pending, running, finished, and unknown jobs."""
    if scheduler != 'slurm':
        raise NotImplementedError

    counts = {
        'submitted': len(ids),
        'pending': 0,
        'running': 0,
        'finished': 0,
        'active': 0,
        'unknown': 0,
    }
    if not ids:
        return counts

    def chunks(values: list[int], n: int):
        for i in range(0, len(values), n):
            yield values[i:i + n]

    active_ids: set[int] = set()
    for chunk in chunks(ids, chunk_size):
        ids_str = ','.join(map(str, chunk))
        res = subprocess.run(
            ['squeue', '-j', ids_str, '--noheader', '--format', '%i,%t'],
            capture_output=True,
            text=True,
            check=False,
        )
        output = res.stdout.strip()
        for line in output.splitlines():
            raw_job_id, _, raw_state = line.partition(',')
            try:
                job_id = int(raw_job_id.split('_', 1)[0])
            except ValueError:
                counts['unknown'] += 1
                continue

            state = raw_state.strip()
            active_ids.add(job_id)
            if state in PENDING_SLURM_STATES:
                counts['pending'] += 1
            elif state in RUNNING_SLURM_STATES:
                counts['running'] += 1
            else:
                counts['unknown'] += 1

    counts['active'] = counts['pending'] + counts['running'] + counts['unknown']
    counts['finished'] = max(len(set(ids)) - len(active_ids), 0)
    return counts


def get_active_jobs(ids: list[int], scheduler: str, chunk_size: int = 1000) -> int:
    """Count active jobs for the supported scheduler."""
    return get_job_state_counts(ids, scheduler, chunk_size)['active']


def wait_for_scheduler_slot(
    *,
    job_ids: list[int],
    scheduler: str,
    max_parallel_jobs: float,
    monitor: Callable[[dict[str, int]], None] | None = None,
    poll_interval: float = 5.0,
) -> None:
    """Wait until the scheduler has capacity for one more submitted job."""
    while True:
        counts = get_job_state_counts(job_ids, scheduler)
        if monitor is not None:
            monitor(counts)
        if counts['active'] < max_parallel_jobs:
            return
        time.sleep(poll_interval)


def control_jobs(
    job_ids: list[int],
    scheduler: str,
    *,
    monitor: Callable[[dict[str, int]], None] | None = None,
    poll_interval: float = 5.0,
) -> None:
    """Block until all submitted jobs finish."""
    while True:
        counts = get_job_state_counts(job_ids, scheduler)
        if monitor is not None:
            monitor(counts)
        if counts['active'] == 0:
            return
        time.sleep(poll_interval)
