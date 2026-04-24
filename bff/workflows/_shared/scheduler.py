from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ...io.schedulers import Slurm

PathLike = str | Path
SCHEDULER_CLASSES = {
    'slurm': Slurm,
}


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


def get_active_jobs(ids: list[int], scheduler: str, chunk_size: int = 1000) -> int:
    """Count active jobs for the supported scheduler."""
    if scheduler != 'slurm':
        raise NotImplementedError

    def chunks(values: list[int], n: int):
        for i in range(0, len(values), n):
            yield values[i:i + n]

    n_active = 0
    for chunk in chunks(ids, chunk_size):
        ids_str = ','.join(map(str, chunk))
        res = subprocess.run(
            ['squeue', '-j', ids_str, '--noheader', '--format', '%i,%t'],
            capture_output=True,
            text=True,
            check=False,
        )
        output = res.stdout.strip()
        if output:
            n_active += len(output.splitlines())

    return n_active


def wait_for_scheduler_slot(
    *,
    job_ids: list[int],
    scheduler: str,
    max_parallel_jobs: float,
) -> None:
    """Wait until the scheduler has capacity for one more submitted job."""
    while get_active_jobs(job_ids, scheduler) >= max_parallel_jobs:
        time.sleep(5)


def control_jobs(job_ids: list[int], scheduler: str) -> None:
    """Block until all submitted jobs finish."""
    while get_active_jobs(job_ids, scheduler) != 0:
        time.sleep(5)
