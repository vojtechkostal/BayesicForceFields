"""One staged CP2K labeling job: a snapshot or an isolated atom.

Run by the hidden ``bff label-snapshot-job <.bff-job.yaml>`` command, locally
or as one Slurm array task.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from ...io.cp2k import write_cp2k_snapshot_extxyz
from ...io.utils import load_yaml, save_yaml
from ..config import check_keys

JobKind = Literal["snapshot", "single_atom"]
JOB_STEPS: dict[str, tuple[tuple[str, str], ...]] = {
    "snapshot": (("md.inp", "md.out"), ("sp.inp", "sp.out")),
    "single_atom": (("input.inp", "atom.out"),),
}
JOB_CONFIG_NAME = ".bff-job.yaml"


def resolve_cp2k_command(cp2k_cmd: str, *, must_exist: bool = True) -> str:
    """Return the CP2K executable; ``cp2k_cmd`` must be one name or path."""
    parts = shlex.split(cp2k_cmd)
    if len(parts) != 1:
        raise ValueError("'cp2k_cmd' must be a single executable name or path.")
    executable = parts[0]
    if "/" in executable:
        path = Path(executable).expanduser()
        if must_exist and not path.exists():
            raise FileNotFoundError(f"CP2K executable not found: {path}")
        return str(path)
    if must_exist:
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"CP2K executable '{executable}' was not found on PATH."
            )
        return resolved
    return executable


def write_job_config(run_dir: Path, kind: JobKind, cp2k_cmd: str) -> Path:
    fn_job = run_dir / JOB_CONFIG_NAME
    save_yaml(
        {"kind": kind, "run_dir": str(run_dir.resolve()), "cp2k_cmd": cp2k_cmd}, fn_job
    )
    return fn_job


def run_cp2k(cp2k_cmd: str, fn_input: str, fn_output: str, cwd: Path) -> None:
    command = [cp2k_cmd, "-i", fn_input, "-o", fn_output]
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun") is not None:
        command = ["srun", *command]
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        output = (completed.stdout or "").strip()
        detail = f"\n{output[-4000:]}" if output else ""
        raise RuntimeError(
            f"CP2K command failed for {cwd / fn_input} with exit code "
            f"{completed.returncode}; inspect {cwd / fn_output}.{detail}"
        )


def run_snapshot_job(kind: JobKind, run_dir: Path, cp2k_cmd: str) -> None:
    """Run the CP2K steps of one job and convert a snapshot to extxyz."""
    for fn_input, fn_output in JOB_STEPS[kind]:
        run_cp2k(cp2k_cmd, fn_input, fn_output, run_dir)
    for pattern in ("*.wfn*", "*.restart*"):
        for path in run_dir.glob(pattern):
            path.unlink()
    if kind == "snapshot":
        write_cp2k_snapshot_extxyz(run_dir)


def main(fn_config: str | Path) -> None:
    config = check_keys(
        load_yaml(fn_config),
        where="Snapshot job config",
        allowed=("kind", "run_dir", "cp2k_cmd"),
        required=("kind", "run_dir", "cp2k_cmd"),
    )
    if config["kind"] not in JOB_STEPS:
        raise ValueError(
            "Snapshot job config 'kind' must be 'snapshot' or 'single_atom'."
        )
    run_dir = Path(config["run_dir"]).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Snapshot job directory not found: {run_dir}")
    cp2k_cmd = resolve_cp2k_command(str(config["cp2k_cmd"]), must_exist=False)
    run_snapshot_job(config["kind"], run_dir, cp2k_cmd)
