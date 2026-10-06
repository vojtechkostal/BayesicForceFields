"""Run GROMACS with optional Colvars or PLUMED biasing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .domain.bias import BiasSpec
from .io.colvars import write_mdp_with_colvars
from .io.commands import build_command
from .io.plumed import ensure_plumed_kernel


def check_gmx_available(gmx_cmd: str = "gmx") -> None:
    """Fail early when the configured GROMACS command cannot be executed."""
    try:
        subprocess.run(
            build_command(gmx_cmd, "--version"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"GROMACS command {gmx_cmd!r} is not available.\n"
            "Make sure the executable is on PATH in the job environment, "
            "or set 'gmx_cmd' to the correct command."
        ) from exc


def run_md(
    deffnm: Path,
    *,
    mdp: Path,
    topology: Path,
    coordinates: Path,
    index: Path | None = None,
    bias: BiasSpec | None = None,
    gmx_cmd: str = "gmx",
    n_steps: int = -2,
    maxwarn: int = 0,
    mdrun_args: tuple[str, ...] = (),
    log: Path,
) -> None:
    """Run ``grompp`` and ``mdrun`` with all outputs named ``deffnm.*``.

    GROMACS runs in the directory of ``deffnm``, so auxiliary files such as
    ``mdout.mdp`` or bias outputs stay next to the run. ``n_steps=-2`` keeps
    the step count of the MDP file. A Colvars bias is copied into the run
    directory and referenced from a generated ``<deffnm>-colvars.mdp``.
    """
    deffnm = Path(deffnm).resolve()
    run_dir = deffnm.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    env = None
    mdrun_args = list(mdrun_args)
    if bias is not None and bias.kind == "colvars":
        local_bias = run_dir / bias.colvars_file.name
        if bias.colvars_file.resolve() != local_bias:
            shutil.copy2(bias.colvars_file, local_bias)
        mdp_run = run_dir / f"{deffnm.name}-colvars.mdp"
        write_mdp_with_colvars(mdp, local_bias, mdp_run, working_dir=run_dir)
        mdp = mdp_run
    elif bias is not None and bias.kind == "plumed":
        env = dict(os.environ)
        env.setdefault("PLUMED_KERNEL", str(ensure_plumed_kernel()))
        mdrun_args += ["-plumed", str(Path(bias.plumed_file).resolve())]

    grompp = build_command(
        gmx_cmd,
        "grompp",
        "-f", Path(mdp).resolve(),
        "-c", Path(coordinates).resolve(),
        "-p", Path(topology).resolve(),
        "-o", deffnm.with_suffix(".tpr"),
        "-maxwarn", str(maxwarn),
    )  # fmt: skip
    if index is not None:
        grompp += ["-n", str(Path(index).resolve())]
    mdrun = build_command(
        gmx_cmd, "mdrun", "-deffnm", deffnm, "-nsteps", str(n_steps), *mdrun_args
    )
    with open(log, "a", encoding="utf-8") as handle:
        subprocess.run(grompp, stdout=handle, stderr=handle, check=True, cwd=run_dir)
        subprocess.run(
            mdrun, stdout=handle, stderr=handle, check=True, cwd=run_dir, env=env
        )
