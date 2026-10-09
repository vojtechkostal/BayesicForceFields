"""Helpers for running GROMACS with an external Colvars input file."""

import os
from pathlib import Path

from .mdp import patch_mdp

PathLike = str | Path


def write_mdp_with_colvars(
    fn_mdp: PathLike,
    fn_colvars: PathLike,
    fn_out: PathLike,
    *,
    seed: int | None = None,
    working_dir: PathLike | None = None,
) -> None:
    """Write an MDP file with Colvars enabled for a GROMACS working directory."""
    fn_colvars = Path(fn_colvars).resolve()
    if working_dir is None:
        working_dir = Path(fn_out).resolve().parent
    else:
        working_dir = Path(working_dir).resolve()
    colvars_configfile = os.path.relpath(fn_colvars, start=working_dir)
    if Path(colvars_configfile).parent == Path("."):
        colvars_configfile = f"./{colvars_configfile}"

    updates = {
        "colvars-active": "yes",
        "colvars-configfile": colvars_configfile,
    }
    if seed is not None:
        updates["colvars-seed"] = str(int(seed))
    patch_mdp(fn_mdp, updates, fn_out)
