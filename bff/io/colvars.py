"""Helpers for running GROMACS with an external Colvars input file."""

from pathlib import Path

from .mdp import patch_mdp

PathLike = str | Path


def write_mdp_with_colvars(
    fn_mdp: PathLike,
    fn_colvars: PathLike,
    fn_out: PathLike,
    *,
    seed: int | None = None,
) -> None:
    """Write an MDP file with GROMACS Colvars enabled."""
    updates = {
        "colvars-active": "yes",
        "colvars-configfile": f"./{Path(fn_colvars).name}",
    }
    if seed is not None:
        updates["colvars-seed"] = str(int(seed))
    patch_mdp(fn_mdp, updates, fn_out)
    with open(fn_out, "r+", encoding="utf-8") as handle:
        content = handle.read()
        block = "\n; colvars\n"
        if "colvars-active" in content:
            content = content.replace("colvars-active", block + "colvars-active", 1)
        handle.seek(0)
        handle.write(content)
        handle.truncate()
