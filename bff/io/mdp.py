from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Union


PathLike = Union[str, Path]


def read_mdp(fn_mdp: PathLike) -> OrderedDict[str, str]:
    """Read a GROMACS MDP file while preserving comments and blank lines.

    Parameters
    ----------
    fn_mdp
        Path to the input MDP file.

    Returns
    -------
    collections.OrderedDict
        Ordered mapping of MDP keys to values. Comment and blank lines are
        stored under synthetic keys starting with ``C`` and ``B``.
    """
    content: OrderedDict[str, str] = OrderedDict()
    i_comment = 0
    i_blank = 0
    with open(fn_mdp, "r") as handle:
        for line in handle:
            if line.startswith(";"):
                content[f"C{i_comment:03d}"] = line
                i_comment += 1
            elif line.strip() == "":
                content[f"B{i_blank:03d}"] = line
                i_blank += 1
            else:
                key, value = line.split("=", 1)
                content[key.strip()] = value.strip()
    return content


def write_mdp(content: Mapping[str, str], fn_out: PathLike) -> None:
    """Write an MDP mapping back to disk.

    Parameters
    ----------
    content
        Ordered MDP mapping produced by :func:`read_mdp`.
    fn_out
        Output MDP path.
    """
    with open(fn_out, "w") as handle:
        for key, value in content.items():
            if key.startswith("C") or key.startswith("B"):
                handle.write(value)
            else:
                handle.write(f"{key:<25} = {value}\n")


def patch_mdp(
    fn_mdp: PathLike,
    updates: Mapping[str, str],
    fn_out: PathLike,
) -> None:
    """Write a modified MDP file with a small set of updated parameters.

    Parameters
    ----------
    fn_mdp
        Input MDP file.
    updates
        Mapping of MDP keys to replacement values.
    fn_out
        Output MDP file.
    """
    content = read_mdp(fn_mdp)
    for key, value in updates.items():
        content[str(key)] = str(value)
    write_mdp(content, fn_out)


def get_n_frames_target(fn_mdp: PathLike) -> tuple[int | None, int | None]:
    """Extract the expected number of saved trajectory frames.

    Parameters
    ----------
    fn_mdp
        Input MDP file.

    Returns
    -------
    tuple
        Number of saved frames and trajectory stride. If ``nsteps`` is zero or
        missing, both entries are returned as ``None``.
    """
    mdp_data = read_mdp(fn_mdp)
    n_steps = int(mdp_data.get("nsteps", 0))
    if n_steps <= 0:
        return None, None
    stride = int(mdp_data["nstxout-compressed"])
    return int(n_steps / stride), stride
