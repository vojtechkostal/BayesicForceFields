"""Read and patch GROMACS MDP files."""

from collections.abc import Mapping
from pathlib import Path

PathLike = str | Path


def _key(name: str) -> str:
    """GROMACS reads MDP keys case-insensitively and ``_`` as ``-``."""
    return name.strip().lower().replace("_", "-")


def read_mdp(fn_mdp: PathLike) -> dict[str, str]:
    """MDP settings as ``{key: value}``, keys normalized as by :func:`_key`."""
    content: dict[str, str] = {}
    for line in Path(fn_mdp).read_text().splitlines():
        setting = line.split(";", 1)[0]
        if "=" in setting:
            key, value = setting.split("=", 1)
            content[_key(key)] = value.strip()
    return content


def patch_mdp(
    fn_mdp: PathLike, updates: Mapping[str, object], fn_out: PathLike
) -> None:
    """Copy an MDP file with the values of ``updates`` replaced or appended.

    Every other line, comments included, is copied verbatim.
    """
    pending = {_key(key): str(value) for key, value in updates.items()}
    lines = []
    for line in Path(fn_mdp).read_text().splitlines():
        setting = line.split(";", 1)[0]
        if "=" in setting:
            key = _key(setting.split("=", 1)[0])
            if key in pending:
                line = f"{key:<25} = {pending.pop(key)}"
        lines.append(line)
    lines += [f"{key:<25} = {value}" for key, value in pending.items()]
    Path(fn_out).write_text("\n".join(lines) + "\n")
