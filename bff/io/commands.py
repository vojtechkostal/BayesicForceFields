import shlex
from collections.abc import Sequence
from pathlib import Path


def split_command(command: str | Sequence[str]) -> list[str]:
    """Normalize a shell-style command into a subprocess argument vector."""
    if isinstance(command, str):
        parts = shlex.split(command)
    else:
        parts = [str(part) for part in command]

    if not parts:
        raise ValueError("Command must not be empty.")
    return parts


def build_command(command: str | Sequence[str], *args: str | Path) -> list[str]:
    """Append stringified arguments to a normalized command vector."""
    return [*split_command(command), *(str(arg) for arg in args)]
