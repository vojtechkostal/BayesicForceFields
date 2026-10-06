"""Parsing helpers shared by the stage configuration loaders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PathLike = str | Path


def check_keys(
    raw: Any,
    *,
    where: str,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> Mapping[str, Any]:
    """Require a mapping with only ``allowed`` keys and all ``required`` keys."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping.")
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError(
            f"{where} contains unsupported key(s): " + ", ".join(sorted(unknown))
        )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(
            f"{where} is missing required key(s): "
            + ", ".join(repr(key) for key in missing)
        )
    return raw


def strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false, got {value!r}.")
    return value


def resolve_path(
    base_dir: Path,
    path: PathLike,
    *,
    must_exist: bool = True,
    kind: str = "path",
) -> Path:
    """Resolve ``path`` relative to the configuration directory."""
    resolved = (base_dir / Path(path).expanduser()).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{kind.capitalize()} not found: {resolved}")
    return resolved
