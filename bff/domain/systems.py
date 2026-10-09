"""Stable system identities and explicitly configured input files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PathValue = Path | tuple[Path, ...] | dict[str, "PathValue"] | None
_SYSTEM_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def validate_system_id(value: object, *, field: str = "system_id") -> str:
    """Return a file-safe semantic system identifier."""
    if not isinstance(value, str) or not _SYSTEM_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must match '[a-z0-9][a-z0-9._-]*', got {value!r}."
        )
    return value


def validate_unique_system_ids(
    values: list[str], *, field: str = "systems"
) -> None:
    """Raise if ``values`` repeat a system ID; ``field`` names the config key."""
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"{field} contains duplicate system_id value(s): "
            + ", ".join(repr(value) for value in duplicates)
        )


def _resolve_value(value: Any, base_dir: Path, *, field: str) -> PathValue:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = (base_dir / Path(value).expanduser()).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{field}: expected an existing file, resolved {path}; correct "
                "the path or create the required file."
            )
        return path
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field} must not be an empty list.")
        resolved = tuple(
            _resolve_value(item, base_dir, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
        if not all(isinstance(item, Path) for item in resolved):
            raise ValueError(f"{field} list entries must all be file paths.")
        return resolved  # type: ignore[return-value]
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_value(item, base_dir, field=f"{field}.{key}")
            for key, item in value.items()
        }
    raise ValueError(
        f"{field} must be a path, path list, mapping, or null; got {value!r}."
    )


@dataclass(frozen=True, slots=True)
class SystemInputs:
    """Named file roles for one system in explicit-input mode."""

    system_id: str
    inputs: dict[str, PathValue]

    def __post_init__(self) -> None:
        validate_system_id(self.system_id)
        if not isinstance(self.inputs, dict) or not self.inputs:
            raise ValueError(
                f"system {self.system_id!r} inputs must be a non-empty role mapping."
            )


def resolve_explicit_inputs(
    raw: Mapping[str, Any],
    *,
    base_dir: Path,
    system_id: str,
    field: str,
) -> SystemInputs:
    """Resolve a user-authored mapping of named file roles."""
    if not raw:
        raise ValueError(f"{field} must be a non-empty mapping.")
    inputs: dict[str, PathValue] = {}
    for role, value in raw.items():
        if not isinstance(role, str) or not role:
            raise ValueError(
                f"{field} role names must be non-empty strings, got {role!r}."
            )
        inputs[role] = _resolve_value(value, base_dir, field=f"{field}.{role}")
    return SystemInputs(system_id=system_id, inputs=inputs)
