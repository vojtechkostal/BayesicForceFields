"""Stable system identities, explicit inputs, and stage-local metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..io.utils import load_yaml, save_yaml

PathValue = Path | tuple[Path, ...] | dict[str, "PathValue"] | None
_SYSTEM_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb "
    "Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os "
    "Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md "
    "No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)


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
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"{field} contains duplicate system_id value(s): "
            + ", ".join(repr(value) for value in duplicates)
        )


def load_system_ids(raw: Any, *, field: str = "systems") -> tuple[str, ...]:
    """Load an explicit ordered list of system IDs."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty list of system IDs.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            if set(item) != {"system_id"}:
                raise ValueError(
                    f"{field}[{index}] must contain only 'system_id', got "
                    f"{sorted(item)}."
                )
            item = item["system_id"]
        values.append(validate_system_id(item, field=f"{field}[{index}].system_id"))
    validate_unique_system_ids(values, field=field)
    return tuple(values)


def _resolve_value(value: Any, base_dir: Path, *, field: str) -> PathValue:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = (base_dir / value).resolve()
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
    system_name: str | None = None

    def __post_init__(self) -> None:
        validate_system_id(self.system_id)
        if self.system_name is not None and not isinstance(self.system_name, str):
            raise ValueError("system_name must be a string or null.")
        if not isinstance(self.inputs, dict) or not self.inputs:
            raise ValueError(
                f"system {self.system_id!r} inputs must be a non-empty role mapping."
            )

    def require_path(self, role: str) -> Path:
        value = self.inputs.get(role)
        if not isinstance(value, Path):
            raise ValueError(
                f"system {self.system_id!r}, input role {role!r}: expected one "
                f"path, got {value!r}."
            )
        return value

    def optional_path(self, role: str) -> Path | None:
        value = self.inputs.get(role)
        if value is None:
            return None
        if not isinstance(value, Path):
            raise ValueError(
                f"system {self.system_id!r}, input role {role!r}: expected one "
                f"path or null, got {value!r}."
            )
        return value


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


def _metadata_path(root: str | Path, system_id: str) -> Path:
    return (
        Path(root).resolve()
        / "systems"
        / validate_system_id(system_id)
        / "system.yaml"
    )


def _load_metadata(
    root: str | Path, system_id: str, *, expected: set[str]
) -> dict[str, Any]:
    path = _metadata_path(root, system_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"system {system_id!r}: expected stage metadata at {path}; point "
            "'source' to the stage output root or regenerate the stage."
        )
    data = load_yaml(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"System metadata {path} must contain a mapping.")
    unknown = set(data) - expected
    missing = expected - {"system_name"} - set(data)
    if unknown or missing:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unsupported " + ", ".join(sorted(unknown)))
        raise ValueError(f"System metadata {path} is invalid: {'; '.join(details)}.")
    if data.get("system_name") is not None and not isinstance(data["system_name"], str):
        raise ValueError(f"system_name in {path} must be a string or null.")
    return dict(data)


def _box(value: Any, *, path: Path) -> tuple[float, ...]:
    box = np.asarray(value, dtype=float)
    if (
        box.shape != (6,)
        or not np.all(np.isfinite(box))
        or np.any(box[:3] <= 0)
        or np.any(box[3:] <= 0)
        or np.any(box[3:] > 180)
    ):
        raise ValueError(
            f"box in {path} must contain six finite values with positive lengths "
            "and angles in (0, 180], "
            f"got {value!r}."
        )
    return tuple(float(item) for item in box)


def _integer(value: Any, *, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} in {path} must be an integer, got {value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class BuildSystemMetadata:
    system_name: str | None
    charge: int
    multiplicity: int
    box: tuple[float, ...]
    maxwarn: int
    production_steps: int


def write_build_system_metadata(
    root: str | Path, system_id: str, metadata: BuildSystemMetadata
) -> Path:
    path = _metadata_path(root, system_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(
        {
            "system_name": metadata.system_name,
            "charge": metadata.charge,
            "multiplicity": metadata.multiplicity,
            "box": list(metadata.box),
            "maxwarn": metadata.maxwarn,
            "production_steps": metadata.production_steps,
        },
        path,
    )
    return path


def load_build_system_metadata(
    root: str | Path, system_id: str
) -> BuildSystemMetadata:
    expected = {
        "system_name", "charge", "multiplicity", "box", "maxwarn", "production_steps"
    }
    data = _load_metadata(root, system_id, expected=expected)
    path = _metadata_path(root, system_id)
    charge = _integer(data["charge"], field="charge", path=path)
    multiplicity = _integer(data["multiplicity"], field="multiplicity", path=path)
    maxwarn = _integer(data["maxwarn"], field="maxwarn", path=path)
    steps = _integer(data["production_steps"], field="production_steps", path=path)
    if steps <= 0:
        raise ValueError(f"production_steps in {path} must be positive, got {steps}.")
    if multiplicity <= 0:
        raise ValueError(
            f"multiplicity in {path} must be positive, got {multiplicity}."
        )
    if maxwarn < 0:
        raise ValueError(f"maxwarn in {path} must be non-negative, got {maxwarn}.")
    return BuildSystemMetadata(
        system_name=data.get("system_name"),
        charge=charge,
        multiplicity=multiplicity,
        box=_box(data["box"], path=path),
        maxwarn=maxwarn,
        production_steps=steps,
    )


@dataclass(frozen=True, slots=True)
class ReferenceSystemMetadata:
    system_name: str | None
    charge: int
    multiplicity: int
    box: tuple[float, ...]
    snapshot_count: int
    elements: tuple[str, ...]


def write_reference_system_metadata(
    root: str | Path, system_id: str, metadata: ReferenceSystemMetadata
) -> Path:
    path = _metadata_path(root, system_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(
        {
            "system_name": metadata.system_name,
            "charge": metadata.charge,
            "multiplicity": metadata.multiplicity,
            "box": list(metadata.box),
            "snapshot_count": metadata.snapshot_count,
            "elements": list(metadata.elements),
        },
        path,
    )
    return path


def load_reference_system_metadata(
    root: str | Path, system_id: str
) -> ReferenceSystemMetadata:
    expected = {
        "system_name", "charge", "multiplicity", "box", "snapshot_count", "elements"
    }
    data = _load_metadata(root, system_id, expected=expected)
    path = _metadata_path(root, system_id)
    charge = _integer(data["charge"], field="charge", path=path)
    multiplicity = _integer(data["multiplicity"], field="multiplicity", path=path)
    count = _integer(data["snapshot_count"], field="snapshot_count", path=path)
    if count <= 0:
        raise ValueError(f"snapshot_count in {path} must be positive, got {count}.")
    if multiplicity <= 0:
        raise ValueError(
            f"multiplicity in {path} must be positive, got {multiplicity}."
        )
    raw_elements = data["elements"]
    if not isinstance(raw_elements, list) or not raw_elements:
        raise ValueError(f"elements in {path} must be a non-empty list.")
    elements = tuple(str(value) for value in raw_elements)
    if len(set(elements)) != len(elements) or list(elements) != sorted(elements):
        raise ValueError(
            f"elements in {path} must be unique and sorted, got {elements!r}."
        )
    unknown = [element for element in elements if element not in _ELEMENTS]
    if unknown:
        raise ValueError(
            f"elements in {path} contains unknown element(s): {unknown!r}."
        )
    return ReferenceSystemMetadata(
        system_name=data.get("system_name"),
        charge=charge,
        multiplicity=multiplicity,
        box=_box(data["box"], path=path),
        snapshot_count=count,
        elements=elements,
    )
