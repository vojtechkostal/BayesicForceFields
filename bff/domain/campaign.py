"""Sampling campaign metadata serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..io.utils import load_yaml, save_yaml
from .systems import validate_system_id, validate_unique_system_ids


def _serialize(value: Any, base_dir: Path) -> Any:
    if isinstance(value, Path):
        path = value.resolve()
        try:
            return str(path.relative_to(base_dir))
        except ValueError:
            return str(path)
    if isinstance(value, (list, tuple)):
        return [_serialize(item, base_dir) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item, base_dir) for key, item in value.items()}
    return value


def load_sample_manifest(filename: str | Path) -> dict[str, Any]:
    """Load and validate sampling campaign handoff metadata."""
    path = Path(filename).resolve()
    data = load_yaml(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"Sample metadata {path} must contain a mapping.")
    unknown_top = set(data) - {"systems", "samples"}
    if unknown_top:
        raise ValueError(
            f"Sample metadata {path} contains unsupported key(s): "
            + ", ".join(sorted(unknown_top))
        )
    systems = data.get("systems")
    samples = data.get("samples")
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"Sample metadata {path}.systems must be a non-empty list.")
    if not isinstance(samples, Mapping):
        raise ValueError(f"Sample metadata {path}.samples must be a mapping.")
    system_ids: list[str] = []
    for index, record in enumerate(systems):
        if not isinstance(record, Mapping) or "system_id" not in record:
            raise ValueError(
                f"Sample metadata {path}.systems[{index}] must define system_id."
            )
        unknown = set(record) - {
            "system_id", "topology", "coordinates", "mdp", "index", "bias", "n_steps"
        }
        if unknown:
            raise ValueError(
                f"Sample metadata {path}.systems[{index}] contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )
        system_ids.append(
            validate_system_id(record["system_id"], field=f"systems[{index}].system_id")
        )
        for role in ("topology", "coordinates"):
            value = record.get(role)
            resolved = path.parent / value if isinstance(value, str) else None
            if resolved is None or not resolved.resolve().is_file():
                raise ValueError(
                    f"Sample metadata {path}.systems[{index}].{role}: expected an "
                    f"existing relative file path, got {value!r}."
                )
    validate_unique_system_ids(system_ids, field="sample metadata systems")
    return dict(data)


def write_sample_manifest(
    systems: list[Mapping[str, Any]], samples: Mapping[str, Any], filename: str | Path
) -> Path:
    """Write sampling metadata with paths relative to the metadata file."""
    path = Path(filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    system_ids = [
        validate_system_id(record.get("system_id"), field="systems[].system_id")
        for record in systems
    ]
    validate_unique_system_ids(system_ids, field="sample metadata systems")
    save_yaml(
        {
            "systems": _serialize(systems, path.parent),
            "samples": _serialize(samples, path.parent),
        },
        path,
    )
    return path
