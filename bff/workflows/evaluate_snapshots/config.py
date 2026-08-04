from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ...domain.systems import (
    PathValue,
    SystemInputs,
    load_reference_system_metadata,
    resolve_explicit_inputs,
    validate_system_id,
    validate_unique_system_ids,
)
from ...io.cp2k import get_cp2k_single_atom_directory_name
from ...io.utils import load_yaml
from .._shared.config import (
    PathLike,
    SchedulerName,
    SlurmConfig,
    _load_slurm_config,
    _resolve_path,
    _strict_bool,
)


@dataclass(frozen=True)
class SnapshotSystemConfig:
    system_id: str
    topology_path: Path
    coordinates_path: Path
    structure_path: Path
    snapshot_files: tuple[Path, ...]
    isolated_atoms: dict[str, tuple[Path, Path]]
    md_input_path: Path
    sp_input_path: Path


def _isolated_atom_inputs(
    value: PathValue,
    *,
    field: str,
) -> dict[str, tuple[Path, Path]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{field} must map element names to input/coordinates roles."
        )
    atoms: dict[str, tuple[Path, Path]] = {}
    for element, record in value.items():
        if not isinstance(record, dict):
            raise ValueError(f"{field}.{element} must be a mapping.")
        unknown = set(record) - {"input", "coordinates"}
        if unknown:
            raise ValueError(
                f"{field}.{element} contains unsupported role(s): "
                + ", ".join(sorted(unknown))
            )
        fn_input = record.get("input")
        fn_coordinates = record.get("coordinates")
        if not isinstance(fn_input, Path) or not isinstance(fn_coordinates, Path):
            raise ValueError(
                f"{field}.{element} requires path roles 'input' and 'coordinates'."
            )
        atoms[str(element)] = (fn_input, fn_coordinates)
    return atoms


def _snapshot_system(inputs: SystemInputs, *, field: str) -> SnapshotSystemConfig:
    allowed = {
        "topology",
        "coordinates",
        "structure",
        "snapshots",
        "md_input",
        "sp_input",
        "isolated_atoms",
    }
    unsupported = set(inputs.inputs) - allowed
    if unsupported:
        raise ValueError(
            f"{field} contains unsupported role(s): "
            + ", ".join(sorted(unsupported))
        )
    snapshots = inputs.inputs.get("snapshots")
    if not isinstance(snapshots, tuple) or not snapshots or not all(
        isinstance(path, Path) for path in snapshots
    ):
        raise ValueError(f"{field}.snapshots must be a non-empty path list.")
    return SnapshotSystemConfig(
        system_id=inputs.system_id,
        topology_path=inputs.require_path("topology"),
        coordinates_path=inputs.require_path("coordinates"),
        structure_path=inputs.require_path("structure"),
        snapshot_files=snapshots,
        isolated_atoms=_isolated_atom_inputs(
            inputs.inputs.get("isolated_atoms"),
            field=f"{field}.isolated_atoms",
        ),
        md_input_path=inputs.require_path("md_input"),
        sp_input_path=inputs.require_path("sp_input"),
    )


def _load_snapshot_systems(
    base_dir: Path,
    systems_raw: Any,
    *,
    source: Path | None,
) -> list[SnapshotSystemConfig]:
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError("'systems' must be a non-empty list.")
    systems: list[SnapshotSystemConfig] = []
    for index, raw in enumerate(systems_raw):
        if isinstance(raw, str):
            raw = {"system_id": raw}
        if not isinstance(raw, Mapping) or "system_id" not in raw:
            raise ValueError(f"systems[{index}] must define system_id.")
        system_id = validate_system_id(
            raw["system_id"], field=f"systems[{index}].system_id"
        )
        if source is not None:
            if set(raw) != {"system_id"}:
                raise ValueError(
                    f"systems[{index}] mixes source selection with direct "
                    "inputs; only system_id is allowed."
                )
            systems.append(_reference_snapshot_system(source, system_id))
            continue
        else:
            if set(raw) != {"system_id", "inputs"}:
                raise ValueError(
                    f"systems[{index}] requires exactly system_id and inputs when "
                    "source is not configured."
                )
            inputs_raw = raw.get("inputs")
            if not isinstance(inputs_raw, Mapping):
                raise ValueError(f"systems[{index}].inputs must be a mapping.")
            inputs = resolve_explicit_inputs(
                inputs_raw,
                base_dir=base_dir,
                system_id=system_id,
                field=f"systems[{index}].inputs",
            )
        systems.append(
            _snapshot_system(inputs, field=f"systems[{index}].inputs")
        )
    validate_unique_system_ids(
        [system.system_id for system in systems], field="systems"
    )
    return systems


def _reference_file(system_dir: Path, relative: str, *, system_id: str) -> Path:
    path = system_dir / relative
    if not path.is_file():
        raise FileNotFoundError(
            f"system {system_id!r}: expected reference input {relative!r} at "
            f"{path}; regenerate prepare-reference or correct 'source'."
        )
    return path


def _reference_snapshot_system(source: Path, system_id: str) -> SnapshotSystemConfig:
    metadata = load_reference_system_metadata(source, system_id)
    system_dir = source / "systems" / system_id
    snapshot_dir = system_dir / "snapshots" / "xyz"
    snapshots = tuple(
        _reference_file(
            system_dir,
            f"snapshots/xyz/snapshot-{index:04d}.xyz",
            system_id=system_id,
        )
        for index in range(metadata.snapshot_count)
    )
    expected = {path.name for path in snapshots}
    actual = {
        path.name for path in snapshot_dir.glob("snapshot-*.xyz") if path.is_file()
    }
    extra = sorted(actual - expected)
    if extra:
        raise ValueError(
            f"system {system_id!r}: reference snapshot directory {snapshot_dir} "
            f"contains files beyond snapshot_count={metadata.snapshot_count}: {extra}; "
            "remove stale files or regenerate prepare-reference."
        )
    isolated_atoms: dict[str, tuple[Path, Path]] = {}
    for element in metadata.elements:
        try:
            directory = get_cp2k_single_atom_directory_name(element)
        except ValueError as exc:
            raise ValueError(
                f"system {system_id!r}: element {element!r} in "
                f"{system_dir / 'system.yaml'} has no supported CP2K isolated-atom "
                "directory; remove it or add CP2K support."
            ) from exc
        isolated_atoms[element] = (
            _reference_file(
                system_dir, f"single-atoms/{directory}/input.inp", system_id=system_id
            ),
            _reference_file(
                system_dir, f"single-atoms/{directory}/pos.xyz", system_id=system_id
            ),
        )
    return SnapshotSystemConfig(
        system_id=system_id,
        topology_path=_reference_file(system_dir, "system.top", system_id=system_id),
        coordinates_path=_reference_file(system_dir, "system.gro", system_id=system_id),
        structure_path=_reference_file(system_dir, "system.xyz", system_id=system_id),
        snapshot_files=snapshots,
        isolated_atoms=isolated_atoms,
        md_input_path=_reference_file(
            system_dir, "snapshots/md.inp", system_id=system_id
        ),
        sp_input_path=_reference_file(
            system_dir, "snapshots/sp.inp", system_id=system_id
        ),
    )


@dataclass(frozen=True, kw_only=True)
class EvaluateSnapshotsConfig:
    fn_config: Path
    output_dir: Path
    log: Path
    results_manifest: Path
    systems: list[SnapshotSystemConfig]
    cp2k_cmd: str
    job_scheduler: SchedulerName
    single_atoms: bool = True
    snapshot_md_steps: int | None = None
    train_fraction: float = 0.8
    seed: int = 2026
    cleanup_snapshots: bool = False
    collection_wait_seconds: float = 60.0
    slurm: Optional[SlurmConfig] = None

    @classmethod
    def load(cls, fn_config: PathLike) -> "EvaluateSnapshotsConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError(
                "Evaluate-snapshots configuration must contain a mapping."
            )
        known_keys = {
            "source",
            "systems",
            "output_dir",
            "log",
            "job_scheduler",
            "cp2k_cmd",
            "single_atoms",
            "snapshot_md_steps",
            "train_fraction",
            "seed",
            "cleanup_snapshots",
            "collection_wait_seconds",
            "slurm",
        }
        unknown = set(config) - known_keys
        if unknown:
            raise ValueError(
                "Evaluate-snapshots configuration contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )
        if "systems" not in config:
            raise ValueError("evaluate-snapshots.systems is required.")
        missing = [key for key in ("job_scheduler", "cp2k_cmd") if key not in config]
        if missing:
            raise ValueError(
                "Missing required evaluate-snapshots configuration key(s): "
                + ", ".join(repr(key) for key in missing)
            )

        output_dir = _resolve_path(
            base_dir,
            config.get("output_dir", "./"),
            must_exist=False,
            kind="output directory",
        )
        source = (
            None
            if config.get("source") is None
            else _resolve_path(
                base_dir, config["source"], kind="prepare-reference stage root"
            )
        )
        scheduler = config["job_scheduler"]
        if scheduler not in {"local", "slurm"}:
            raise ValueError(
                f"job_scheduler must be 'local' or 'slurm', got {scheduler!r}."
            )
        slurm = (
            _load_slurm_config(config.get("slurm"))
            if scheduler == "slurm"
            else None
        )
        train_fraction = float(config.get("train_fraction", 0.8))
        if not 0 < train_fraction < 1:
            raise ValueError("'train_fraction' must be between 0 and 1.")
        snapshot_md_steps = config.get("snapshot_md_steps")
        if snapshot_md_steps is not None:
            snapshot_md_steps = int(snapshot_md_steps)
            if snapshot_md_steps <= 0:
                raise ValueError("'snapshot_md_steps' must be a positive integer.")
        collection_wait_seconds = float(config.get("collection_wait_seconds", 60.0))
        if collection_wait_seconds < 0:
            raise ValueError("'collection_wait_seconds' must be non-negative.")

        return cls(
            fn_config=fn_config,
            output_dir=output_dir,
            log=_resolve_path(
                base_dir,
                config.get("log", output_dir / "evaluate-snapshots.log"),
                must_exist=False,
                kind="evaluate-snapshots log file",
            ),
            results_manifest=output_dir / "snapshot-results.yaml",
            cp2k_cmd=str(config["cp2k_cmd"]),
            job_scheduler=scheduler,
            systems=_load_snapshot_systems(
                base_dir, config["systems"], source=source
            ),
            single_atoms=_strict_bool(
                config.get("single_atoms", True), field="single_atoms"
            ),
            snapshot_md_steps=snapshot_md_steps,
            train_fraction=train_fraction,
            seed=int(config.get("seed", 2026)),
            cleanup_snapshots=_strict_bool(
                config.get("cleanup_snapshots", False),
                field="cleanup_snapshots",
            ),
            collection_wait_seconds=collection_wait_seconds,
            slurm=slurm,
        )
