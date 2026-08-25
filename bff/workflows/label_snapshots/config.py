from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from ...domain.systems import validate_system_id, validate_unique_system_ids
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
    trajectory_path: Path
    md_input_path: Path
    sp_input_path: Path
    n_snapshots: int
    atom_selection: str = "all"
    single_atom_input_paths: dict[str, Path] | None = None


def _load_systems(
    base_dir: Path,
    raw_systems: Any,
    *,
    require_single_atom_inputs: bool,
) -> list[SnapshotSystemConfig]:
    if not isinstance(raw_systems, list) or not raw_systems:
        raise ValueError("'systems' must be a non-empty list.")

    systems: list[SnapshotSystemConfig] = []
    required = {
        "system_id",
        "topology",
        "trajectory",
        "md_input",
        "sp_input",
        "n_snapshots",
    }
    allowed = required | {"atom_selection", "single_atom_inputs"}
    for index, raw in enumerate(raw_systems):
        if not isinstance(raw, Mapping):
            raise ValueError(f"systems[{index}] must be a mapping.")
        unknown = set(raw) - allowed
        missing = required - set(raw)
        if unknown or missing:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unsupported " + ", ".join(sorted(unknown)))
            raise ValueError(
                f"systems[{index}] is invalid: " + "; ".join(details) + "."
            )

        n_snapshots = int(raw["n_snapshots"])
        if n_snapshots <= 0:
            raise ValueError(f"systems[{index}].n_snapshots must be positive.")
        atom_selection = raw.get("atom_selection", "all")
        if not isinstance(atom_selection, str) or not atom_selection.strip():
            raise ValueError(
                f"systems[{index}].atom_selection must be a non-empty string."
            )

        raw_atom_inputs = raw.get("single_atom_inputs")
        if require_single_atom_inputs:
            if not isinstance(raw_atom_inputs, Mapping) or not raw_atom_inputs:
                raise ValueError(
                    f"systems[{index}].single_atom_inputs must be a non-empty "
                    "mapping when single_atoms is true."
                )
        elif raw_atom_inputs is not None:
            raise ValueError(
                f"systems[{index}].single_atom_inputs is not allowed when "
                "single_atoms is false."
            )

        single_atom_input_paths: dict[str, Path] | None = None
        if raw_atom_inputs is not None:
            single_atom_input_paths = {}
            for raw_element, raw_path in raw_atom_inputs.items():
                element = str(raw_element).strip().capitalize()
                if re.fullmatch(r"[A-Z][a-z]?", element) is None:
                    raise ValueError(
                        f"systems[{index}].single_atom_inputs contains invalid "
                        f"element symbol {raw_element!r}."
                    )
                if element in single_atom_input_paths:
                    raise ValueError(
                        f"systems[{index}].single_atom_inputs contains duplicate "
                        f"canonical element {element!r}."
                    )
                single_atom_input_paths[element] = _resolve_path(
                    base_dir,
                    raw_path,
                    kind=(
                        f"systems[{index}] isolated-atom CP2K input for {element}"
                    ),
                )
        systems.append(
            SnapshotSystemConfig(
                system_id=validate_system_id(
                    raw["system_id"], field=f"systems[{index}].system_id"
                ),
                topology_path=_resolve_path(
                    base_dir,
                    raw["topology"],
                    kind=f"systems[{index}] topology file",
                ),
                trajectory_path=_resolve_path(
                    base_dir,
                    raw["trajectory"],
                    kind=f"systems[{index}] trajectory file",
                ),
                md_input_path=_resolve_path(
                    base_dir,
                    raw["md_input"],
                    kind=f"systems[{index}] CP2K MD input",
                ),
                sp_input_path=_resolve_path(
                    base_dir,
                    raw["sp_input"],
                    kind=f"systems[{index}] CP2K single-point input",
                ),
                n_snapshots=n_snapshots,
                atom_selection=atom_selection,
                single_atom_input_paths=single_atom_input_paths,
            )
        )

    validate_unique_system_ids(
        [system.system_id for system in systems], field="systems"
    )
    return systems


@dataclass(frozen=True, kw_only=True)
class LabelSnapshotsConfig:
    fn_config: Path
    output_dir: Path
    log: Path
    results_manifest: Path
    systems: list[SnapshotSystemConfig]
    cp2k_cmd: str
    job_scheduler: SchedulerName
    train_fraction: float = 0.8
    seed: int = 2026
    single_atoms: bool = True
    cleanup_snapshots: bool = False
    collection_wait_seconds: float = 60.0
    slurm: Optional[SlurmConfig] = None

    @classmethod
    def load(cls, fn_config: PathLike) -> "LabelSnapshotsConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError("Label-snapshots configuration must contain a mapping.")

        known = {
            "output_dir",
            "log",
            "systems",
            "cp2k_cmd",
            "job_scheduler",
            "train_fraction",
            "seed",
            "single_atoms",
            "cleanup_snapshots",
            "collection_wait_seconds",
            "slurm",
        }
        unknown = set(config) - known
        if unknown:
            raise ValueError(
                "Label-snapshots configuration contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )
        missing = [
            key
            for key in ("systems", "cp2k_cmd", "job_scheduler")
            if key not in config
        ]
        if missing:
            raise ValueError(
                "Missing required label-snapshots configuration key(s): "
                + ", ".join(repr(key) for key in missing)
            )

        scheduler = config["job_scheduler"]
        if scheduler not in {"local", "slurm"}:
            raise ValueError(
                f"job_scheduler must be 'local' or 'slurm', got {scheduler!r}."
            )
        train_fraction = float(config.get("train_fraction", 0.8))
        if not 0 < train_fraction < 1:
            raise ValueError("'train_fraction' must be between 0 and 1.")
        collection_wait_seconds = float(config.get("collection_wait_seconds", 60.0))
        if collection_wait_seconds < 0:
            raise ValueError("'collection_wait_seconds' must be non-negative.")

        output_dir = _resolve_path(
            base_dir,
            config.get("output_dir", "./"),
            must_exist=False,
            kind="output directory",
        )
        single_atoms = _strict_bool(
            config.get("single_atoms", True), field="single_atoms"
        )
        return cls(
            fn_config=fn_config,
            output_dir=output_dir,
            log=_resolve_path(
                base_dir,
                config.get("log", output_dir / "label-snapshots.log"),
                must_exist=False,
                kind="label-snapshots log file",
            ),
            results_manifest=output_dir / "label-results.yaml",
            systems=_load_systems(
                base_dir,
                config["systems"],
                require_single_atom_inputs=single_atoms,
            ),
            cp2k_cmd=str(config["cp2k_cmd"]),
            job_scheduler=scheduler,
            train_fraction=train_fraction,
            seed=int(config.get("seed", 2026)),
            single_atoms=single_atoms,
            cleanup_snapshots=_strict_bool(
                config.get("cleanup_snapshots", False),
                field="cleanup_snapshots",
            ),
            collection_wait_seconds=collection_wait_seconds,
            slurm=(
                _load_slurm_config(config.get("slurm"))
                if scheduler == "slurm"
                else None
            ),
        )
