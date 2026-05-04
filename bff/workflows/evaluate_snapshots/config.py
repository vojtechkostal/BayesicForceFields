from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from ...io.utils import load_yaml
from .._shared.config import (
    PathLike,
    SchedulerName,
    SlurmConfig,
    _load_slurm_config,
    _resolve_optional_path,
    _resolve_path,
)

EvaluateMode = Literal['run', 'import']


@dataclass(frozen=True)
class SnapshotSystemConfig:
    system_id: str
    assets_dir: Path
    fn_system_top: Path
    fn_system_gro: Path
    fn_system_xyz: Path
    snapshots_dir: Path
    snapshot_xyz_dir: Path
    single_atoms_dir: Path
    fn_snapshot_md: Path
    fn_snapshot_sp: Path


@dataclass(frozen=True)
class ImportedSnapshotSystemConfig:
    system_id: str
    fn_topol: Path
    fn_gro: Path
    fn_trj: Path


def _load_snapshot_asset_system(
    assets_dir: Path,
    *,
    system_id: str,
    fn_snapshot_md: Path | None = None,
    fn_snapshot_sp: Path | None = None,
) -> SnapshotSystemConfig:
    fn_system_top = assets_dir / 'system.top'
    fn_system_gro = assets_dir / 'system.gro'
    fn_system_xyz = assets_dir / 'system.xyz'
    md_dir = assets_dir / 'md'
    snapshots_dir = assets_dir / 'snapshots'
    single_atoms_dir = assets_dir / 'single-atoms'
    snapshot_xyz_dir = snapshots_dir / 'xyz'
    fn_snapshot_md = fn_snapshot_md or snapshots_dir / 'md.inp'
    fn_snapshot_sp = fn_snapshot_sp or snapshots_dir / 'sp.inp'

    required_paths = [
        (fn_system_top, 'system topology'),
        (fn_system_gro, 'system coordinates'),
        (fn_system_xyz, 'system XYZ'),
        (md_dir, 'staged MD directory'),
        (snapshots_dir, 'snapshot directory'),
        (single_atoms_dir, 'single-atom directory'),
        (snapshot_xyz_dir, 'snapshot XYZ directory'),
        (fn_snapshot_md, 'snapshot MD input'),
        (fn_snapshot_sp, 'snapshot single-point input'),
    ]
    for path, kind in required_paths:
        if not path.exists():
            raise FileNotFoundError(f'Required {kind} not found: {path}')

    return SnapshotSystemConfig(
        system_id=system_id,
        assets_dir=assets_dir,
        fn_system_top=fn_system_top,
        fn_system_gro=fn_system_gro,
        fn_system_xyz=fn_system_xyz,
        snapshots_dir=snapshots_dir,
        snapshot_xyz_dir=snapshot_xyz_dir,
        single_atoms_dir=single_atoms_dir,
        fn_snapshot_md=fn_snapshot_md,
        fn_snapshot_sp=fn_snapshot_sp,
    )


def _load_snapshot_asset_systems(
    base_dir: Path,
    systems_raw: Any,
) -> list[SnapshotSystemConfig]:
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError("'systems' must be a non-empty list.")

    systems: list[SnapshotSystemConfig] = []
    for index, system in enumerate(systems_raw):
        fn_snapshot_md = None
        fn_snapshot_sp = None
        if isinstance(system, (str, Path)):
            assets_raw = system
        elif isinstance(system, Mapping):
            if 'assets' not in system:
                raise ValueError(f"systems[{index}] must define 'assets'.")
            assets_raw = system['assets']
            fn_snapshot_md = _resolve_optional_path(
                base_dir,
                system.get('md'),
                kind=f'systems[{index}] snapshot MD input',
            )
            fn_snapshot_sp = _resolve_optional_path(
                base_dir,
                system.get('sp'),
                kind=f'systems[{index}] snapshot single-point input',
            )
        else:
            raise ValueError(
                f"systems[{index}] must be a path or a mapping with 'assets'."
            )

        assets_dir = _resolve_path(
            base_dir,
            assets_raw,
            kind=f'systems[{index}] snapshot assets directory',
        )
        systems.append(
            _load_snapshot_asset_system(
                assets_dir,
                system_id=f'{index:03d}',
                fn_snapshot_md=fn_snapshot_md,
                fn_snapshot_sp=fn_snapshot_sp,
            )
        )
    return systems


def _require_suffix(path: Path, suffix: str, *, kind: str) -> None:
    if path.suffix.lower() != suffix:
        raise ValueError(f'{kind} must be a {suffix} file, got {path}.')


def _load_imported_snapshot_systems(
    base_dir: Path,
    systems_raw: Any,
) -> list[ImportedSnapshotSystemConfig]:
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError("'systems' must be a non-empty list.")

    systems: list[ImportedSnapshotSystemConfig] = []
    for index, system in enumerate(systems_raw):
        if not isinstance(system, Mapping):
            raise ValueError(f'systems[{index}] must be a mapping.')
        for key in ('topology', 'coordinates', 'trajectory'):
            if key not in system:
                raise ValueError(
                    f"systems[{index}] is missing required key {key!r}."
                )

        fn_topol = _resolve_path(
            base_dir,
            system['topology'],
            kind=f'systems[{index}] topology file',
        )
        fn_gro = _resolve_path(
            base_dir,
            system['coordinates'],
            kind=f'systems[{index}] coordinates file',
        )
        fn_trj = _resolve_path(
            base_dir,
            system['trajectory'],
            kind=f'systems[{index}] trajectory file',
        )
        _require_suffix(fn_topol, '.top', kind='Imported snapshot topology')
        _require_suffix(fn_gro, '.gro', kind='Imported snapshot coordinates')

        systems.append(
            ImportedSnapshotSystemConfig(
                system_id=f'{index:03d}',
                fn_topol=fn_topol,
                fn_gro=fn_gro,
                fn_trj=fn_trj,
            )
        )
    return systems


@dataclass(frozen=True, kw_only=True)
class EvaluateSnapshotsConfig:
    fn_config: Path
    mode: EvaluateMode
    output_dir: Path
    systems: list[SnapshotSystemConfig | ImportedSnapshotSystemConfig]
    cp2k_cmd: str | None = None
    job_scheduler: SchedulerName | None = None
    single_atoms: bool = True
    snapshot_md_steps: int | None = None
    train_fraction: float = 0.8
    seed: int = 2026
    cleanup_snapshots: bool = False
    collection_wait_seconds: float = 60.0
    slurm: Optional[SlurmConfig] = None

    @classmethod
    def load(cls, fn_config: PathLike) -> 'EvaluateSnapshotsConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError(
                "Evaluate-snapshots configuration must contain a mapping."
            )

        mode = str(config.get('mode', 'run'))
        if mode not in {'run', 'import'}:
            raise ValueError("'mode' must be either 'run' or 'import'.")

        output_dir = _resolve_path(
            base_dir,
            config.get('output_dir', './'),
            must_exist=False,
            kind='output directory',
        )

        if 'systems' not in config:
            raise ValueError(
                "Missing required evaluate-snapshots configuration key 'systems'."
            )

        if mode == 'import':
            return cls(
                fn_config=fn_config,
                mode='import',
                output_dir=output_dir,
                systems=_load_imported_snapshot_systems(base_dir, config['systems']),
            )

        required = ['job_scheduler', 'cp2k_cmd']
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                'Missing required evaluate-snapshots configuration key(s): '
                + ', '.join(repr(key) for key in missing)
            )

        scheduler = config['job_scheduler']
        if scheduler not in {'local', 'slurm'}:
            raise ValueError(
                f'Unsupported scheduler {scheduler!r}. Supported values are '
                "'local' and 'slurm'."
            )

        slurm = None
        if scheduler == 'slurm':
            slurm = _load_slurm_config(config.get('slurm'))

        train_fraction = float(config.get('train_fraction', 0.8))
        if not 0 < train_fraction < 1:
            raise ValueError("'train_fraction' must be between 0 and 1.")

        snapshot_md_steps = config.get('snapshot_md_steps')
        if snapshot_md_steps is not None:
            snapshot_md_steps = int(snapshot_md_steps)
            if snapshot_md_steps <= 0:
                raise ValueError("'snapshot_md_steps' must be a positive integer.")

        seed = int(config.get('seed', 2026))
        collection_wait_seconds = float(config.get('collection_wait_seconds', 60.0))
        if collection_wait_seconds < 0:
            raise ValueError("'collection_wait_seconds' must be non-negative.")

        return cls(
            fn_config=fn_config,
            mode='run',
            output_dir=output_dir,
            cp2k_cmd=str(config['cp2k_cmd']),
            job_scheduler=scheduler,
            systems=_load_snapshot_asset_systems(base_dir, config['systems']),
            single_atoms=bool(config.get('single_atoms', True)),
            snapshot_md_steps=snapshot_md_steps,
            train_fraction=train_fraction,
            seed=seed,
            cleanup_snapshots=bool(config.get('cleanup_snapshots', False)),
            collection_wait_seconds=collection_wait_seconds,
            slurm=slurm,
        )
