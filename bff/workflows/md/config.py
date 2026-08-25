from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...domain.systems import validate_system_id
from ...io.utils import load_yaml
from .._shared.config import (
    PathLike,
    SchedulerName,
    SimulationSystemConfig,
    _load_simulation_systems,
    _normalize_store,
    _resolve_optional_path,
    _resolve_path,
    _strict_bool,
)


@dataclass(frozen=True)
class MDJobConfig:
    fn_config: Path
    sample_id: str
    params: list[float]
    campaign_dir: Path
    fn_specs: Optional[Path]
    gmx_cmd: str
    job_scheduler: SchedulerName
    store: tuple[str, ...]
    cleanup: bool
    run: bool
    systems: list[SimulationSystemConfig]

    @classmethod
    def load(cls, fn_config: PathLike) -> 'MDJobConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, dict):
            raise ValueError("MD job configuration must contain a mapping.")
        allowed = {
            'sample_id',
            'params',
            'campaign_dir',
            'fn_specs',
            'gmx_cmd',
            'job_scheduler',
            'store',
            'cleanup',
            'run',
            'systems',
        }
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                'MD job configuration contains unsupported key(s): '
                + ', '.join(sorted(unknown))
            )

        required = [
            'sample_id',
            'params',
            'campaign_dir',
            'gmx_cmd',
            'job_scheduler',
            'systems',
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                'Missing required MD job option(s): '
                + ', '.join(repr(key) for key in missing)
            )
        if config['job_scheduler'] not in {'local', 'slurm'}:
            raise ValueError(
                "job_scheduler must be 'local' or 'slurm', got "
                f"{config['job_scheduler']!r}."
            )
        if not isinstance(config['params'], list):
            raise ValueError("params must be a list of numeric values.")

        return cls(
            fn_config=fn_config,
            sample_id=validate_system_id(config['sample_id'], field='sample_id'),
            params=[float(value) for value in config['params']],
            campaign_dir=_resolve_path(
                base_dir,
                config['campaign_dir'],
                kind='campaign directory',
            ),
            fn_specs=_resolve_optional_path(
                base_dir,
                config.get('fn_specs'),
                kind='specs file',
            ),
            gmx_cmd=str(config['gmx_cmd']),
            job_scheduler=config['job_scheduler'],
            store=tuple(_normalize_store(config.get('store'))),
            cleanup=_strict_bool(config.get('cleanup', False), field='cleanup'),
            run=_strict_bool(config.get('run', True), field='run'),
            systems=_load_simulation_systems(
                base_dir,
                config['systems'],
                key='systems',
            ),
        )
