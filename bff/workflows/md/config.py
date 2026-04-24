from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...io.utils import load_yaml
from .._shared.config import (
    PathLike,
    SchedulerName,
    SimulationSystemConfig,
    _load_simulation_systems,
    _normalize_store,
    _resolve_optional_path,
    _resolve_path,
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
    run: bool
    systems: list[SimulationSystemConfig]

    @classmethod
    def load(cls, fn_config: PathLike) -> 'MDJobConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

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

        return cls(
            fn_config=fn_config,
            sample_id=str(config['sample_id']),
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
            run=bool(config.get('run', True)),
            systems=_load_simulation_systems(
                base_dir,
                config['systems'],
                key='systems',
            ),
        )
