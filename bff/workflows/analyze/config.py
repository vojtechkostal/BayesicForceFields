from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from ...io.utils import load_yaml
from ...qoi.routines import (
    AnalysisRoutineConfig,
    AnalysisRuntimeConfig,
    normalize_analysis_runtime_config,
    normalize_routine_list,
)
from .._shared.config import PathLike, _resolve_path


@dataclass(frozen=True)
class AnalyzeSystemConfig:
    fn_coord: Path
    fn_topol: Path
    fn_trj: Path
    routines: tuple[AnalysisRoutineConfig, ...]


@dataclass(frozen=True)
class AnalyzeSampleConfig:
    dir: Path
    start: int = 1
    stop: Optional[int] = None
    step: int = 1
    workers: int = -1
    progress_stride: int = 10


@dataclass(frozen=True)
class AnalyzeReferenceConfig:
    systems: list[AnalyzeSystemConfig]
    start: int = 0
    stop: int = -1
    step: int = 1


@dataclass(frozen=True)
class AnalyzeOutputConfig:
    path: Path = Path('./analysis')
    log: Path = Path('./out.log')
    write_raw: bool = False


@dataclass(frozen=True)
class AnalyzeConfig:
    fn_config: Path
    sample: AnalyzeSampleConfig
    reference: AnalyzeReferenceConfig
    run: AnalysisRuntimeConfig = AnalysisRuntimeConfig()
    output: AnalyzeOutputConfig = AnalyzeOutputConfig()

    @classmethod
    def load(cls, fn_config: PathLike) -> 'AnalyzeConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ('sample', 'reference'):
            if key not in config:
                raise ValueError(f'Missing required configuration section: {key!r}.')

        sample = config['sample']
        if not isinstance(sample, Mapping):
            raise ValueError("'sample' must be a mapping.")
        sample_dir = sample.get('dir')
        if sample_dir is None:
            raise ValueError("Missing 'dir' in sample configuration.")

        reference = config['reference']
        if not isinstance(reference, Mapping):
            raise ValueError("'reference' must be a mapping.")
        systems_raw = reference.get('systems')
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'reference.systems' must be a non-empty list.")

        systems: list[AnalyzeSystemConfig] = []
        for i, system in enumerate(systems_raw):
            if not isinstance(system, Mapping):
                raise ValueError(f'reference.systems[{i}] must be a mapping.')
            for key in ('coordinates', 'topology', 'trajectory'):
                if key not in system:
                    raise ValueError(
                        f'reference.systems[{i}] is missing required key {key!r}.'
                    )
            routines = system.get('routines')
            if not isinstance(routines, list) or not routines:
                raise ValueError(
                    f'reference.systems[{i}].routines must be a non-empty list.'
                )

            systems.append(
                AnalyzeSystemConfig(
                    fn_coord=_resolve_path(
                        base_dir,
                        system['coordinates'],
                        kind=f'reference.systems[{i}] coordinates file',
                    ),
                    fn_topol=_resolve_path(
                        base_dir,
                        system['topology'],
                        kind=f'reference.systems[{i}] topology file',
                    ),
                    fn_trj=_resolve_path(
                        base_dir,
                        system['trajectory'],
                        kind=f'reference.systems[{i}] trajectory file',
                    ),
                    routines=normalize_routine_list(routines, base_dir=base_dir),
                )
            )

        run = normalize_analysis_runtime_config(config.get('run'))
        output = config.get('output', {})
        if not isinstance(output, Mapping):
            raise ValueError("'output' must be a mapping.")

        return cls(
            fn_config=fn_config,
            sample=AnalyzeSampleConfig(
                dir=_resolve_path(
                    base_dir,
                    sample_dir,
                    kind='sample campaign directory',
                ),
                start=int(sample.get('start', 1)),
                stop=sample.get('stop'),
                step=int(sample.get('step', 1)),
                workers=int(sample.get('workers', -1)),
                progress_stride=int(sample.get('progress_stride', 10)),
            ),
            reference=AnalyzeReferenceConfig(
                systems=systems,
                start=int(reference.get('start', 0)),
                stop=reference.get('stop', -1),
                step=int(reference.get('step', 1)),
            ),
            run=run,
            output=AnalyzeOutputConfig(
                path=_resolve_path(
                    base_dir,
                    output.get('path', './analysis'),
                    must_exist=False,
                    kind='analysis output path',
                ),
                log=_resolve_path(
                    base_dir,
                    output.get('log', './out.log'),
                    must_exist=False,
                    kind='log file',
                ),
                write_raw=bool(output.get('write_raw', False)),
            ),
        )
