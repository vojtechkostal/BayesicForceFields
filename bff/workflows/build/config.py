from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...domain.bias import BiasSpec
from ...domain.systems import validate_system_id, validate_unique_system_ids
from ...io.utils import load_yaml
from ..config import PathLike, check_keys, resolve_path


@dataclass(frozen=True)
class BuildSystemConfig:
    system_id: str
    system_name: str | None
    topology_path: Path
    templates: dict[str, Path]
    charge: int
    mult: int
    box: list[float] | None
    bias: BiasSpec
    nsteps_npt: int
    nsteps_prod: int
    mdp_em_path: Path
    mdp_npt_path: Path
    mdp_production_path: Path


@dataclass(frozen=True)
class BuildConfig:
    fn_config: Path
    project_dir: Path
    gmx_cmd: str
    fn_log: Optional[Path]
    systems: list[BuildSystemConfig]

    @classmethod
    def load(cls, fn_config: PathLike) -> 'BuildConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = check_keys(
            load_yaml(fn_config),
            where='Build configuration',
            allowed=('project', 'gromacs', 'systems', 'fn_log'),
            required=('project', 'gromacs', 'systems'),
        )

        project = config['project']
        if isinstance(project, str):
            project_dir = resolve_path(
                base_dir,
                project,
                must_exist=False,
                kind='project directory',
            )
            fn_log_raw = config.get('fn_log')
        elif isinstance(project, dict):
            check_keys(
                project,
                where='project',
                allowed=('directory', 'log'),
                required=('directory',),
            )
            project_dir = resolve_path(
                base_dir,
                project['directory'],
                must_exist=False,
                kind='project directory',
            )
            fn_log_raw = project.get('log')
        else:
            raise ValueError("'project' must be a string or mapping.")

        gromacs = check_keys(
            config['gromacs'],
            where='gromacs',
            allowed=('command',),
            required=('command',),
        )
        systems_raw = config['systems']
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'systems' must be a non-empty list.")

        systems: list[BuildSystemConfig] = []
        for i, system in enumerate(systems_raw):
            check_keys(
                system,
                where=f'systems[{i}]',
                allowed=(
                    'system_id',
                    'system_name',
                    'topology',
                    'templates',
                    'charge',
                    'multiplicity',
                    'box',
                    'bias',
                    'nsteps',
                    'mdp',
                ),
                required=(
                    'system_id',
                    'topology',
                    'charge',
                    'multiplicity',
                    'nsteps',
                    'mdp',
                ),
            )
            templates_raw = system.get('templates', {})
            if not isinstance(templates_raw, dict):
                raise ValueError(f'System {i} templates must be a mapping.')
            if not all(
                isinstance(name, str) and isinstance(path, (str, Path))
                for name, path in templates_raw.items()
            ):
                raise ValueError(
                    f'System {i} templates must map residue names to file paths.'
                )

            box = system.get('box')
            if box is None:
                box_values = None
            else:
                if not isinstance(box, list) or len(box) not in {3, 6}:
                    raise ValueError(
                        f'Invalid box dimensions at index {i}: {box}. '
                        'Expected 3 or 6 numeric values.'
                    )
                if not all(isinstance(value, (int, float)) for value in box):
                    raise ValueError(f'Invalid box dimensions at index {i}: {box}.')
                if len(box) == 3:
                    box = [*box, 90.0, 90.0, 90.0]
                box_values = [float(value) for value in box]

            steps = check_keys(
                system['nsteps'],
                where=f'systems[{i}].nsteps',
                allowed=('npt', 'prod'),
                required=('npt', 'prod'),
            )
            nsteps_npt = int(steps['npt'])
            nsteps_prod = int(steps['prod'])
            if nsteps_npt < 0 or nsteps_prod < 0:
                raise ValueError(f'System {i} nsteps values must be non-negative.')

            mdp = check_keys(
                system['mdp'],
                where=f'systems[{i}].mdp',
                allowed=('em', 'npt', 'prod'),
                required=('em', 'npt', 'prod'),
            )

            systems.append(
                BuildSystemConfig(
                    system_id=validate_system_id(
                        system['system_id'], field=f'systems[{i}].system_id'
                    ),
                    system_name=system.get('system_name'),
                    topology_path=resolve_path(
                        base_dir,
                        system['topology'],
                        kind=f'system {i} topology file',
                    ),
                    templates={
                        name: resolve_path(
                            base_dir,
                            path,
                            kind=f'system {i} template file for {name!r}',
                        )
                        for name, path in templates_raw.items()
                    },
                    charge=int(system['charge']),
                    mult=int(system['multiplicity']),
                    box=box_values,
                    bias=BiasSpec.from_any(system.get('bias'), base_dir=base_dir),
                    nsteps_npt=nsteps_npt,
                    nsteps_prod=nsteps_prod,
                    mdp_em_path=resolve_path(
                        base_dir,
                        mdp['em'],
                        kind=f'system {i} em mdp file',
                    ),
                    mdp_npt_path=resolve_path(
                        base_dir,
                        mdp['npt'],
                        kind=f'system {i} npt mdp file',
                    ),
                    mdp_production_path=resolve_path(
                        base_dir,
                        mdp['prod'],
                        kind=f'system {i} production mdp file',
                    ),
                )
            )

            if system.get('system_name') is not None and not isinstance(
                system['system_name'], str
            ):
                raise ValueError(f'Systems[{i}].system_name must be a string.')

        validate_unique_system_ids(
            [system.system_id for system in systems], field='systems'
        )

        resolved_log = (
            project_dir / 'build.log'
            if fn_log_raw is None
            else resolve_path(
                base_dir,
                fn_log_raw,
                must_exist=False,
                kind='log file',
            )
        )
        return cls(
            fn_config=fn_config,
            project_dir=project_dir,
            gmx_cmd=str(gromacs['command']),
            fn_log=resolved_log,
            systems=systems,
        )
