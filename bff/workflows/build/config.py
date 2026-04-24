from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...domain.bias import BiasSpec
from ...io.utils import load_yaml
from .._shared.config import PathLike, _resolve_path


@dataclass(frozen=True)
class BuildSystemConfig:
    fn_topol: Path
    templates: dict[str, Path]
    charge: int
    mult: int
    box: list[float] | None
    bias: BiasSpec
    nsteps_npt: int
    nsteps_prod: int
    fn_mdp_em: Path
    fn_mdp_npt: Path
    fn_mdp_prod: Path


@dataclass(frozen=True)
class BuildConfig:
    fn_config: Path
    project_dir: Path
    gmx_cmd: str
    fn_log: Optional[Path]
    systems: list[BuildSystemConfig]
    n_single_point_snapshots: int = 1000

    @classmethod
    def load(cls, fn_config: PathLike) -> 'BuildConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        required = ['project', 'gromacs', 'systems']
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                'Missing required build option(s): '
                + ', '.join(repr(key) for key in missing)
            )

        project = config['project']
        if isinstance(project, str):
            project_dir = _resolve_path(
                base_dir,
                project,
                must_exist=False,
                kind='project directory',
            )
            fn_log_raw = config.get('fn_log')
        elif isinstance(project, dict):
            if 'directory' not in project:
                raise ValueError('project.directory is required.')
            project_dir = _resolve_path(
                base_dir,
                project['directory'],
                must_exist=False,
                kind='project directory',
            )
            fn_log_raw = project.get('log')
        else:
            raise ValueError("'project' must be a string or mapping.")

        gromacs = config['gromacs']
        if not isinstance(gromacs, dict):
            raise ValueError("'gromacs' must be a mapping.")
        if 'command' not in gromacs:
            raise ValueError('gromacs.command is required.')
        defaults = config.get('defaults', {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("'defaults' must be a mapping.")
        default_steps = defaults.get('nsteps', {})
        if default_steps is None:
            default_steps = {}
        if not isinstance(default_steps, dict):
            raise ValueError("'defaults.nsteps' must be a mapping.")
        nsteps_npt_default = int(default_steps.get('npt', 0))
        nsteps_prod_default = int(default_steps.get('prod', 100000))
        if nsteps_npt_default < 0 or nsteps_prod_default < 0:
            raise ValueError('defaults.nsteps values must be non-negative.')

        systems_raw = config['systems']
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'systems' must be a non-empty list.")

        systems: list[BuildSystemConfig] = []
        for i, system in enumerate(systems_raw):
            if not isinstance(system, dict):
                raise ValueError(f'System {i} must be a mapping.')
            for key in ('topology', 'templates', 'charge', 'multiplicity'):
                if key not in system:
                    raise ValueError(f'System {i} is missing required key {key!r}.')
            templates_raw = system['templates']
            if not isinstance(templates_raw, dict) or not templates_raw:
                raise ValueError(f'System {i} templates must be a non-empty mapping.')
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

            steps = system.get('nsteps', {})
            if steps is None:
                steps = {}
            if not isinstance(steps, dict):
                raise ValueError(f'System {i} nsteps must be a mapping.')
            nsteps_npt = int(steps.get('npt', nsteps_npt_default))
            nsteps_prod = int(steps.get('prod', nsteps_prod_default))
            if nsteps_npt < 0 or nsteps_prod < 0:
                raise ValueError(f'System {i} nsteps values must be non-negative.')
            if (
                box_values is not None
                and len(box_values) == 6
                and box_values[3:] == [90.0, 90.0, 90.0]
                and len(system.get('box', [])) == 3
            ):
                nsteps_npt = 0

            mdp = system.get('mdp')
            if not isinstance(mdp, dict):
                raise ValueError(f'System {i} mdp must be a mapping.')
            missing_mdp = [key for key in ('em', 'npt', 'prod') if key not in mdp]
            if missing_mdp:
                raise ValueError(
                    f'System {i} mdp is missing required key(s): '
                    + ', '.join(repr(key) for key in missing_mdp)
                )

            systems.append(
                BuildSystemConfig(
                    fn_topol=_resolve_path(
                        base_dir,
                        system['topology'],
                        kind=f'system {i} topology file',
                    ),
                    templates={
                        name: _resolve_path(
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
                    fn_mdp_em=_resolve_path(
                        base_dir,
                        mdp['em'],
                        kind=f'system {i} em mdp file',
                    ),
                    fn_mdp_npt=_resolve_path(
                        base_dir,
                        mdp['npt'],
                        kind=f'system {i} npt mdp file',
                    ),
                    fn_mdp_prod=_resolve_path(
                        base_dir,
                        mdp['prod'],
                        kind=f'system {i} production mdp file',
                    ),
                )
            )

        reference = config.get('reference', {})
        if reference is None:
            reference = {}
        if not isinstance(reference, dict):
            raise ValueError("'reference' must be a mapping.")

        resolved_log = None if fn_log_raw is None else _resolve_path(
            base_dir,
            fn_log_raw,
            must_exist=False,
            kind='log file',
        )
        n_single_point_snapshots = int(reference.get('n_single_point_snapshots', 1000))
        if n_single_point_snapshots <= 0:
            raise ValueError(
                "'reference.n_single_point_snapshots' must be a positive integer."
            )
        return cls(
            fn_config=fn_config,
            project_dir=project_dir,
            gmx_cmd=str(gromacs['command']),
            fn_log=resolved_log,
            systems=systems,
            n_single_point_snapshots=n_single_point_snapshots,
        )
