from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from gmxtopology import Topology

from ...domain.specs import ChargeConstraint, RandomParamsGenerator, Specs
from ...io.logs import Logger
from ...io.utils import compress_results, load_yaml, save_yaml
from ...topology import TopologyModifier
from ..md.main import modify_topology
from ..sample.config import SampleConfig
from .config import SimulationCampaignConfig, SimulationSystemConfig
from .scheduler import (
    SCHEDULER_CLASSES,
    bff_cli_command,
    build_slurm_cli_job,
    control_jobs,
    get_job_state_counts,
    wait_for_scheduler_slot,
)

PathLike = str | Path


def _relative_path(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(base_dir))


def _system_record(
    system: SimulationSystemConfig,
    campaign_dir: Path,
) -> dict[str, Any]:
    return {
        'system_id': system.system_id,
        'topology': _relative_path(system.fn_topol, campaign_dir),
        'coordinates': _relative_path(system.fn_coordinates, campaign_dir),
        'mdp': {
            'em': _relative_path(system.fn_mdp_em, campaign_dir),
            'prod': _relative_path(system.fn_mdp_prod, campaign_dir),
        },
        'index': _relative_path(system.fn_ndx, campaign_dir),
        'bias': _relative_path(system.bias.input_file, campaign_dir),
        'n_steps': int(system.n_steps),
    }


def build_specs(config: SampleConfig) -> Path:
    config.campaign_dir.resolve().mkdir(parents=True, exist_ok=True)
    modifiers = [TopologyModifier(system.fn_topol) for system in config.systems]
    charge_params = [name for name in config.bounds if name.startswith('charge ')]
    parameter_indices: list[dict[str, set[int]]] = []
    resolved_tokens = {name: set() for name in charge_params}
    for modifier in modifiers:
        by_parameter: dict[str, set[int]] = {}
        atom_owner: dict[int, str] = {}
        for parameter in charge_params:
            matches = modifier.charge_parameter_matches(parameter)
            resolved_tokens[parameter].update(matches)
            indices = set().union(*matches.values())
            by_parameter[parameter] = indices
            for index in indices:
                if index in atom_owner:
                    raise ValueError(
                        f"Charge parameters {atom_owner[index]!r} and {parameter!r} "
                        f"both modify atom {modifier.atoms[index].name!r} in "
                        f"{modifier.source}."
                    )
                atom_owner[index] = parameter
        parameter_indices.append(by_parameter)

    for parameter, tokens in resolved_tokens.items():
        missing = set(parameter.split()[1:]) - tokens
        if missing:
            names = ', '.join(sorted(repr(name) for name in missing))
            raise ValueError(
                f"Charge parameter {parameter!r} references atom name or type "
                f"token(s) not found in any configured system: {names}."
            )

    compiled: list[dict[str, Any]] = []
    selected_sets: list[list[set[int]]] = []
    for constraint in config.charge_constraints:
        equations: list[tuple[dict[str, float], float]] = []
        selections: list[set[int]] = []
        for modifier, by_parameter in zip(modifiers, parameter_indices):
            try:
                groups = modifier.selected_groups(
                    constraint.selection,
                    constraint.scope,
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid MDAnalysis selection {constraint.selection!r}."
                ) from exc
            selections.append(set().union(*groups))
            for group in groups:
                coefficients = {
                    parameter: float(len(group & indices))
                    for parameter, indices in by_parameter.items()
                    if group & indices
                }
                controlled = set().union(*(
                    group & indices for indices in by_parameter.values()
                ))
                fixed_charge = float(sum(
                    modifier.atoms[index].charge for index in group - controlled
                ))
                equations.append((coefficients, fixed_charge))

        if not equations:
            raise ValueError(
                f"Charge-constraint selection {constraint.selection!r} does not "
                "match any atoms in the configured systems."
            )

        coefficients, fixed_charge = equations[0]
        implicit_coefficient = coefficients.get(constraint.implicit, 0.0)
        if np.isclose(implicit_coefficient, 0.0):
            raise ValueError(
                f"Implicit parameter {constraint.implicit!r} is not selected by "
                f"its owning constraint {constraint.selection!r}."
            )
        reference_row = np.asarray([
            coefficients.get(name, 0.0) / implicit_coefficient
            for name in config.bounds
        ])
        reference_target = (constraint.target - fixed_charge) / implicit_coefficient
        for candidate, candidate_fixed in equations[1:]:
            candidate_implicit = candidate.get(constraint.implicit, 0.0)
            if np.isclose(candidate_implicit, 0.0):
                raise ValueError(
                    f"Implicit parameter {constraint.implicit!r} is not selected "
                    f"consistently by {constraint.selection!r}."
                )
            row = np.asarray([
                candidate.get(name, 0.0) / candidate_implicit
                for name in config.bounds
            ])
            target = (constraint.target - candidate_fixed) / candidate_implicit
            if not np.allclose(row, reference_row) or not np.isclose(
                target, reference_target
            ):
                raise ValueError(
                    f"Charge constraint {constraint.selection!r} does not define "
                    "one consistent equation across the configured systems and "
                    f"{constraint.scope} groups."
                )

        compiled.append({
            'selection': constraint.selection,
            'target': constraint.target,
            'scope': constraint.scope,
            'implicit': constraint.implicit,
            'coefficients': coefficients,
            'fixed_charge': fixed_charge,
        })
        selected_sets.append(selections)

    for first in range(len(compiled)):
        for second in range(first + 1, len(compiled)):
            relations: set[str] = set()
            for selected_first, selected_second in zip(
                selected_sets[first], selected_sets[second]
            ):
                if not selected_first or not selected_second:
                    continue
                overlap = selected_first & selected_second
                if not overlap:
                    relations.add('disjoint')
                elif selected_first == selected_second:
                    raise ValueError(
                        "Charge constraints must not select exactly the same atoms: "
                        f"{compiled[first]['selection']!r} and "
                        f"{compiled[second]['selection']!r}."
                    )
                elif selected_first < selected_second:
                    relations.add('first-child')
                elif selected_second < selected_first:
                    relations.add('second-child')
                else:
                    raise ValueError(
                        "Charge constraints may be disjoint or nested, but must not "
                        f"partially overlap: {compiled[first]['selection']!r} and "
                        f"{compiled[second]['selection']!r}."
                    )
            if len(relations) > 1:
                raise ValueError(
                    "Charge-constraint hierarchy changes between configured systems: "
                    f"{compiled[first]['selection']!r} and "
                    f"{compiled[second]['selection']!r}."
                )
            if relations == {'first-child'}:
                child, parent = first, second
            elif relations == {'second-child'}:
                child, parent = second, first
            else:
                continue
            parent_implicit = compiled[parent]['implicit']
            if compiled[child]['coefficients'].get(parent_implicit, 0.0):
                raise ValueError(
                    f"Implicit parameter {parent_implicit!r} owned by parent "
                    f"constraint {compiled[parent]['selection']!r} must not appear "
                    f"in descendant constraint {compiled[child]['selection']!r}."
                )

    specs = Specs({'bounds': config.bounds, 'charge_constraints': compiled})
    fn_specs = config.campaign_dir / 'specs.yaml'
    specs.write(fn_specs)
    return fn_specs


def stage_systems(
    systems: list[SimulationSystemConfig],
    campaign_dir: Path,
) -> list[SimulationSystemConfig]:
    staged_systems: list[SimulationSystemConfig] = []
    for index, system in enumerate(systems):
        fn_topol = campaign_dir / f'window-{index:03d}.top'
        Topology(system.fn_topol).write(fn_topol, overwrite=True)

        fn_coordinates = campaign_dir / f'window-{index:03d}.gro'
        shutil.copy2(system.fn_coordinates, fn_coordinates)

        fn_mdp_em = None
        if system.fn_mdp_em is not None:
            fn_mdp_em = campaign_dir / f'window-{index:03d}.em.mdp'
            shutil.copy2(system.fn_mdp_em, fn_mdp_em)

        fn_mdp_prod = campaign_dir / f'window-{index:03d}.mdp'
        shutil.copy2(system.fn_mdp_prod, fn_mdp_prod)

        fn_ndx = campaign_dir / f'window-{index:03d}.ndx'
        shutil.copy2(system.fn_ndx, fn_ndx)

        bias = system.bias
        if bias.input_file is not None and bias.input_filename is not None:
            fn_bias = campaign_dir / f'window-{index:03d}.{bias.input_filename}'
            shutil.copy2(bias.input_file, fn_bias)
            bias = type(bias).load(fn_bias)

        staged_systems.append(
            SimulationSystemConfig(
                system_id=system.system_id,
                fn_topol=fn_topol,
                fn_coordinates=fn_coordinates,
                fn_mdp_em=fn_mdp_em,
                fn_mdp_prod=fn_mdp_prod,
                fn_ndx=fn_ndx,
                bias=bias,
                n_steps=system.n_steps,
            )
        )
    return staged_systems


def stage_campaign(
    config: SimulationCampaignConfig,
    *,
    fn_specs: Path | None = None,
) -> tuple[Path | None, list[SimulationSystemConfig]]:
    campaign_dir = config.campaign_dir.resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)

    systems = stage_systems(config.systems, campaign_dir)
    resolved_specs = None if fn_specs is None else fn_specs.resolve()
    save_yaml(
        {
            'systems': [_system_record(system, campaign_dir) for system in systems],
            'samples': {},
        },
        campaign_dir / 'samples.yaml',
    )
    return resolved_specs, systems


def write_sample_job_config(
    *,
    sample_id: str,
    sample: np.ndarray,
    campaign_dir: Path,
    fn_specs: Path,
    gmx_cmd: str,
    job_scheduler: str,
    store: tuple[str, ...],
    systems: list[SimulationSystemConfig],
) -> Path:
    config_md = {
        'sample_id': sample_id,
        'params': np.asarray(sample, dtype=float).tolist(),
        'campaign_dir': str(campaign_dir),
        'fn_specs': str(fn_specs.resolve()),
        'gmx_cmd': gmx_cmd,
        'job_scheduler': job_scheduler,
        'store': list(store),
        'systems': [system.to_dict() for system in systems],
    }
    fn_config_md = campaign_dir / f'config-{sample_id}.yaml'
    save_yaml(config_md, fn_config_md)
    return fn_config_md


def stage_sample_topologies(
    *,
    sample_id: str,
    sample: np.ndarray,
    campaign_dir: Path,
    fn_specs: Path,
    systems: list[SimulationSystemConfig],
) -> None:
    for index, system in enumerate(systems):
        fn_topol = campaign_dir / f'md-{sample_id}-{index:03d}.top'
        modify_topology(system.fn_topol, fn_specs, sample, True, fn_topol)


def build_submission_script(
    *,
    sample_id: str,
    fn_config_md: Path,
    config: SimulationCampaignConfig,
):
    if config.job_scheduler != 'slurm':
        return None

    campaign_dir = config.campaign_dir.resolve()
    assert config.slurm is not None
    fn_stdout = campaign_dir / f'run-{sample_id}.out'
    submit_specs = dict(config.slurm.sbatch or {}) | {'output': fn_stdout}
    return build_slurm_cli_job(
        command=bff_cli_command('md', fn_config_md),
        slurm_config=config.slurm,
        sbatch=submit_specs,
    )


def _result_record(sample_id: str, campaign_dir: Path) -> dict[str, Any]:
    fn_result = campaign_dir / f'result-{sample_id}.yaml'
    if fn_result.exists():
        return load_yaml(fn_result)
    return {}


def collect_campaign_metadata(
    *,
    samples: dict[str, dict[str, Any]],
    systems: list[SimulationSystemConfig],
    campaign_dir: Path,
    compress: bool = False,
    remove: bool = False,
) -> None:
    sample_records: dict[str, Any] = {}
    for sample_id, sample_data in samples.items():
        result = _result_record(sample_id, campaign_dir)
        sample_records[sample_id] = {
            'params': sample_data['params'],
            'job_id': sample_data.get('job_id'),
            'status': result.get('status', sample_data.get('status', 'failed')),
            'outputs': result.get('outputs', sample_data.get('outputs', [])),
        }

    save_yaml(
        {
            'systems': [_system_record(system, campaign_dir) for system in systems],
            'samples': sample_records,
        },
        campaign_dir / 'samples.yaml',
    )

    for fn_result in campaign_dir.glob('result-*.yaml'):
        fn_result.unlink(missing_ok=True)

    if compress:
        compress_results(campaign_dir)
        if remove:
            shutil.rmtree(campaign_dir)
        return

    if remove:
        for pattern in ('run-*.sh', 'run-*.out', 'config-*.yaml'):
            for file in campaign_dir.glob(pattern):
                file.unlink(missing_ok=True)


def print_sample_summary(
    config: SampleConfig,
    fn_specs: PathLike,
    logger: Logger,
) -> None:
    specs = Specs(fn_specs)
    logger.section('Sampling Campaign')
    logger.kv('Campaign directory', config.campaign_dir.resolve())
    logger.kv('Systems', len(config.systems))
    logger.kv('Samples', config.n_samples)
    logger.kv('Scheduler', config.job_scheduler)
    logger.kv('Dispatch', 'yes' if config.dispatch else 'no (stage only)')
    logger.kv('Stored outputs', ', '.join(config.store) if config.store else 'none')
    if not config.dispatch:
        logger.warn('Simulation jobs will only be staged; no MD will be submitted.')
    if not config.store:
        logger.warn(
            'No simulation outputs are configured to be stored after completion.'
        )
    logger.info('parameters:', level=1)
    for name, bounds in specs.bounds.by_name.items():
        label = f'{name}: {bounds}'
        if name in specs.implicit_params:
            label += ' (implicit)'
        logger.info(label, level=2)
    logger.kv('Charge constraints', len(specs.charge_constraints))
    logger.blank()


def print_validate_summary(
    config: SimulationCampaignConfig,
    n_samples: int,
    logger: Logger,
) -> None:
    logger.section('Validation Campaign')
    logger.kv('Log file', config.log.resolve())
    logger.kv('Campaign directory', config.campaign_dir.resolve())
    logger.kv('Systems', len(config.systems))
    logger.kv('Samples', n_samples)
    logger.kv('Scheduler', config.job_scheduler)
    logger.kv('Dispatch', 'yes' if config.dispatch else 'no (stage only)')
    logger.kv('Stored outputs', ', '.join(config.store) if config.store else 'none')
    if not config.dispatch:
        logger.warn('Validation jobs will only be staged; no MD will be submitted.')
    if not config.store:
        logger.warn(
            'No validation outputs are configured to be stored after completion.'
        )
    parameter_source = getattr(config, 'parameters', None)
    if parameter_source is not None:
        logger.kv('Parameter source', Path(parameter_source).resolve())
    logger.blank()


def run_campaign(
    *,
    config: SimulationCampaignConfig,
    fn_specs: Path,
    systems: list[SimulationSystemConfig],
    parameter_samples: np.ndarray,
    logger: Logger,
) -> None:
    n_total = len(parameter_samples)
    samples: dict[str, dict[str, Any]] = {}
    job_ids: list[int] = []
    pad = len(str(max(n_total, 1)))
    campaign_dir = config.campaign_dir.resolve()
    job_scheduler = config.job_scheduler
    action = 'Running MD' if config.dispatch else 'Staging jobs'
    campaign_finished = False
    count_width = len(str(max(n_total, 1)))

    if config.dispatch and job_scheduler not in {'local', *SCHEDULER_CLASSES}:
        supported = ['local', *SCHEDULER_CLASSES]
        raise NotImplementedError(
            f"Unsupported scheduler '{job_scheduler}'. Supported: {supported}"
        )

    def log_job_monitor(counts: dict[str, int], *, overwrite: bool = True) -> None:
        finished_percent = (
            100 if n_total == 0 else 100 * counts['finished'] / n_total
        )
        logger.status(
            'Scheduler jobs',
            (
                f"submitted {counts['submitted']:>{count_width}d}/"
                f"{n_total:<{count_width}d} | "
                f"pending {counts['pending']:>{count_width}d} | "
                f"running {counts['running']:>{count_width}d} | "
                f"finished {counts['finished']:>{count_width}d} "
                f"[{finished_percent:3.0f}%]"
            ),
            level=1,
            overwrite=overwrite,
        )

    try:
        max_parallel_jobs = None
        if config.dispatch and job_scheduler == 'slurm':
            assert config.slurm is not None
            max_parallel_jobs = config.slurm.max_parallel_jobs
            max_parallel_jobs = np.inf if max_parallel_jobs == -1 else max_parallel_jobs

        for idx, sample in enumerate(parameter_samples):
            sample_id = f'{idx:0{pad}d}'
            if (not config.dispatch) or job_scheduler == 'local':
                logger.status(
                    action,
                    (
                        f'{idx + 1:>{pad}d}/{n_total:<{pad}d} '
                        f'[{((idx + 1) / n_total * 100):3.0f}%]'
                    ),
                    level=1,
                    overwrite=True,
                )

            sample = np.asarray(sample, dtype=float).reshape(-1)
            fn_config_md = write_sample_job_config(
                sample_id=sample_id,
                sample=sample,
                campaign_dir=campaign_dir,
                fn_specs=fn_specs,
                gmx_cmd=config.gmx_cmd,
                job_scheduler=config.job_scheduler,
                store=config.store,
                systems=systems,
            )
            samples[sample_id] = {
                'params': sample.tolist(),
                'job_id': None,
                'status': 'staged' if not config.dispatch else 'failed',
                'outputs': [],
            }

            if not config.dispatch:
                stage_sample_topologies(
                    sample_id=sample_id,
                    sample=sample,
                    campaign_dir=campaign_dir,
                    fn_specs=fn_specs,
                    systems=systems,
                )
                submit_script = build_submission_script(
                    sample_id=sample_id,
                    fn_config_md=fn_config_md,
                    config=config,
                )
                if submit_script is not None:
                    submit_script.save(campaign_dir / f'run-{sample_id}.sh')
                continue

            if job_scheduler == 'local':
                subprocess.run(
                    bff_cli_command('md', fn_config_md),
                    cwd=str(campaign_dir),
                    check=True,
                )
                continue

            assert max_parallel_jobs is not None
            wait_for_scheduler_slot(
                job_ids=job_ids,
                scheduler=job_scheduler,
                max_parallel_jobs=max_parallel_jobs,
                monitor=log_job_monitor,
            )
            submit_script = build_submission_script(
                sample_id=sample_id,
                fn_config_md=fn_config_md,
                config=config,
            )
            assert submit_script is not None
            job_id = submit_script.submit(campaign_dir / f'run-{sample_id}.sh')
            job_ids.append(job_id)
            samples[sample_id]['job_id'] = job_id
            log_job_monitor(get_job_state_counts(job_ids, job_scheduler))

        if config.dispatch and job_scheduler == 'slurm':
            control_jobs(job_ids, job_scheduler, monitor=log_job_monitor)
            log_job_monitor(
                get_job_state_counts(job_ids, job_scheduler),
                overwrite=False,
            )

        campaign_finished = True
        logger.done(action, detail=f'{n_total}/{n_total} [100%]', level=1)
    finally:
        collect_campaign_metadata(
            samples=samples,
            systems=systems,
            campaign_dir=campaign_dir,
            compress=(
                config.compress if config.dispatch and campaign_finished else False
            ),
            remove=config.cleanup if config.dispatch and campaign_finished else False,
        )


def build_parameter_samples(config: SampleConfig) -> tuple[Path, np.ndarray]:
    fn_specs = build_specs(config)
    constraint = ChargeConstraint(fn_specs)
    sampler = RandomParamsGenerator(constraint.explicit_bounds, constraint)
    parameter_samples = np.asarray(
        [sampler(1).squeeze(0) for _ in range(config.n_samples)],
        dtype=float,
    )
    return fn_specs, parameter_samples


def _load_yaml_parameter_samples(
    data: dict[str, Any],
    *,
    specs: Specs,
) -> np.ndarray:
    explicit_names = list(specs.parameter_names(explicit_only=True))
    if all(name in data for name in explicit_names):
        lengths = {len(data[name]) for name in explicit_names}
        if len(lengths) != 1:
            raise ValueError(
                'Column-oriented YAML sample lists must all have the same length.'
            )
        return np.column_stack(
            [np.asarray(data[name], dtype=float) for name in explicit_names]
        )

    raise ValueError(
        'Unsupported YAML parameter-sample format. Expected a column-oriented '
        'mapping keyed by explicit parameter names.'
    )


def load_parameter_samples(
    fn_samples: PathLike,
    fn_specs: PathLike,
) -> np.ndarray:
    fn_samples = Path(fn_samples).resolve()
    specs = Specs(fn_specs)
    if fn_samples.suffix != '.yaml':
        raise ValueError(
            f'Unsupported sample file {fn_samples}. Expected a .yaml file.'
        )

    raw = load_yaml(fn_samples)
    if not isinstance(raw, dict):
        raise ValueError(
            f'YAML parameter sample file {fn_samples} must contain a mapping.'
        )
    samples = _load_yaml_parameter_samples(raw, specs=specs)

    if samples.ndim != 2:
        raise ValueError(
            f'Parameter samples must form a 2D array, got shape {samples.shape}.'
        )
    expected = len(specs.parameter_names(explicit_only=True))
    if samples.shape[1] != expected:
        raise ValueError(
            f'Expected {expected} explicit parameter values per sample, got '
            f'{samples.shape[1]}.'
        )
    if samples.shape[0] == 0:
        raise ValueError('No validation parameter samples were found.')
    return samples
