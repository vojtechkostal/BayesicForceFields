import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from gmxtopology import Topology

from ..domain.specs import ChargeConstraint, RandomParamsGenerator, Specs
from ..io.logs import Logger
from ..io.schedulers import Slurm
from ..io.utils import compress_results, load_yaml, save_yaml
from ..topology import TopologyModifier
from .md import modify_topology
from .configs import SimulateConfig, SimulationCampaignConfig, SimulationSystemConfig


PathLike = str | Path


SCHEDULER_CLASSES = {
    "slurm": Slurm,
}


def _relative_path(path: Path | None, base_dir: Path) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(base_dir))


def _system_record(
    system: SimulationSystemConfig,
    trainset_dir: Path,
) -> dict[str, Any]:
    """Serialize one staged system into the campaign metadata file."""
    return {
        "system_id": system.system_id,
        "topology": _relative_path(system.fn_topol, trainset_dir),
        "coordinates": _relative_path(system.fn_coordinates, trainset_dir),
        "mdp": {
            "em": _relative_path(system.fn_mdp_em, trainset_dir),
            "prod": _relative_path(system.fn_mdp_prod, trainset_dir),
        },
        "index": _relative_path(system.fn_ndx, trainset_dir),
        "bias": _relative_path(system.bias.input_file, trainset_dir),
        "n_steps": int(system.n_steps),
    }


def build_specs(config: SimulateConfig) -> Path:
    """Create the force-field specification file for one sampled campaign."""
    config.trainset_dir.resolve().mkdir(parents=True, exist_ok=True)
    top_modifier = TopologyModifier(
        config.systems[0].fn_topol,
        config.mol_resname,
        config.implicit_atoms,
    )
    bounds_resolved = top_modifier.resolve_parameter_names(config.bounds)

    group: list[str] = []
    for name in bounds_resolved:
        param_name, *atoms = name.split()
        if param_name == "charge":
            group.extend(atoms)
    group_charge = top_modifier.group_charge(group)
    constraint_charge = (
        config.total_charge - top_modifier.total_charge + group_charge
    )

    specs = Specs(
        {
            "mol_resname": config.mol_resname,
            "implicit_atoms": [atom.name for atom in top_modifier.implicit_atoms],
            "bounds": bounds_resolved,
            "total_charge": config.total_charge,
            "constraint_charge": constraint_charge,
        }
    )
    fn_specs = config.trainset_dir / "specs.yaml"
    specs.write(fn_specs)
    return fn_specs


def stage_systems(
    systems: list[SimulationSystemConfig],
    trainset_dir: Path,
) -> list[SimulationSystemConfig]:
    """Copy static simulation-system inputs into the campaign directory."""
    staged_systems: list[SimulationSystemConfig] = []
    for index, system in enumerate(systems):
        fn_topol = trainset_dir / f"window-{index:03d}.top"
        Topology(system.fn_topol).write(fn_topol)

        fn_coordinates = trainset_dir / f"window-{index:03d}.gro"
        shutil.copy2(system.fn_coordinates, fn_coordinates)

        fn_mdp_em = None
        if system.fn_mdp_em is not None:
            fn_mdp_em = trainset_dir / f"window-{index:03d}.em.mdp"
            shutil.copy2(system.fn_mdp_em, fn_mdp_em)

        fn_mdp_prod = trainset_dir / f"window-{index:03d}.mdp"
        shutil.copy2(system.fn_mdp_prod, fn_mdp_prod)

        fn_ndx = trainset_dir / f"window-{index:03d}.ndx"
        shutil.copy2(system.fn_ndx, fn_ndx)

        bias = system.bias
        if bias.input_file is not None and bias.input_filename is not None:
            fn_bias = trainset_dir / f"window-{index:03d}.{bias.input_filename}"
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
    """Prepare the trainset directory and write static campaign inputs."""
    trainset_dir = config.trainset_dir.resolve()
    trainset_dir.mkdir(parents=True, exist_ok=True)

    systems = stage_systems(config.systems, trainset_dir)
    if fn_specs is not None:
        resolved_specs = fn_specs.resolve()
    else:
        resolved_specs = None

    save_yaml(
        {
            "systems": [_system_record(system, trainset_dir) for system in systems],
            "samples": {},
        },
        trainset_dir / "samples.yaml",
    )
    return resolved_specs, systems


def write_sample_job_config(
    *,
    sample_id: str,
    sample: np.ndarray,
    trainset_dir: Path,
    fn_specs: Path,
    gmx_cmd: str,
    job_scheduler: str,
    store: tuple[str, ...],
    systems: list[SimulationSystemConfig],
) -> Path:
    """Write one MD job configuration for a sampled force-field vector."""
    config_md = {
        "sample_id": sample_id,
        "params": np.asarray(sample, dtype=float).tolist(),
        "trainset_dir": str(trainset_dir),
        "fn_specs": str(fn_specs.resolve()),
        "gmx_cmd": gmx_cmd,
        "job_scheduler": job_scheduler,
        "store": list(store),
        "systems": [system.to_dict() for system in systems],
    }
    fn_config_md = trainset_dir / f"config-{sample_id}.yaml"
    save_yaml(config_md, fn_config_md)
    return fn_config_md


def stage_sample_topologies(
    *,
    sample_id: str,
    sample: np.ndarray,
    trainset_dir: Path,
    fn_specs: Path,
    systems: list[SimulationSystemConfig],
) -> None:
    """Write modified per-sample topology files into the campaign directory."""
    for i, system in enumerate(systems):
        fn_topol = trainset_dir / f"md-{sample_id}-{i:03d}.top"
        modify_topology(system.fn_topol, fn_specs, sample, True, fn_topol)


def build_submission_script(
    *,
    sample_id: str,
    fn_config_md: Path,
    config: SimulationCampaignConfig,
) -> Slurm | None:
    """Build the Slurm submission script for one staged sample."""
    if config.job_scheduler != "slurm":
        return None

    trainset_dir = config.trainset_dir.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    cmd_run = [sys.executable, "-m", "bff.cli", "md", str(fn_config_md)]

    submit_cls = SCHEDULER_CLASSES[config.job_scheduler]
    assert config.slurm is not None
    fn_stdout = trainset_dir / f"run-{sample_id}.out"
    submit_specs = dict(config.slurm.sbatch or {}) | {"output": fn_stdout}
    submit_script = submit_cls(**submit_specs)

    for cmd in config.slurm.setup:
        submit_script.add_command(cmd)
    pythonpath_prefix = (
        f"PYTHONPATH={shlex.quote(str(repo_root))}"
        '${PYTHONPATH:+":$PYTHONPATH"}'
    )
    submit_script.add_command(f"{pythonpath_prefix} {shlex.join(cmd_run)}")
    for cmd in config.slurm.teardown:
        submit_script.add_command(cmd)

    return submit_script


def write_submission_script(
    *,
    sample_id: str,
    fn_config_md: Path,
    config: SimulationCampaignConfig,
) -> Path | None:
    """Write the Slurm submission script for one staged sample."""
    submit_script = build_submission_script(
        sample_id=sample_id,
        fn_config_md=fn_config_md,
        config=config,
    )
    if submit_script is None:
        return None

    fn_submit = config.trainset_dir.resolve() / f"run-{sample_id}.sh"
    submit_script.save(fn_submit)
    return fn_submit


def dispatch_simulation_job(
    *,
    sample_id: str,
    config: SimulationCampaignConfig,
) -> int | None:
    """Run or submit one MD job for a force-field parameter vector."""
    trainset_dir = config.trainset_dir.resolve()
    fn_config_md = trainset_dir / f"config-{sample_id}.yaml"
    cmd_run = [sys.executable, "-m", "bff.cli", "md", str(fn_config_md)]
    if config.job_scheduler == "local":
        subprocess.run(cmd_run, cwd=str(trainset_dir), check=True)
        return None

    submit_script = build_submission_script(
        sample_id=sample_id,
        fn_config_md=fn_config_md,
        config=config,
    )
    assert submit_script is not None
    fn_submit = trainset_dir / f"run-{sample_id}.sh"
    return submit_script.submit(fn_submit)


def get_active_jobs(ids: list[int], scheduler: str, chunk_size: int = 1000) -> int:
    """Count active jobs for the supported scheduler."""
    if scheduler != "slurm":
        raise NotImplementedError

    def chunks(values: list[int], n: int):
        for i in range(0, len(values), n):
            yield values[i:i+n]

    n_active = 0
    for chunk in chunks(ids, chunk_size):
        ids_str = ",".join(map(str, chunk))
        res = subprocess.run(
            ["squeue", "-j", ids_str, "--noheader", "--format", "%i,%t"],
            capture_output=True,
            text=True,
            check=False,
        )
        n_active += res.stdout.strip().count("\n")
        if res.stdout.strip() and not res.stdout.endswith("\n"):
            n_active += 1

    return n_active


def wait_for_scheduler_slot(
    *,
    job_ids: list[int],
    scheduler: str,
    max_parallel_jobs: float,
) -> None:
    """Wait until the scheduler has capacity for one more submitted job."""
    while get_active_jobs(job_ids, scheduler) >= max_parallel_jobs:
        time.sleep(5)


def control_jobs(job_ids: list[int], scheduler: str) -> None:
    """Block until all submitted jobs finish."""
    while True:
        if get_active_jobs(job_ids, scheduler) == 0:
            break
        time.sleep(5)


def _result_record(sample_id: str, trainset_dir: Path) -> dict[str, Any]:
    fn_result = trainset_dir / f"result-{sample_id}.yaml"
    if fn_result.exists():
        return load_yaml(fn_result)
    return {}


def collect_campaign_metadata(
    *,
    samples: dict[str, dict[str, Any]],
    systems: list[SimulationSystemConfig],
    trainset_dir: Path,
    compress: bool = False,
    remove: bool = False,
) -> None:
    """Merge per-sample result files into the final campaign metadata file."""
    sample_records: dict[str, Any] = {}
    for sample_id, sample_data in samples.items():
        result = _result_record(sample_id, trainset_dir)
        sample_records[sample_id] = {
            "params": sample_data["params"],
            "job_id": sample_data.get("job_id"),
            "status": result.get("status", sample_data.get("status", "failed")),
            "outputs": result.get("outputs", sample_data.get("outputs", [])),
        }

    save_yaml(
        {
            "systems": [_system_record(system, trainset_dir) for system in systems],
            "samples": sample_records,
        },
        trainset_dir / "samples.yaml",
    )

    for fn_result in trainset_dir.glob("result-*.yaml"):
        fn_result.unlink(missing_ok=True)

    if compress:
        compress_results(trainset_dir)
        if remove:
            shutil.rmtree(trainset_dir)
        return

    if remove:
        patterns = ["run-*.sh", "run-*.out", "config-*.yaml"]
        for pattern in patterns:
            for file in trainset_dir.glob(pattern):
                file.unlink(missing_ok=True)


def print_simulate_summary(
    config: SimulateConfig,
    fn_specs: PathLike,
    logger: Logger,
) -> None:
    """Print a concise summary of the sampled simulation campaign."""
    specs = Specs(fn_specs)
    logger.info("", level=0)
    logger.info("=== Running simulation campaign ===\n", level=0)
    logger.info(f"molecule name: {specs.mol_resname}", level=1)
    logger.info(f"trainset dir: {config.trainset_dir.resolve()}", level=1)
    logger.info(
        f"systems: {len(config.systems)} | samples: {config.n_samples} "
        f"| scheduler: {config.job_scheduler}",
        level=1,
    )
    logger.info(
        f"dispatch: {'yes' if config.dispatch else 'no (stage only)'}",
        level=1,
    )
    logger.info(
        f"stored outputs: {', '.join(config.store) if config.store else 'none'}",
        level=1,
    )
    logger.info("parameters:", level=1)
    for name, bounds in specs.bounds.by_name.items():
        label = f"{name}: {bounds}"
        if name == specs.implicit_param:
            label += " (implicit)"
        logger.info(label, level=2)
    logger.info(f"total charge: {specs.total_charge}\n", level=1)


def print_validate_summary(
    config: SimulationCampaignConfig,
    fn_specs: PathLike,
    n_samples: int,
    logger: Logger,
) -> None:
    """Print a concise summary of the validation campaign."""
    specs = Specs(fn_specs)
    logger.info("", level=0)
    logger.info("=== Running validation campaign ===\n", level=0)
    logger.info(f"molecule name: {specs.mol_resname}", level=1)
    logger.info(f"trainset dir: {config.trainset_dir.resolve()}", level=1)
    logger.info(
        f"systems: {len(config.systems)} | samples: {n_samples} "
        f"| scheduler: {config.job_scheduler}",
        level=1,
    )
    logger.info(
        f"dispatch: {'yes' if config.dispatch else 'no (stage only)'}",
        level=1,
    )
    logger.info(
        f"stored outputs: {', '.join(config.store) if config.store else 'none'}\n",
        level=1,
    )
    fn_samples = getattr(config, "fn_samples", None)
    if fn_samples is not None:
        logger.info(f"sample source: {Path(fn_samples).resolve()}\n", level=1)


def run_campaign(
    *,
    config: SimulationCampaignConfig,
    fn_specs: Path,
    systems: list[SimulationSystemConfig],
    parameter_samples: np.ndarray,
    logger: Logger,
) -> None:
    """Run a local or Slurm-backed simulation campaign."""
    n_total = len(parameter_samples)
    samples: dict[str, dict[str, Any]] = {}
    job_ids: list[int] = []
    pad = len(str(max(n_total, 1)))
    trainset_dir = config.trainset_dir.resolve()
    job_scheduler = config.job_scheduler
    max_parallel_jobs = None
    action = "Running MD" if config.dispatch else "Staging jobs"
    campaign_finished = False

    if config.dispatch and job_scheduler not in {"local", *SCHEDULER_CLASSES}:
        raise NotImplementedError(
            f"Unsupported scheduler '{job_scheduler}'. "
            f"Supported: {['local', *SCHEDULER_CLASSES]}"
        )

    try:
        for idx, sample in enumerate(parameter_samples):
            sample_id = f"{idx:0{pad}d}"
            logger.info(
                f"{action}: {idx + 1}/{n_total} "
                f"({((idx + 1) / n_total * 100):.0f}%)",
                level=1,
                overwrite=True,
            )

            sample = np.asarray(sample, dtype=float).reshape(-1)
            fn_config_md = write_sample_job_config(
                sample_id=sample_id,
                sample=sample,
                trainset_dir=trainset_dir,
                fn_specs=fn_specs,
                gmx_cmd=config.gmx_cmd,
                job_scheduler=config.job_scheduler,
                store=config.store,
                systems=systems,
            )
            samples[sample_id] = {
                "params": sample.tolist(),
                "job_id": None,
                "status": "staged" if not config.dispatch else "failed",
                "outputs": [],
            }

            if not config.dispatch:
                stage_sample_topologies(
                    sample_id=sample_id,
                    sample=sample,
                    trainset_dir=trainset_dir,
                    fn_specs=fn_specs,
                    systems=systems,
                )
                write_submission_script(
                    sample_id=sample_id,
                    fn_config_md=fn_config_md,
                    config=config,
                )
                continue

            if job_scheduler == "local":
                dispatch_simulation_job(
                    sample_id=sample_id,
                    config=config,
                )
                continue

            assert config.slurm is not None
            max_parallel_jobs = config.slurm.max_parallel_jobs
            max_parallel_jobs = np.inf if max_parallel_jobs == -1 else max_parallel_jobs
            if max_parallel_jobs > 0:
                wait_for_scheduler_slot(
                    job_ids=job_ids,
                    scheduler=job_scheduler,
                    max_parallel_jobs=max_parallel_jobs,
                )
                job_id = dispatch_simulation_job(
                    sample_id=sample_id,
                    config=config,
                )
                if job_id is not None:
                    job_ids.append(job_id)
                    samples[sample_id]["job_id"] = job_id

        if (
            config.dispatch
            and job_scheduler != "local"
            and max_parallel_jobs is not None
            and max_parallel_jobs > 0
        ):
            control_jobs(job_ids, job_scheduler)

        campaign_finished = True
        logger.info(f"{action}: {n_total}/{n_total} (100%) | Done.", level=1)
    finally:
        collect_campaign_metadata(
            samples=samples,
            systems=systems,
            trainset_dir=trainset_dir,
            compress=(
                config.compress
                if config.dispatch and campaign_finished
                else False
            ),
            remove=(
                config.cleanup
                if config.dispatch and campaign_finished
                else False
            ),
        )


def build_parameter_samples(config: SimulateConfig) -> tuple[Path, np.ndarray]:
    """Build ``specs.yaml`` and sample explicit parameters for one campaign."""
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

    if isinstance(data.get("samples"), list):
        records = data["samples"]
        parameter_names = list(data.get("parameter_names", explicit_names))
        rows: list[list[float]] = []
        for i, record in enumerate(records):
            if isinstance(record, dict) and isinstance(record.get("params"), dict):
                params = record["params"]
                missing = [name for name in explicit_names if name not in params]
                if missing:
                    raise ValueError(
                        f"Sample record {i} is missing parameter(s): "
                        + ", ".join(repr(name) for name in missing)
                    )
                rows.append([float(params[name]) for name in explicit_names])
                continue
            if isinstance(record, dict) and isinstance(record.get("params"), list):
                values = np.asarray(record["params"], dtype=float).reshape(-1)
                if values.size != len(parameter_names):
                    raise ValueError(
                        f"Sample record {i} has {values.size} values, expected "
                        f"{len(parameter_names)}."
                    )
                params = dict(zip(parameter_names, values.tolist()))
                missing = [name for name in explicit_names if name not in params]
                if missing:
                    raise ValueError(
                        f"Sample record {i} is missing parameter(s): "
                        + ", ".join(repr(name) for name in missing)
                    )
                rows.append([float(params[name]) for name in explicit_names])
                continue
            if isinstance(record, list):
                values = np.asarray(record, dtype=float).reshape(-1)
                if values.size != len(parameter_names):
                    raise ValueError(
                        f"Sample row {i} has {values.size} values, expected "
                        f"{len(parameter_names)}."
                    )
                params = dict(zip(parameter_names, values.tolist()))
                missing = [name for name in explicit_names if name not in params]
                if missing:
                    raise ValueError(
                        f"Sample row {i} is missing parameter(s): "
                        + ", ".join(repr(name) for name in missing)
                    )
                rows.append([float(params[name]) for name in explicit_names])
                continue
            raise ValueError(
                f"Unsupported YAML sample record at index {i}: {record!r}."
            )
        return np.asarray(rows, dtype=float)

    if all(name in data for name in explicit_names):
        lengths = {len(data[name]) for name in explicit_names}
        if len(lengths) != 1:
            raise ValueError(
                "Column-oriented YAML sample lists must all have the same length."
            )
        return np.column_stack(
            [np.asarray(data[name], dtype=float) for name in explicit_names]
        )

    raise ValueError(
        "Unsupported YAML parameter-sample format. Expected either a top-level "
        "'samples' list or a column-oriented mapping keyed by explicit "
        "parameter names."
    )


def load_parameter_samples(
    fn_samples: PathLike,
    fn_specs: PathLike,
) -> np.ndarray:
    """Load validation parameter samples from ``.npy`` or YAML."""
    fn_samples = Path(fn_samples).resolve()
    specs = Specs(fn_specs)

    if fn_samples.suffix == ".npy":
        samples = np.asarray(np.load(fn_samples), dtype=float)
    elif fn_samples.suffix in {".yaml", ".yml"}:
        raw = load_yaml(fn_samples)
        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML parameter sample file {fn_samples} must contain a mapping."
            )
        samples = _load_yaml_parameter_samples(raw, specs=specs)
    else:
        raise ValueError(
            f"Unsupported sample file {fn_samples}. Expected .npy, .yaml, or .yml."
        )

    if samples.ndim != 2:
        raise ValueError(
            f"Parameter samples must form a 2D array, got shape {samples.shape}."
        )
    expected = len(specs.parameter_names(explicit_only=True))
    if samples.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} explicit parameter values per sample, got "
            f"{samples.shape[1]}."
        )
    if samples.shape[0] == 0:
        raise ValueError("No validation parameter samples were found.")
    return samples
