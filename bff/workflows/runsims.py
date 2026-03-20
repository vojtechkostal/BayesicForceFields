import argparse
import time
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Union, List, Dict

from gmxtop import Topology
from ..domain.bias import BiasSpec
from ..topology import TopologyModifier
from ..domain.specs import ChargeConstraint, RandomParamsGenerator, Specs
from ..io.utils import save_yaml, compress_results
from ..io.schedulers import Slurm
from ..io.logs import Logger
from .configs import RunsimsConfig


PathLike = Union[str, Path]


SCHEDULER_CLASSES = {
    'slurm': Slurm,
    # TODO: 'pbs': PBS,
}

def dispatch_md_job(
    *,
    sample_id: str,
    sample: np.ndarray,
    config: RunsimsConfig,
    fn_specs: PathLike,
) -> Union[None, int]:
    """
    Submit an MD simulation either locally or via a job scheduler.
    """

    # Prepare job-specific configurations
    data_dir = config.data_dir.resolve()
    config_md = {
        "hash": sample_id,
        "params": np.asarray(sample, dtype=float).tolist(),
        "data_dir": str(data_dir),
        "fn_specs": str(Path(fn_specs).resolve()),
        "gmx_cmd": config.gmx_cmd,
        "job_scheduler": config.job_scheduler,
        "store": list(config.store),
        "gromacs": config.gromacs.to_dict(),
    }
    fn_config_md = data_dir / f'config-{sample_id}.yaml'
    save_yaml(config_md, fn_config_md)

    cmd_run = ["bff", "md", str(fn_config_md)]

    if config.job_scheduler == 'local':
        subprocess.run(cmd_run, cwd=str(data_dir), check=True)
        return None

    fn_submit = data_dir / f'run-{sample_id}.sh'
    fn_stdout = data_dir / f'run-{sample_id}.out'

    submit_cls = SCHEDULER_CLASSES[config.job_scheduler]
    assert config.slurm is not None
    submit_specs = config.slurm['preamble'] | {'output': fn_stdout}
    submit_script = submit_cls(**submit_specs)

    for cmd in config.slurm['commands']:
        resolved = " ".join(cmd_run) if 'RUN MD' in cmd else cmd
        submit_script.add_command(resolved)

    return submit_script.submit(fn_submit)


# ---- Preparation ----
def prepare_environment(config: RunsimsConfig) -> Path:
    """
    Set up the directory structure and save settings and initial data.
    """

    # Create the data directory
    data_dir = Path(config.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if Specs file is provided or needs to be generated
    if config.validate:
        assert config.fn_specs is not None
        fn_specs = Path(config.fn_specs).resolve()
    else:
        top_modifier = TopologyModifier(
            config.gromacs.fn_topol[0],
            config.mol_resname,
            config.implicit_atoms,
        )
        bounds_resolved = top_modifier.resolve_params(config.bounds)

        # determine charge of the modified group of atoms
        group = []
        for param in bounds_resolved.keys():
            param_name, *atoms = param.split()
            if param_name == 'charge':
                group.extend(atoms)
        target_charge = config.total_charge
        group_charge = top_modifier.group_charge(group)
        constraint_charge = target_charge - top_modifier.total_charge + group_charge

        implicit_atoms = [atom.name for atom in top_modifier.implicit_atoms]

        specs_data = {
            'mol_resname': config.mol_resname,
            'implicit_atoms': implicit_atoms,
            'bounds': bounds_resolved,
            'total_charge': config.total_charge,
            'constraint_charge': constraint_charge,
        }

        # Save the Specs data into a file
        fn_specs = data_dir / 'specs.yaml'
        Specs(specs_data).write(fn_specs)

    files = zip(
        config.gromacs.fn_topol,
        config.gromacs.fn_coordinates,
        config.gromacs.fn_mdp_em,
        config.gromacs.fn_mdp_prod,
        config.gromacs.fn_ndx,
        config.gromacs.fn_bias,
    )
    for i, (
        fn_top,
        fn_coords,
        fn_mdp_em,
        fn_mdp_prod,
        fn_ndx,
        fn_bias,
    ) in enumerate(files):

        # Write topology file
        Topology(fn_top).write(data_dir / f'topol-{i:03d}.top')

        # File mapping: (source, destination filename template)
        files_to_copy = [
            (fn_coords, f'coords-{i:03d}.gro'),
            (fn_mdp_em, f'md-em-{i:03d}.mdp'),
            (fn_mdp_prod, f'md-prod-{i:03d}.mdp'),
            (fn_ndx, f'index-{i:03d}.ndx'),
        ]

        if fn_bias is not None:
            bias = BiasSpec.load(fn_bias)
            if bias.input_file is not None:
                if bias.kind == "colvars":
                    local_bias_input = data_dir / f"bias-{i:03d}.colvars.dat"
                elif bias.kind == "plumed":
                    local_bias_input = data_dir / f"bias-{i:03d}.plumed.dat"
                else:
                    local_bias_input = data_dir / Path(bias.input_file).name
                shutil.copy2(bias.input_file, local_bias_input)

        # Copy files to the data directory
        for src, dst_name in files_to_copy:
            if src is None:
                continue
            dst_path = data_dir / dst_name
            dst_path.write_text(Path(src).read_text())

    return fn_specs


# ---- Job Control ----
def get_active_jobs(ids: List[int], scheduler: str, chunk_size: int = 1000) -> int:
    if scheduler != 'slurm':
        raise NotImplementedError

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    n_active = 0
    for chunk in chunks(ids, chunk_size):
        ids_str = ','.join(map(str, chunk))
        res = subprocess.run(
            ['squeue', '-j', ids_str, '--noheader', '--format', '%i,%t'],
            capture_output=True,
            text=True
        )
        # each line = one active job
        n_active += res.stdout.strip().count('\n')
        # handle case with single line no newline
        if res.stdout.strip() and not res.stdout.endswith('\n'):
            n_active += 1

    return n_active


def control_jobs(job_ids: List[int], scheduler: str) -> None:
    """Monitor active jobs until completion."""
    while True:
        if get_active_jobs(job_ids, scheduler) == 0:
            break
        time.sleep(5)


# ---- Cleanup ----
def clean_up_train_dir(
    samples: Dict[str, List[float]],
    data_dir: PathLike,
    compress: bool = False,
    remove: bool = False
) -> None:
    """
    Remove failed simulations and optionally compress results.
    """

    data_dir = Path(data_dir)
    samples_to_store = {}
    for hash, params in samples.items():
        fn_trial = sorted(data_dir.glob(f'*-{hash}-*.xtc'))  # sort by filename
        if fn_trial and all(f.exists() for f in fn_trial):
            samples_to_store[str(hash)] = {
                'params': params,
                'fn_trj': [f.name for f in fn_trial]
            }

    save_yaml(samples_to_store, data_dir / 'samples.yaml')

    if compress:
        compress_results(data_dir)
        if remove:
            shutil.rmtree(data_dir)
    elif remove:
        patterns = ['run-*.sh', 'run-*.out', 'config-*.yaml']
        for pattern in patterns:
            for f in Path(data_dir).glob(pattern):
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Warning: Failed to remove {f}: {e}")


def print_train_summary(fn_specs: PathLike, logger: Logger) -> None:
    """
    Print a summary of the configuration settings.
    """

    specs = Specs(fn_specs)

    logger.info("", level=0)
    logger.info("=== Generating training set ===\n", level=0)
    logger.info(f"molecule name: {specs.mol_resname}", level=1)
    logger.info("parameters:", level=1)
    for name, b in specs.bounds.by_name.items():
        if name == specs.implicit_param:
            logger.info(f"{name}: {b} (implicit)", level=2)
        else:
            logger.info(f"{name}: {b}", level=2)
    logger.info(f"total charge: {specs.total_charge}\n", level=1)


def print_validate_summary(fn_specs: PathLike, logger: Logger) -> None:
    """
    Print a summary of the configuration settings.
    """
    specs = Specs(fn_specs)
    logger.info("", level=0)
    logger.info("=== Generating validation set ===\n", level=0)
    logger.info(f"molecule name: {specs.mol_resname}\n", level=1)


# ---- Main Workflow ----
def main(fn_config: PathLike) -> None:
    """
    Main function to execute the training set generation.
    """

    # Initialization
    config = RunsimsConfig.load(fn_config)
    fn_specs = prepare_environment(config)

    logger = Logger("runsims")

    if config.validate:
        print_validate_summary(fn_specs, logger)
        assert config.inputs is not None
        inputs = np.load(config.inputs)
        if inputs.ndim != 2:
            raise ValueError("'inputs' must contain a 2D array of parameter samples.")
        iterator = inputs
        n_total = len(inputs)
    else:
        print_train_summary(fn_specs, logger)
        assert config.n_samples is not None
        iterator = range(config.n_samples)
        n_total = config.n_samples

        # prepare the parameter sampler
        constraint = ChargeConstraint(fn_specs)
        sampler = RandomParamsGenerator(constraint.explicit_bounds, constraint)

    # Main loop to generate samples
    samples = {}
    job_ids = []
    pad = len(str(n_total))
    job_scheduler = config.job_scheduler
    n_max = None
    for idx, sample in enumerate(iterator):
        sample_id = f"{idx:0{pad}d}"

        logger.info(
            f"Running MD: {idx+1}/{n_total} "
            f"({((idx + 1) / n_total * 100):.0f}%)",
            level=1,
            overwrite=True
        )

        # Generate sample or use provided input
        if not config.validate:
            max_attempts = 1000
            for _ in range(max_attempts):
                sample = sampler(1)
                if sample.size > 0:
                    sample = sample.squeeze(0)
                    break
            else:
                raise RuntimeError(
                    f"Failed to generate valid sample after {max_attempts} attempts.")
        else:
            sample = np.asarray(sample, dtype=float)

        # Store sample info
        samples[sample_id] = sample.tolist()

        # Submit job while controling total number of running jobs
        if job_scheduler == 'local':
            dispatch_md_job(
                sample_id=sample_id,
                sample=sample,
                config=config,
                fn_specs=fn_specs,
            )

        elif job_scheduler not in SCHEDULER_CLASSES:
            raise NotImplementedError(
                f"Unsupported scheduler '{job_scheduler}'. "
                f"Supported: {list(SCHEDULER_CLASSES)}"
            )

        else:
            assert config.slurm is not None
            n_max = config.slurm.get('max_parallel_jobs', 1)
            n_max = np.inf if n_max == -1 else n_max
            while True:
                if n_max > 0:
                    n_active = get_active_jobs(job_ids, job_scheduler)
                    if n_active < n_max and idx < n_total:
                        id = dispatch_md_job(
                            sample_id=sample_id,
                            sample=sample,
                            config=config,
                            fn_specs=fn_specs,
                        )
                        if id is not None:
                            job_ids.append(id)
                        break
                    else:
                        time.sleep(5)
                else:
                    break

    # Wait for jobs to finish if using a job scheduler
    if job_scheduler != 'local' and n_max is not None and n_max > 0:
        control_jobs(job_ids, job_scheduler)

    logger.info(f"Running MD: {n_total}/{n_total} (100%) | Done.", level=1)

    # Cleanup
    clean_up_train_dir(
        samples,
        config.data_dir,
        compress=config.compress,
        remove=config.cleanup,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Generate training/validation set for MD simulations.'
        )
    )
    parser.add_argument('fn_config', help='Path to the config file [YAML].')
    args = parser.parse_args()
    main(args.fn_config)
