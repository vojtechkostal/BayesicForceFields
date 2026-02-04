import argparse
import time
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import Any, Union, Tuple, List, Dict

from gmxtop import Topology
from ..topology import TopologyModifier
from ..structures import Specs, RandomParamsGenerator
from ..io.utils import load_yaml, save_yaml, compress_results
from ..io.schedulers import Slurm
from ..io.logs import Logger


PathLike = Union[str, Path]


SCHEDULER_CLASSES = {
    'slurm': Slurm,
    # TODO: 'pbs': PBS,
}

MD_SCRIPT = 'BayesicForceFields.bff.workflows.md'


def load_config(config: PathLike) -> Tuple[dict, bool]:
    """
    Check if the configuration is valid.
    """

    # load the configuration file
    config = load_yaml(config)

    # Check the mandatory keys
    required_keys = ['data_dir', 'gromacs', 'job_scheduler', 'gmx_cmd']
    validate = 'inputs' in config
    if validate:
        required_keys.extend(['inputs', 'fn_specs'])
    else:
        required_keys.extend(
            ['mol_resname', 'bounds', 'total_charge', 'implicit_atoms', 'n_samples']
        )

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: '{key}'")
        if key in ['fn_specs', 'data_dir']:
            config[key] = str(Path(config[key]).resolve())

    # --- GROMACS section ---
    gmx_required = [
        'fn_topol', 'fn_coordinates', 'fn_mdp_em', 'fn_mdp_prod', 'fn_ndx', 'n_steps'
    ]
    gmx = config['gromacs']
    for key in gmx_required:
        if key not in gmx:
            raise ValueError(f"Missing required gromacs key: '{key}'")

    # All list keys except n_steps must be the same length
    file_keys = [k for k in gmx_required if k != 'n_steps']
    lengths = [len(gmx[k]) for k in file_keys]
    if len(set(lengths)) != 1:
        raise ValueError(
            "GROMACS file lists must have same length, "
            f"got lengths: {dict(zip(file_keys, lengths))}"
        )

    # Check if all the GROMACS files exist
    for key in file_keys:
        resolved_paths = []
        for f in gmx[key]:
            if not Path(f).is_file():
                raise ValueError(f"File not found: {f} (for '{key}')")
            resolved_paths.append(str(Path(f).resolve()))
        config['gromacs'][key] = resolved_paths

    # --- Bounds ---
    if not isinstance(config['bounds'], dict):
        raise ValueError("'bounds' must be a dict")
    for name, b in config['bounds'].items():
        if not (
            isinstance(b, (list, tuple))
            and len(b) == 2
            and all(isinstance(x, (int, float)) for x in b)
        ):
            raise ValueError(f"Invalid bounds for '{name}': {b}")

    # --- Check whether implicit atoms are part of bounds ---
    bound_keys = set(
        b.split(maxsplit=1)[1]
        for b in config['bounds'].keys()
        if 'charge' in b
    )
    if isinstance(config['implicit_atoms'], str):
        implicit_atoms_str = config['implicit_atoms']
    elif isinstance(config['implicit_atoms'], (list, tuple)):
        implicit_atoms_str = ' '.join(config['implicit_atoms'])
    else:
        raise ValueError("'implicit_atoms' must be a string or list of strings")

    if bound_keys and implicit_atoms_str not in bound_keys:
        raise ValueError(
            f"'implicit_atoms' ({implicit_atoms_str}) "
            "must be one of the defined bounds."
        )

    # --- Numeric fields ---
    if not isinstance(config['total_charge'], (int, float)):
        raise ValueError("'total_charge' must be float")
    if (
        'n_samples' in config
        and (
            not isinstance(config['n_samples'], int)
            or config['n_samples'] <= 0
        )
    ):
        raise ValueError("'n_samples' must be a positive integer")

    # --- Scheduler ---
    scheduler = config.get('job_scheduler', 'local')
    if scheduler != 'local':
        if scheduler not in SCHEDULER_CLASSES:
            raise ValueError(
                f"Unsupported scheduler '{scheduler}'. "
                f"Supported: {list(SCHEDULER_CLASSES)}"
            )
        if scheduler not in config:
            raise ValueError(f"Missing scheduler settings for '{scheduler}'")
        sched_conf = config[scheduler]
        required_sched_keys = ['preamble', 'commands']
        for key in required_sched_keys:
            if key not in sched_conf:
                raise ValueError(f"Scheduler '{scheduler}' must define '{key}'")

    # --- misc ---
    config['compress'] = config.get('compress', False)
    config['cleanup'] = config.get('cleanup', False)
    config['store'] = config.get('store', ['xtc'])

    return config, validate


def dispatch_md_job(
    hash: str,
    sample: List[float],
    config: Dict[str, Any],
    job_scheduler: object
) -> Union[None, int]:
    """
    Submit an MD simulation either locally or via a job scheduler.
    """

    # Prepare job-specific configurations
    config_md = config | {'params': sample, 'hash': hash}
    data_dir = Path(config_md['data_dir']).resolve()
    fn_config_md = data_dir / f'config-{hash}.yaml'
    save_yaml(config_md, fn_config_md)

    cmd_run = ["bff", "md", str(fn_config_md)]

    if job_scheduler == 'local':
        subprocess.run(cmd_run, cwd=str(data_dir))
        return None

    fn_submit = data_dir / f'run-{hash}.sh'
    fn_stdout = data_dir / f'run-{hash}.out'

    submit_cls = SCHEDULER_CLASSES[job_scheduler]
    submit_specs = config[job_scheduler]['preamble'] | {'output': fn_stdout}
    submit_script = submit_cls(**submit_specs)

    if config.get(job_scheduler) is None:
        raise ValueError(
            f"Configuration for job scheduler '{job_scheduler}' is missing."
        )

    for cmd in config[job_scheduler]['commands']:
        resolved = " ".join(cmd_run) if 'RUN MD' in cmd else cmd
        submit_script.add_command(resolved)

    return submit_script.submit(fn_submit)


# ---- Initialization ----
def initialize_environment(config: Dict[str, Any], validate: bool) -> Path:
    """
    Set up the directory structure and save settings and initial data.
    """

    # Create the data directory
    data_dir = Path(config['data_dir']).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if Specs file is provided or needs to be generated
    if validate:
        fn_specs = Path(config['fn_specs']).resolve()
    else:
        top_modifier = TopologyModifier(
            config['gromacs']['fn_topol'][0],
            config['mol_resname'],
            config['implicit_atoms'])
        bounds_resolved = top_modifier.resolve_params(config['bounds'])

        # determine charge of the modified group of atoms
        group = []
        for param in bounds_resolved.keys():
            param_name, *atoms = param.split()
            if param_name == 'charge':
                group.extend(atoms)
        target_charge = config['total_charge']
        group_charge = top_modifier.group_charge(group)
        constraint_charge = target_charge - top_modifier.total_charge + group_charge

        atoms = {atom.name: atom.type.name for atom in top_modifier.mol.atoms}
        implicit_atoms = [atom.name for atom in top_modifier.implicit_atoms]

        specs_data = {
            'mol_resname': config['mol_resname'],
            'atoms': atoms,
            'implicit_atoms': implicit_atoms,
            'bounds': bounds_resolved,
            'total_charge': config['total_charge'],
            'constraint_charge': constraint_charge,
        }

        config['bounds'] = bounds_resolved
        config['constraint_charge'] = constraint_charge
        config['implicit_atoms'] = implicit_atoms

        # Check if all targetted atoms are present in the molecules
        # and check for clashes with implicit atoms or duplicate definitions
        # additionaly, modify bounds to be named by atom names in the case of charges

        # Save the Specs data into a file
        fn_specs = data_dir / 'specs.yaml'
        Specs(specs_data).save(fn_specs)

    gmx_keys = ['fn_topol', 'fn_coordinates', 'fn_mdp_em', 'fn_mdp_prod', 'fn_ndx']
    files = zip(*(config['gromacs'][key] for key in gmx_keys))
    for i, (fn_top, fn_coords, fn_mdp_em, fn_mdp_prod, fn_ndx) in enumerate(files):

        # Write topology file
        Topology(fn_top).write(str(data_dir / f'topol-{i:03d}.top'))

        # File mapping: (source, destination filename template)
        file_map = {
            fn_coords: f'coords-{i:03d}.gro',
            fn_mdp_em: f'md-em-{i:03d}.mdp',
            fn_mdp_prod: f'md-prod-{i:03d}.mdp',
            fn_ndx: f'index-{i:03d}.ndx',
        }

        # Copy files to the data directory
        for src, dst_name in file_map.items():
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
            samples_to_store[hash] = {
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
    for name, b in specs.bounds_explicit._bounds.items():
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
    config, validate = load_config(fn_config)
    fn_specs = initialize_environment(config, validate)

    logger = Logger(None)

    if validate:
        print_validate_summary(fn_specs, logger)
        inputs = np.load(config['inputs'])
        iterator = inputs
        n_total = len(inputs)
    else:
        print_train_summary(fn_specs, logger)
        sampler = RandomParamsGenerator(fn_specs)
        iterator = range(config['n_samples'])
        n_total = config['n_samples']

    # Main loop to generate samples
    samples = {}
    job_ids = []
    pad = len(str(n_total))
    for idx, sample in enumerate(iterator):

        logger.info(
            f"Running MD: {idx+1}/{n_total} "
            f"({((idx + 1) / n_total * 100):.0f}%)",
            level=1,
            overwrite=True
        )

        # Generate sample or use provided input
        if not validate:
            hash = f"{idx:0{pad}d}"
            max_attempts = 1000
            for _ in range(max_attempts):
                sample = sampler.generate(1)
                if sample.size > 0:
                    sample = sample.squeeze(0)
                    break
            else:
                raise RuntimeError(
                    f"Failed to generate valid sample after {max_attempts} attempts.")

        # Store sample info
        samples[hash] = sample

        # Submit job while controling total number of running jobs
        job_scheduler = config.get('job_scheduler', 'local')
        if job_scheduler == 'local':
            dispatch_md_job(hash, sample, config, job_scheduler)

        elif job_scheduler not in SCHEDULER_CLASSES:
            raise NotImplementedError(
                f"Unsupported scheduler '{job_scheduler}'. "
                f"Supported: {list(SCHEDULER_CLASSES)}"
            )

        else:
            n_max = config[job_scheduler].get('max_parallel_jobs', 1)
            n_max = np.inf if n_max == -1 else n_max
            while True:
                if n_max > 0:
                    n_active = get_active_jobs(job_ids, job_scheduler)
                    if n_active < n_max and idx < n_total:
                        id = dispatch_md_job(hash, sample, config, job_scheduler)
                        if id is not None:
                            job_ids.append(id)
                        break
                    else:
                        time.sleep(5)
                else:
                    break

    # Wait for jobs to finish if using a job scheduler
    if job_scheduler != 'local' and n_max > 0:
        control_jobs(job_ids, job_scheduler)

    logger.info(f"Running MD: {n_total}/{n_total} (100%) | Done.", level=1)

    # Cleanup
    clean_up_train_dir(
        samples,
        Path(config['data_dir']),
        compress=config.get('compress', False),
        remove=config.get('cleanup', False)
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
