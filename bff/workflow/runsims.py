import argparse
import time
import subprocess
import shutil
import numpy as np
from pathlib import Path
from ..topology import TopologyParser
from ..structures import Specs, RandomParamsGenerator
from ..io.utils import load_yaml, save_yaml, compress_results
from ..io.schedulers import Slurm


SCHEDULER_CLASSES = {
    'slurm': Slurm,
    # TODO: 'pbs': PBS,
}

PATH_MD_SCRIPT = 'BayesicForceFields/bff/workflow/simulate'


def resolve_config_paths(config: dict) -> dict:
    # Resolve key paths
    for key in ['fn_specs', 'data_dir']:
        if key in config:
            config[key] = str(Path(config[key]).resolve())
    
    # Resolve GROMACS file paths
    for fn_group, items in config['gromacs'].items():
        if fn_group == 'n_steps':
            config['gromacs'][fn_group] = items
        else:
            config['gromacs'][fn_group] = [
                str(Path(f).resolve()) for f in items
            ]

    return config


def dispatch_md_job(hash, sample, config, job_scheduler):
    """
    Submit an MD simulation either locally or via a job scheduler.
    """

    # Prepare job-specific configurations
    config_md = resolve_config_paths(config) | {'params': sample, 'hash': hash}
    data_dir = Path(config_md['data_dir'])
    fn_config_md = data_dir / f'config-{hash}.yaml'
    save_yaml(config_md, fn_config_md)

    cmd_run = [config['python'], '-m', str(PATH_MD_SCRIPT), '-f', str(fn_config_md)]

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
        resolved = ' '.join(map(str, cmd_run)) if 'RUN MD' in cmd else cmd
        submit_script.add_command(resolved)

    return submit_script.submit(fn_submit)


# ---- Initialization ----
def initialize_environment(config):
    """
    Set up the directory structure and save settings and initial data.
    """

    # Create the data directory
    data_dir = Path(config['data_dir']).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if Specs file is provided or needs to be generated
    if not config.get('fn_specs'):
        topol = TopologyParser(config['gromacs']['fn_topol'][0]) # NOTE: assumes the mol_resname is the same for all topologies
        topol.select_mol(config['mol_resname'], config['implicit_atomtype'])
        specs_data = {
            'mol_resname': config['mol_resname'],
            'atomtype_counts': topol.mol_atomtype_counts,
            'implicit_atomtype': config['implicit_atomtype'],
            'bounds': config['bounds'],
            'total_charge': config['total_charge'],
        }

        # Save the Specs data into a file
        fn_specs = data_dir / 'specs.yaml'
        Specs(specs_data).save(fn_specs)
    else:
        fn_specs = Path(config['fn_specs']).resolve()

    gmx_keys = ['fn_topol', 'fn_coordinates', 'fn_mdp_em', 'fn_mdp_prod', 'fn_ndx']
    files = zip(*(config['gromacs'][key] for key in gmx_keys))
    for i, (fn_top, fn_coords, fn_mdp_em, fn_mdp_prod, fn_ndx) in enumerate(files):

        # Write topology file
        TopologyParser(fn_top).write(str(data_dir / f'topol-{i:03d}.top'))

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


# ---- Job Monitoring ----
def get_active_jobs(ids, scheduler):
    """Get the number of active jobs from a list of job IDs."""
    ids_str = ','.join(ids)

    if scheduler == 'slurm':
        jobs = subprocess.run(
            ['squeue', '-j', ids_str, '--noheader', '--format', '%i,%t'],
            capture_output=True,
            text=True)
    else:
        raise NotImplementedError(
            f"Job scheduler '{scheduler}' is not supported for job monitoring."
        )
    n_active = jobs.stdout.count('\n')
    return n_active


def control_jobs(job_ids, scheduler):
    """Monitor active jobs until completion."""
    while True:
        if get_active_jobs(job_ids, scheduler) == 0:
            break
        time.sleep(5)


# ---- Cleanup ----
def clean_up_train_dir(samples, data_dir, compress=False, remove=False):
    """
    Remove failed simulations and optionally compress results.
    """

    data_dir = Path(data_dir)
    samples_to_store = {}
    for hash, params in samples.items():
        fn_trial = sorted(data_dir.glob(f'*{hash}*.xtc'))  # sort by filename
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


def print_train_summary(config):
    """
    Print a summary of the configuration settings.
    """
    print()
    print(f'> Generating training set for: {config['mol_resname']}')
    print("  > parameters:")
    for name, b in config['bounds'].items():
        print(f"    {name}: {b}")
    print(f"  > total charge: {config['total_charge']}")
    print()


def print_validate_summary(config):
    """
    Print a summary of the configuration settings.
    """
    print()
    print(f'> Generating validation set for: {config['mol_resname']}')
    print()


# ---- Main Workflow ----
def main(fn_config):
    """
    Main function to execute the training set generation.
    """

    # Initialization
    config = load_yaml(fn_config)
    fn_specs = initialize_environment(config)
    validate = 'inputs' in config

    if validate:
        print_validate_summary(config)
        inputs = np.load(config['inputs'])
        iterator = inputs
        n_total = len(inputs)
    else:
        print_train_summary(config)
        sampler = RandomParamsGenerator(fn_specs)
        iterator = range(config['n_samples'])
        n_total = config['n_samples']

    # Main loop to generate samples
    samples = {}
    job_ids = []
    for idx, p in enumerate(iterator):

        print(
            f"> Running MD: {idx+1}/{n_total} "
            f"({((idx + 1) / n_total * 100):.0f}%)\r",
            end='', flush=True
        )

        # Generate sample or use provided input
        if validate:
            hash, sample = str(idx), p
        else:
            max_attempts = 1000
            for _ in range(max_attempts):
                hash, sample = sampler.generate(1, assign_hash=True)
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

    print(f'> Running MD: {n_total}/{n_total} (100%) | Done.'.ljust(50))

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
