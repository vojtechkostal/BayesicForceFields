import argparse
import os
import time
import subprocess
import numpy as np
from pathlib import Path
from ..topology import TopologyParser
from ..structures import Specs, RandomParamsGenerator
from ..io.utils import load_yaml, save_yaml, compress_results
from ..io.slurm import Slurm


def dispatch_md_job(hash, sample, config):
    """
    Submit an MD simulation job using Slurm.
    """

    config_md = config | {'params': sample, 'hash': hash}
    data_dir = Path(config_md['data_dir']).resolve()
    FN_MD_SCRIPT = 'BayesicForceFields.bff.workflow.simulate'

    # Change to full paths
    if config_md.get('fn_specs'):
        config_md['fn_specs'] = str(Path(config_md['fn_specs']).resolve())
    config_md['data_dir'] = str(Path(config_md['data_dir']).resolve())
    for fn_group, items in config_md['gromacs'].items():
        if fn_group != 'n_steps':
            config_md['gromacs'][fn_group] = [
                str(Path(f).resolve()) for f in items
            ]
        else:
            config_md['gromacs'][fn_group] = items

    # Save the simulation-specific config file
    fn_config_md = data_dir / f'config-{hash}.yaml'
    save_yaml(config_md, fn_config_md)

    cmd_list = [
        config['python'],
        '-m',
        FN_MD_SCRIPT,
        '-f',
        fn_config_md
    ]

    if config['execution_mode'] == 'slurm':
        fn_submit = data_dir / f'run-{hash}.sh'
        fn_slurm = data_dir / f'slurm-{hash}.out'

        slurm_script = Slurm(**config['slurm']['sbatch'] | {'output': fn_slurm})
        for cmd in config['slurm']['commands']:
            if 'RUN MD' in cmd:
                cmd = ' '.join(map(str, cmd_list))
            slurm_script.add_command(cmd)
        id = slurm_script.submit(fn_submit)
        return id

    elif config.get('execution_mode') == 'local':
        subprocess.run(cmd_list, cwd=str(data_dir))


# ---- Initialization ----
def initialize_environment(config):
    """
    Set up the directory structure and save settings and initial data.
    """

    data_dir = Path(config['data_dir']).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not config.get('fn_specs'):
        topol = TopologyParser(config['gromacs']['fn_topol'][0])
        topol.select_mol(config['mol_resname'], config['implicit_atomtype'])
        specs_data = {
            'mol_resname': config['mol_resname'],
            'atomtype_counts': topol.mol_atomtype_counts,
            'implicit_atomtype': config['implicit_atomtype'],
            'bounds': config['bounds'],
            'total_charge': config['total_charge'],
        }

        fn_specs = data_dir / 'specs.yaml'
        Specs(specs_data).save(fn_specs)
    else:
        fn_specs = Path(config['fn_specs']).resolve()

    for i, (fn_top, fn_coords, fn_mdp, fn_ndx) in enumerate(
        zip(
            config['gromacs']['fn_topol'],
            config['gromacs']['fn_coordinates'],
            config['gromacs']['fn_mdp_prod'],
            config['gromacs']['fn_ndx']
        )
    ):
        top = TopologyParser(fn_top)
        fn_top_out = data_dir / f'topol-{i:03d}.top'
        top.write(str(fn_top_out))

        fn_coords_out = data_dir / f'coords-{i:03d}.gro'
        fn_coords_out.write_text(Path(fn_coords).read_text())

        fn_mdp_out = data_dir / f'nvt-{i:03d}.mdp'
        fn_mdp_out.write_text(Path(fn_mdp).read_text())

        fn_ndx_out = data_dir / f'index-{i:03d}.ndx'
        fn_ndx_out.write_text(Path(fn_ndx).read_text())

    return fn_specs


# ---- Job Monitoring ----
def get_active_jobs(ids):
    """Get the number of active jobs from a list of job IDs."""
    ids_str = ','.join(ids)
    jobs = subprocess.run(
        ['squeue', '-j', ids_str, '--noheader', '--format', '%i,%t'],
        capture_output=True,
        text=True)
    n_active = jobs.stdout.count('\n')
    return n_active


def control_jobs(job_ids):
    """Monitor active jobs until completion."""
    while True:
        if get_active_jobs(job_ids) == 0:
            break
        time.sleep(5)


# ---- Cleanup ----
def clean_up_train_dir(samples, data_dir, compress=False, remove=False):
    """
    Remove failed simulations and optionally compress results.
    """

    samples_to_store = {}
    for hash, params in samples.items():
        fn_trial = sorted(data_dir.glob(f'*{hash}*.xtc'))  # sort by filename
        if fn_trial and all(f.exists() for f in fn_trial):
            samples_to_store[hash] = {
                'params': params,
                'fn_trj': [f.name for f in fn_trial]
            }

    save_yaml(samples_to_store, os.path.join(data_dir, 'samples.yaml'))

    if compress:
        compress_results(data_dir)
        if remove:
            os.system(f'rm -rf {data_dir}')
    elif remove:
        os.system(f'rm {data_dir}/slurm* {data_dir}/config-* {data_dir}/run-*')


def print_train_summary(config):
    """
    Print a summary of the configuration settings.
    """
    print()
    print(' Generating training set '.center(40, '-'))
    print()
    print(f"> molecule: {config['mol_resname']}")
    print("> parameters:")
    for name, b in config['bounds'].items():
        print(f"    {name}: {b}")
    print(f"> total charge: {config['total_charge']}")
    print()


def print_validate_summary(config):
    """
    Print a summary of the configuration settings.
    """
    print()
    print(' Generating validation set '.center(40, '-'))
    print()


# ---- Main Workflow ----
def main(mode, fn_config):
    """
    Main function to execute the training set generation.
    """

    # Initialization
    config = load_yaml(fn_config)
    fn_specs = initialize_environment(config)
    execution_mode = config.get('execution_mode', 'local')

    if mode == 'train':
        print_train_summary(config)
        sampler = RandomParamsGenerator(fn_specs)
        iterator = range(config['n_samples'])
        n_total_tasks = config['n_samples']
    elif mode == 'validate':
        print_validate_summary(config)
        inputs = np.load(config['inputs'])
        iterator = inputs
        n_total_tasks = len(inputs)
    else:
        raise ValueError(
            f"Unknown mode: {mode}. The mode must be either 'train' or 'validate'."
        )

    # Main loop to generate samples
    samples = {}
    job_ids = []
    for idx, p in enumerate(iterator):

        # Generate sample or use provided input
        if mode == 'train':
            max_attempts = 100
            for attempt in range(max_attempts):
                hash, sample = sampler.generate(1, assign_hash=True)
                if sample.size > 0:
                    sample = sample.squeeze(0)
                    break
            if attempt == max_attempts:
                raise RuntimeError(
                    "Failed to generate a valid sample after 100 attempts."
                )
        elif mode == 'validate':
            hash, sample = str(idx), p

        # Store sample info
        samples[hash] = sample

        # Submit job while controling total number of running jobs
        if execution_mode == 'slurm':
            n_max = config['slurm'].get('max_parallel_jobs', 1)
            n_max = np.inf if n_max == -1 else n_max
            while True:
                if n_max > 0:
                    n_active = get_active_jobs(job_ids)
                    if n_active < n_max and idx < n_total_tasks:
                        id = dispatch_md_job(hash, sample, config)
                        job_ids.append(id)
                        break
                    else:
                        time.sleep(5)
                else:
                    break

        elif execution_mode == 'local':
            dispatch_md_job(hash, sample, fn_config)

        else:
            raise NotImplementedError(
                f"Execution manager '{execution_mode}' is not supported."
                "Please use 'slurm' or 'local'."
            )

        print(
            f"> Running MD: {idx+1}/{n_total_tasks} "
            f"({((idx + 1) / n_total_tasks * 100):.0f}%)\r",
            end='', flush=True
        )

    # Wait for jobs to finish if using Slurm
    if execution_mode == 'slurm':
        if n_max > 0:
            control_jobs(job_ids)

    print(f'> Running MD: {n_total_tasks}/{n_total_tasks} (100%) | Done.'.ljust(50))

    # Cleanup
    clean_up_train_dir(
        samples,
        Path(config['data_dir']),
        compress=config['compress'],
        remove=config['cleanup']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Generate training/validation set for MD simulations.'
        )
    )
    parser.add_argument('mode', help='Mode to run the workflow [train|validate].')
    parser.add_argument('fn_config', help='Path to the config file [YAML].')
    args = parser.parse_args()
    main(args.mode, args.fn_config)
