import argparse
import os
import subprocess

import numpy as np

from pathlib import Path

from amo.io.mdp import get_n_frames_target
from amo.structures import Specs
from amo.topology import TopologyParser
from amo.io.utils import load_yaml


def check_success(fn_trj, fn_mdp, n_target):
    """Check if the trajectory has reached the target number of frames."""
    output = subprocess.run(
        ['gmx', 'check', '-f', fn_trj], capture_output=True, text=True)
    step_line = [line.split()[1]
                 for line in output.stderr.splitlines()
                 if line.startswith('Step')]
    n = int(step_line[0])

    n_target_mdp, stride = get_n_frames_target(fn_mdp)
    n_target = (n_target // stride) if n_target != -2 else n_target_mdp

    return n >= n_target


def modify_topology(
    fn_topol: str,
    specs: str | dict | Specs,
    params: list | np.ndarray,
    implicit: bool,
    fn_out: str
) -> None:

    if isinstance(specs, (str, dict)):
        specs = Specs(specs)

    if implicit:
        total_charge = specs.total_charge
        params_dict = dict(zip(specs.bounds_implicit.params, params))
    else:
        total_charge = None
        params_dict = dict(zip(specs.bounds_explicit.params, params))

    topol = TopologyParser(fn_topol)
    topol.select_mol(specs.mol_resname, specs.implicit_atomtype)
    topol.update_params(params_dict, total_charge)

    if fn_out:
        topol.write(fn_out)
    return topol


def main(fn_config):

    # Parse the command-line arguments and load specs
    config = load_yaml(fn_config)
    hash = config['hash']
    params = config['params']
    if config.get('fn_specs'):
        specs = Specs(config['fn_specs'])
    else:
        specs = Specs(config)

    # Run Gromacs simulation(s)
    success = []
    implicit = config.get('implicit', True)
    data_dir = Path(config['data_dir'])
    fn_mdp_em = config['gromacs']['fn_mdp_em']
    fn_mdp_prod = config['gromacs']['fn_mdp_prod']
    fn_coord = config['gromacs']['fn_coordinates']
    fn_topol = config['gromacs']['fn_topol']
    fn_ndx = config['gromacs']['fn_ndx']
    n_jobs = len(fn_mdp_prod)
    n_steps = config['gromacs'].get('n_steps', -2)
    n_steps = np.repeat(n_steps, n_jobs) if isinstance(
        n_steps, int) else n_steps
    fn_log = str(data_dir / 'gmx.log')
    with open(fn_log, 'a+') as log:
        for i, (em, prod, c, t, ndx, n) in enumerate(
                zip(fn_mdp_em, fn_mdp_prod, fn_coord, fn_topol, fn_ndx, n_steps)):

            run_dir = (
                Path('./').resolve()
                if config['execution_mode'] in ['slurm']
                else data_dir
            )
            deffnm = run_dir / f"md-{hash}-{i}"
            fn_tpr = run_dir / f"{deffnm}.tpr"

            # Create the topology file
            fn_top_new = str(run_dir / f"topol-{hash}-{i:03d}.top")
            _ = modify_topology(t, specs, params, implicit, fn_top_new)

            # Minimize energy
            if em:
                subprocess.run(
                    ['gmx', 'grompp', '-f', em, '-c', c, '-p',
                     fn_top_new, '-n', ndx, '-o', fn_tpr, '-maxwarn', '2'],
                    cwd=run_dir, stdout=log, stderr=log
                )

                subprocess.run(
                    ['gmx', 'mdrun', '-s', fn_tpr, '-deffnm', deffnm],
                    cwd=run_dir, stdout=log, stderr=log
                )
            else:
                deffnm.with_suffix('.gro').write_text(c.read_text())

            # Run production MD
            subprocess.run(
                ['gmx', 'grompp', '-f', prod, '-c', deffnm.with_suffix('.gro'),
                 '-p', fn_top_new, '-n', ndx, '-o', fn_tpr, '-maxwarn', '2'],
                cwd=run_dir, stdout=log, stderr=log
            )

            subprocess.run(
                ['gmx', 'mdrun', '-s', fn_tpr, '-deffnm', deffnm,
                 '-nsteps', str(n), '-dlb', 'yes', '-ntmpi', '1'],
                cwd=run_dir, stdout=log, stderr=log
            )

            # Check if the simulation finished aka has the expected number of frames
            success.append(check_success(f'{deffnm}.xtc', prod, n))

    if np.all(success):
        if config['execution_mode'] == 'local':
            pass
        else:
            os.system(f"cp *.xtc {data_dir}")
            if config.get('store', False):
                os.system(f"cp *.log *.edr *.tpr *.top {data_dir}")
    else:
        os.system(f"rm {deffnm}-*.xtc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gromacs simulations.")
    parser.add_argument("-f", "--fn_config", type=str,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.fn_config)
