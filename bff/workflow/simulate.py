import argparse
import subprocess
import shutil

import numpy as np

from pathlib import Path

from ..io.mdp import get_n_frames_target
from ..structures import Specs
from ..topology import TopologyParser
from ..io.utils import load_yaml


def check_success(fn_trj: str, fn_mdp: str, n_target: int) -> bool:
    """Check if the trajectory has reached the target number of frames."""
    output = subprocess.run(
        ['gmx', 'check', '-f', fn_trj], capture_output=True, text=True)
    
    for line in output.stderr.splitlines():
        if line.startswith("Step"):
            n = int(line.split()[1])
            break

    n_target_mdp, stride = get_n_frames_target(fn_mdp)
    n_target = n_target_mdp if n_target == -2 else (n_target // stride)

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


def main(fn_config: str) -> None:

    # Parse the command-line arguments load general configuration
    config = load_yaml(fn_config)
    hash = config['hash']
    params = config['params']
    data_dir = Path(config['data_dir'])
    specs = Specs(config['fn_specs']) if config.get('fn_specs') else Specs(config)
    implicit = config.get('implicit', True)
    gmx_cmd = config.get('gmx_cmd', 'gmx')

    # Unpack Gromacs configuration
    job_keys = ['fn_mdp_em', 'fn_mdp_prod', 'fn_coordinates', 'fn_ndx', 'n_steps']
    fn_topol = config['gromacs']['fn_topol']
    for fn_group in job_keys:
        if isinstance(config['gromacs'][fn_group], str):
            config['gromacs'][fn_group] = [config['gromacs'][fn_group]] * len(fn_topol)
            
    fn_mdp_em = config['gromacs']['fn_mdp_em']
    fn_mdp_prod = config['gromacs']['fn_mdp_prod']
    fn_coord = config['gromacs']['fn_coordinates']
    fn_ndx = config['gromacs']['fn_ndx']
    n_steps = config['gromacs']['n_steps']

    # Determine the run directory
    job_scheduler = config.get('job_scheduler', 'local')
    run_dir = data_dir if job_scheduler == 'local' else Path('./').resolve()

    fn_log = str(data_dir / 'gmx.log')
    md_specs = [fn_mdp_em, fn_mdp_prod, fn_coord, fn_topol, fn_ndx, n_steps]
    success = []
    with open(fn_log, 'a+') as log:
        for i, (em, prod, c, t, ndx, n) in enumerate(zip(md_specs)):

            # Define the output file names
            deffnm = run_dir / f"md-{hash}-{i}"
            fn_tpr = run_dir / f"{deffnm}.tpr"

            # Create topology with new parameters
            fn_top_new = str(run_dir / f"topol-{hash}-{i:03d}.top")
            _ = modify_topology(t, specs, params, implicit, fn_top_new)

            # Minimize energy
            if em:
                subprocess.run(
                    [gmx_cmd, 'grompp', '-f', em, '-c', c, '-p',
                     fn_top_new, '-n', ndx, '-o', fn_tpr, '-maxwarn', '2'],
                    cwd=run_dir, stdout=log, stderr=log
                )

                subprocess.run(
                    [gmx_cmd, 'mdrun', '-s', fn_tpr, '-deffnm', deffnm],
                    cwd=run_dir, stdout=log, stderr=log
                )
            else:
                deffnm.with_suffix('.gro').write_text(c.read_text())

            # Run production MD
            subprocess.run(
                [gmx_cmd, 'grompp', '-f', prod, '-c', deffnm.with_suffix('.gro'),
                 '-p', fn_top_new, '-n', ndx, '-o', fn_tpr, '-maxwarn', '2'],
                cwd=run_dir, stdout=log, stderr=log
            )

            subprocess.run(
                [gmx_cmd, 'mdrun', '-s', fn_tpr, '-deffnm', deffnm,
                 '-nsteps', str(n), '-dlb', 'yes', '-ntmpi', '1'],
                cwd=run_dir, stdout=log, stderr=log
            )

            # Check if the simulation finished aka has the expected number of frames
            success.append(check_success(f'{deffnm}.xtc', prod, n))

    if np.all(success):
        if job_scheduler != 'local':
            pattern = "*" if config.get('store', False) else "*.xtc"
            for file in run_dir.glob(pattern):
                shutil.copy(file, data_dir)
    else:
        for file in run_dir.glob(f"*.xtc"):
            try:
                Path(file).unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gromacs simulations.")
    parser.add_argument("-f", "--fn_config", type=str,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.fn_config)
