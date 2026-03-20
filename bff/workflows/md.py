import argparse
import os
import subprocess
import shutil
from typing import Union

import numpy as np

from pathlib import Path

from ..io.colvars import write_mdp_with_colvars
from ..io.mdp import get_n_frames_target
from ..io.plumed import ensure_plumed_kernel
from ..domain.bias import BiasSpec
from ..domain.specs import Specs
from ..topology import TopologyModifier
from .configs import MDJobConfig


PathLike = Union[str, Path]


def check_success(
    fn_trj: str,
    fn_mdp: str,
    n_target: int,
    gmx_cmd: str = "gmx",
) -> bool:
    """Check if the trajectory has reached the target number of frames."""
    output = subprocess.run(
        [gmx_cmd, 'check', '-f', fn_trj],
        capture_output=True,
        text=True,
        check=True,
    )

    n = None
    for line in output.stderr.splitlines():
        if line.startswith("Step"):
            n = int(line.split()[1])
            break
    if n is None:
        raise RuntimeError(f"Could not determine trajectory length from {fn_trj}.")

    n_target_mdp, stride = get_n_frames_target(fn_mdp)
    n_target = n_target_mdp if n_target == -2 else (n_target // stride)

    return n >= n_target


def modify_topology(
    fn_topol: str,
    specs: Union[str, dict, Specs],
    params: Union[list, np.ndarray],
    implicit: bool,
    fn_out: str
) -> None:

    if isinstance(specs, (str, dict)):
        specs = Specs(specs)

    if implicit:
        constraint_charge = specs.constraint_charge
        param_names = specs.bounds.without(specs.implicit_param).names
    else:
        constraint_charge = None
        param_names = specs.bounds.names

    params_dict = dict(zip(param_names, params))

    top_modifier = TopologyModifier(fn_topol, specs.mol_resname, specs.implicit_atoms)
    top_modifier.update_params(params_dict, constraint_charge)

    if fn_out:
        top_modifier.write(fn_out)
    return top_modifier


def main(fn_config: PathLike) -> None:

    config = MDJobConfig.load(fn_config)
    if config.fn_specs is None:
        raise ValueError("MD job configuration requires 'fn_specs'.")
    hash = config.hash
    params = config.params
    data_dir = config.data_dir
    specs = Specs(config.fn_specs)
    implicit = True
    gmx_cmd = config.gmx_cmd
    run = config.run
    if any(
        fn_bias is not None and BiasSpec.load(fn_bias).kind == "plumed"
        for fn_bias in config.gromacs.fn_bias
    ):
        ensure_plumed_kernel()

    # Determine the run directory
    job_scheduler = config.job_scheduler
    run_dir = data_dir if job_scheduler == 'local' else Path('./').resolve()

    fn_log = data_dir / 'gmx.log'
    md_specs = zip(
        config.gromacs.fn_mdp_em,
        config.gromacs.fn_mdp_prod,
        config.gromacs.fn_coordinates,
        config.gromacs.fn_topol,
        config.gromacs.fn_ndx,
        config.gromacs.n_steps,
        config.gromacs.fn_bias,
    )
    success = []
    with open(fn_log, 'a+') as log:
        for i, md_specs_ in enumerate(md_specs):

            em, prod, coord, top, ndx, steps, fn_bias = md_specs_
            bias = BiasSpec() if fn_bias is None else BiasSpec.load(fn_bias)

            # Define the output file names
            deffnm = run_dir / f"md-{hash}-{i}"
            fn_tpr = run_dir / f"{deffnm}.tpr"

            # Create topology with new parameters
            fn_top_new = run_dir / f"md-{hash}-{i:03d}.top"
            _ = modify_topology(top, specs, params, implicit, fn_top_new)
            fn_prod_mdp = prod
            mdrun_extra_args: list[str] = []
            run_env = None
            if bias.kind == "colvars" and bias.input_file is not None:
                fn_prod_mdp = run_dir / f"md-{hash}-{i:03d}-colvars.mdp"
                write_mdp_with_colvars(prod, bias.input_file, fn_prod_mdp)
            elif bias.kind == "plumed" and bias.input_file is not None:
                kernel = ensure_plumed_kernel()
                run_env = dict(os.environ)
                run_env.setdefault("PLUMED_KERNEL", str(kernel))
                mdrun_extra_args.extend(["-plumed", str(bias.input_file)])

            # Skip running the simulation if specified
            if not run:
                success.append(True)
                continue

            # Minimize energy
            if em:
                subprocess.run(
                    [gmx_cmd, 'grompp', '-f', em, '-c', coord, '-p',
                     fn_top_new, '-n', ndx, '-o', fn_tpr, '-maxwarn', '2'],
                    cwd=run_dir, stdout=log, stderr=log, check=True
                )

                subprocess.run(
                    [gmx_cmd, 'mdrun', '-s', fn_tpr, '-deffnm', deffnm],
                    cwd=run_dir, stdout=log, stderr=log, check=True
                )
            else:
                deffnm.with_suffix('.gro').write_text(coord.read_text())

            # Run production MD
            subprocess.run(
                [gmx_cmd, 'grompp', '-f', fn_prod_mdp,
                 '-c', deffnm.with_suffix('.gro'),
                 '-p', fn_top_new, '-n', ndx, '-o', fn_tpr, '-maxwarn', '2'],
                cwd=run_dir, stdout=log, stderr=log, check=True
            )

            subprocess.run(
                [gmx_cmd, 'mdrun', '-deffnm', deffnm,
                 '-nsteps', str(steps), '-dlb', 'yes', '-ntmpi', '1', *mdrun_extra_args],
                cwd=run_dir, stdout=log, stderr=log, check=True, env=run_env
            )

            # Check if the simulation finished aka has the expected number of frames
            success.append(check_success(f'{deffnm}.xtc', fn_prod_mdp, steps, gmx_cmd))

    if np.all(success):
        if job_scheduler != 'local':
            patterns = ["*." + ext for ext in config.store]
            files = list(sum((list(run_dir.glob(p)) for p in patterns), []))
            for file in files:
                shutil.copy(file, data_dir / file.name)
    else:
        for file in run_dir.glob("*.xtc"):
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
