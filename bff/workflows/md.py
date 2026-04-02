import argparse
import os
import subprocess
import shutil
from typing import Union

import numpy as np

from pathlib import Path

from ..io.colvars import write_mdp_with_colvars
from ..io.commands import build_command
from ..io.mdp import get_n_frames_target
from ..io.plumed import ensure_plumed_kernel
from ..domain.specs import Specs
from ..io.utils import save_yaml
from ..topology import TopologyModifier
from .configs import MDJobConfig


PathLike = Union[str, Path]


def _sample_output_paths(
    run_dir: Path,
    sample_id: str,
    system_index: int,
) -> list[Path]:
    """Return all files created for one sample/system pair."""
    files: set[Path] = set()
    for pattern in (
        f"md-{sample_id}-{system_index}*",
        f"md-{sample_id}-{system_index:03d}*",
    ):
        files.update(path for path in run_dir.glob(pattern) if path.is_file())
    return sorted(files)


def _prune_unstored_outputs(
    files: list[Path],
    store: tuple[str, ...],
) -> None:
    """Remove generated files whose suffix is not requested in ``store``."""
    keep_suffixes = {"." + ext.lstrip(".") for ext in store}
    for file in files:
        if file.suffix in keep_suffixes:
            continue
        file.unlink(missing_ok=True)


def _prune_auxiliary_outputs(run_dir: Path, store: tuple[str, ...]) -> None:
    """Remove shared GROMACS or PLUMED aux files that are not requested."""
    keep_suffixes = {"." + ext.lstrip(".") for ext in store}
    for pattern in ("mdout.mdp", "PLUMED.OUT", "bck*.PLUMED.OUT"):
        for file in run_dir.glob(pattern):
            if file.suffix in keep_suffixes:
                continue
            file.unlink(missing_ok=True)


def check_success(
    fn_trj: str,
    fn_mdp: str,
    n_target: int,
    gmx_cmd: str = "gmx",
) -> bool:
    """Check if the trajectory has reached the target number of frames."""
    output = subprocess.run(
        build_command(gmx_cmd, 'check', '-f', fn_trj),
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

    if isinstance(specs, (str, Path, dict)):
        specs = Specs(specs)

    constraint_charge = specs.constraint_charge if implicit else None
    params_dict = specs.parameter_dict(params, explicit_only=implicit)

    top_modifier = TopologyModifier(
        fn_topol,
        specs.mol_resname,
        specs.implicit_atoms,
    )
    top_modifier.apply_parameters(params_dict, constraint_charge=constraint_charge)

    if fn_out:
        top_modifier.write(fn_out)
    return top_modifier


def main(fn_config: PathLike) -> None:
    config = MDJobConfig.load(fn_config)
    if config.fn_specs is None:
        raise ValueError("MD job configuration requires 'fn_specs'.")
    sample_id = config.sample_id
    params = config.params
    trainset_dir = config.trainset_dir
    specs = Specs(config.fn_specs)
    implicit = True
    gmx_cmd = config.gmx_cmd
    run = config.run
    if any(
        system.bias.kind == "plumed"
        for system in config.systems
    ):
        ensure_plumed_kernel()

    # Determine the run directory
    job_scheduler = config.job_scheduler
    run_dir = trainset_dir if job_scheduler == "local" else Path("./").resolve()

    fn_log = trainset_dir / "gmx.log"
    success = []
    outputs: list[dict[str, str | None]] = []
    with open(fn_log, 'a+') as log:
        try:
            for i, system in enumerate(config.systems):
                em = system.fn_mdp_em
                prod = system.fn_mdp_prod
                coord = system.fn_coordinates
                top = system.fn_topol
                ndx = system.fn_ndx
                steps = system.n_steps
                bias = system.bias

                # Define the output file names
                deffnm = run_dir / f"md-{sample_id}-{i}"
                fn_tpr = run_dir / f"{deffnm}.tpr"
                fn_coord_prod = coord

                # Create topology with new parameters
                fn_top_new = run_dir / f"md-{sample_id}-{i:03d}.top"
                _ = modify_topology(top, specs, params, implicit, fn_top_new)
                fn_prod_mdp = prod
                mdrun_extra_args: list[str] = []
                run_env = None
                if bias.kind == "colvars" and bias.input_file is not None:
                    fn_bias_local = run_dir / Path(bias.input_file).name
                    if Path(bias.input_file).resolve() != fn_bias_local.resolve():
                        shutil.copy2(bias.input_file, fn_bias_local)
                    fn_prod_mdp = run_dir / f"md-{sample_id}-{i:03d}-colvars.mdp"
                    write_mdp_with_colvars(prod, fn_bias_local, fn_prod_mdp)
                elif bias.kind == "plumed" and bias.input_file is not None:
                    kernel = ensure_plumed_kernel()
                    run_env = dict(os.environ)
                    run_env.setdefault("PLUMED_KERNEL", str(kernel))
                    mdrun_extra_args.extend(["-plumed", str(bias.input_file)])

                # Skip running the simulation if specified
                if not run:
                    success.append(True)
                    outputs.append(
                        {
                            "system_id": system.system_id,
                            "trajectory": None,
                        }
                    )
                    continue

                # Minimize energy
                if em:
                    deffnm_em = run_dir / f"md-{sample_id}-{i}-em"
                    fn_tpr_em = run_dir / f"{deffnm_em}.tpr"
                    subprocess.run(
                        build_command(
                            gmx_cmd,
                            'grompp',
                            '-f',
                            em,
                            '-c',
                            coord,
                            '-p',
                            fn_top_new,
                            '-n',
                            ndx,
                            '-o',
                            fn_tpr_em,
                            '-maxwarn',
                            '2',
                        ),
                        cwd=run_dir, stdout=log, stderr=log, check=True
                    )

                    subprocess.run(
                        build_command(
                            gmx_cmd,
                            'mdrun',
                            '-s',
                            fn_tpr_em,
                            '-deffnm',
                            deffnm_em,
                        ),
                        cwd=run_dir,
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
                    fn_coord_prod = deffnm_em.with_suffix(".gro")
                else:
                    deffnm.with_suffix('.gro').write_text(coord.read_text())
                    fn_coord_prod = deffnm.with_suffix(".gro")

                # Run production MD
                subprocess.run(
                    build_command(
                        gmx_cmd,
                        'grompp',
                        '-f',
                        fn_prod_mdp,
                        '-c',
                        fn_coord_prod,
                        '-p',
                        fn_top_new,
                        '-n',
                        ndx,
                        '-o',
                        fn_tpr,
                        '-maxwarn',
                        '2',
                    ),
                    cwd=run_dir, stdout=log, stderr=log, check=True
                )

                subprocess.run(
                    build_command(
                        gmx_cmd,
                        'mdrun',
                        '-deffnm',
                        deffnm,
                        '-nsteps',
                        str(steps),
                        '-dlb',
                        'yes',
                        *mdrun_extra_args,
                    ),
                    cwd=run_dir, stdout=log, stderr=log, check=True, env=run_env
                )

                # Check if the simulation finished aka has the expected number of frames
                success.append(
                    check_success(f'{deffnm}.xtc', fn_prod_mdp, steps, gmx_cmd)
                )
                generated_files = _sample_output_paths(run_dir, sample_id, i)
                if job_scheduler == "local":
                    _prune_unstored_outputs(generated_files, config.store)
                trajectory_name = (
                    f"{deffnm.name}.xtc" if "xtc" in config.store else None
                )
                outputs.append(
                    {
                        "system_id": system.system_id,
                        "trajectory": trajectory_name,
                    }
                )
        except Exception:
            for system_index in range(len(config.systems)):
                for file in _sample_output_paths(run_dir, sample_id, system_index):
                    file.unlink(missing_ok=True)
            save_yaml(
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "outputs": outputs,
                },
                trainset_dir / f"result-{sample_id}.yaml",
            )
            raise

    if np.all(success):
        if job_scheduler != 'local':
            patterns = ["*." + ext for ext in config.store]
            files = list(sum((list(run_dir.glob(p)) for p in patterns), []))
            for file in files:
                shutil.copy(file, trainset_dir / file.name)
        else:
            _prune_auxiliary_outputs(run_dir, config.store)
        save_yaml(
            {
                "sample_id": sample_id,
                "status": "completed",
                "outputs": outputs,
            },
            trainset_dir / f"result-{sample_id}.yaml",
        )
    else:
        for system_index in range(len(config.systems)):
            for file in _sample_output_paths(run_dir, sample_id, system_index):
                file.unlink(missing_ok=True)
        save_yaml(
            {
                "sample_id": sample_id,
                "status": "failed",
                "outputs": outputs,
            },
            trainset_dir / f"result-{sample_id}.yaml",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gromacs simulations.")
    parser.add_argument("-f", "--fn_config", type=str,
                        help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.fn_config)
