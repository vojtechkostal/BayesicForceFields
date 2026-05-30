import os
import shutil
import subprocess
from pathlib import Path
from typing import Union

import numpy as np

from ...domain.specs import ChargeConstraint, Specs
from ...io.colvars import write_mdp_with_colvars
from ...io.commands import build_command
from ...io.mdp import get_n_frames_target
from ...io.plumed import ensure_plumed_kernel
from ...io.utils import save_yaml
from ...topology import TopologyModifier
from .config import MDJobConfig

PathLike = Union[str, Path]


def check_gmx_available(gmx_cmd: str = "gmx") -> None:
    """Check if the configured GROMACS command can be executed."""
    try:
        subprocess.run(
            build_command(gmx_cmd, '--version'),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"GROMACS command {gmx_cmd!r} is not available.\n"
            "Make sure the executable is on PATH in the job environment, "
            "or set 'gmx_cmd' to the correct command."
        ) from exc


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
    fn_trj: str | Path,
    fn_mdp: str | Path,
    n_steps: int,
) -> bool:
    """Check if the trajectory reached the expected number of saved frames."""
    fn_trj = Path(fn_trj)
    if not fn_trj.exists():
        return False

    _, stride = get_n_frames_target(fn_mdp)
    if stride in (None, 0):
        return False

    expected_frames = max(1, n_steps // stride + 1)
    try:
        from MDAnalysis.coordinates.XTC import XTCReader

        with XTCReader(str(fn_trj)) as reader:
            n_frames = reader.n_frames
    except Exception:
        return False

    return n_frames >= expected_frames


def modify_topology(
    fn_topol: str,
    specs: Union[str, dict, Specs],
    params: Union[list, np.ndarray],
    implicit: bool,
    fn_out: str
) -> None:

    if isinstance(specs, (str, Path, dict)):
        specs = Specs(specs)

    if implicit and not ChargeConstraint(specs)(params).all():
        raise ValueError(
            "Explicit parameter values or reconstructed implicit charges violate "
            "the configured bounds."
        )
    values = (
        specs.with_implicit_charges(params).reshape(-1)
        if implicit
        else np.asarray(params, dtype=float).reshape(-1)
    )
    params_dict = specs.parameter_dict(values)

    top_modifier = TopologyModifier(fn_topol)
    top_modifier.apply_parameters(params_dict)
    for constraint in specs.charge_constraints:
        for group in top_modifier.selected_groups(
            constraint.selection,
            constraint.scope,
        ):
            actual = sum(top_modifier.atoms[index].charge for index in group)
            if not np.isclose(actual, constraint.target, atol=1e-8):
                raise ValueError(
                    f"Applied charge constraint {constraint.selection!r} has "
                    f"charge {actual}, expected {constraint.target}."
                )

    if fn_out:
        top_modifier.write(fn_out)
    return top_modifier


def main(fn_config: PathLike) -> None:
    config = MDJobConfig.load(fn_config)
    if config.fn_specs is None:
        raise ValueError("MD job configuration requires 'fn_specs'.")
    sample_id = config.sample_id
    params = config.params
    campaign_dir = config.campaign_dir
    specs = Specs(config.fn_specs)
    implicit = True
    gmx_cmd = config.gmx_cmd
    run = config.run
    check_gmx_available(gmx_cmd)
    if any(
        system.bias.kind == "plumed"
        for system in config.systems
    ):
        ensure_plumed_kernel()

    # Determine the run directory
    job_scheduler = config.job_scheduler
    run_dir = campaign_dir if job_scheduler == "local" else Path("./").resolve()

    fn_log = campaign_dir / f"gmx-{sample_id}.log"
    success = []
    outputs: list[dict[str, str | None]] = []
    status = "failed"
    try:
        with open(fn_log, 'a+') as log:
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
                    check_success(f'{deffnm}.xtc', fn_prod_mdp, steps)
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

        if np.all(success):
            status = "completed"
            if job_scheduler != 'local':
                for ext in config.store:
                    for file in run_dir.glob("*." + ext):
                        shutil.copy(file, campaign_dir / file.name)
            else:
                _prune_auxiliary_outputs(run_dir, config.store)
        else:
            for system_index in range(len(config.systems)):
                for file in _sample_output_paths(run_dir, sample_id, system_index):
                    file.unlink(missing_ok=True)

    except Exception:
        for system_index in range(len(config.systems)):
            for file in _sample_output_paths(run_dir, sample_id, system_index):
                file.unlink(missing_ok=True)
        save_yaml(
            {
                "sample_id": sample_id,
                "status": status,
                "outputs": outputs,
            },
            campaign_dir / f"result-{sample_id}.yaml",
        )
        raise

    save_yaml(
        {
            "sample_id": sample_id,
            "status": status,
            "outputs": outputs,
        },
        campaign_dir / f"result-{sample_id}.yaml",
    )
