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
    system_id: str,
) -> list[Path]:
    """Return all files created for one sample/system pair."""
    system_dir = run_dir / "samples" / sample_id / system_id
    if not system_dir.exists():
        return []
    return sorted(path for path in system_dir.iterdir() if path.is_file())


def _collect_working_outputs(
    working_dir: Path,
    system_run_dir: Path,
    state_before_run: dict[Path, tuple[int, int]],
    store: tuple[str, ...],
    cleanup: bool,
) -> list[Path]:
    """Move files created in the shared cwd into the producing system."""
    keep_suffixes = {"." + extension.lstrip(".") for extension in store}
    operational_names = {
        "config.yaml",
        "gmx.log",
        "result.yaml",
        "run.out",
        "run.sh",
    }
    collected: list[Path] = []
    for path in working_dir.iterdir():
        if (
            not path.is_file()
            or path.name in operational_names
            or (cleanup and path.suffix not in keep_suffixes)
        ):
            continue
        stat = path.stat()
        current_state = (stat.st_mtime_ns, stat.st_size)
        if state_before_run.get(path) == current_state:
            continue
        destination = system_run_dir / path.name
        if destination.exists():
            destination.unlink()
        shutil.move(path, destination)
        collected.append(destination)
    return collected


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
    except (EOFError, OSError, ValueError):
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

    constraint = ChargeConstraint(specs)
    if implicit and not constraint(params).all():
        raise ValueError(
            "Explicit parameter values or reconstructed implicit charges violate "
            "the configured bounds: "
            + constraint.describe_violations(params)
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

    working_dir = campaign_dir / "outputs" / sample_id
    working_dir.mkdir(parents=True, exist_ok=True)

    fn_log = working_dir / "gmx.log"
    success = []
    outputs: list[dict[str, object]] = []
    status = "failed"
    try:
        with open(fn_log, 'a+') as log:
            for system in config.systems:
                em = system.mdp_em_path
                prod = system.mdp_production_path
                coord = system.coordinates_path
                top = system.topology_path
                ndx = system.index_path
                steps = system.n_steps
                bias = system.bias

                # Define the output file names
                system_run_dir = (
                    campaign_dir / "samples" / sample_id / system.system_id
                )
                system_run_dir.mkdir(parents=True, exist_ok=True)
                deffnm = system_run_dir / "production"
                fn_tpr = deffnm.with_suffix(".tpr")
                fn_coord_prod = coord

                # Create topology with new parameters
                fn_top_new = system_run_dir / "topology.top"
                modify_topology(top, specs, params, implicit, fn_top_new)
                fn_prod_mdp = prod
                mdrun_extra_args: list[str] = []
                run_env = None
                if bias.kind == "colvars" and bias.input_file is not None:
                    fn_bias_local = system_run_dir / Path(bias.input_file).name
                    if Path(bias.input_file).resolve() != fn_bias_local.resolve():
                        shutil.copy2(bias.input_file, fn_bias_local)
                    fn_prod_mdp = system_run_dir / "production-colvars.mdp"
                    write_mdp_with_colvars(
                        prod,
                        fn_bias_local,
                        fn_prod_mdp,
                        working_dir=working_dir,
                    )
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
                    deffnm_em = system_run_dir / "em"
                    fn_tpr_em = deffnm_em.with_suffix(".tpr")
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
                        cwd=working_dir, stdout=log, stderr=log, check=True
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
                        cwd=working_dir,
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
                    cwd=working_dir, stdout=log, stderr=log, check=True
                )

                working_state = {
                    path: (path.stat().st_mtime_ns, path.stat().st_size)
                    for path in working_dir.iterdir()
                    if path.is_file()
                }
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
                    cwd=working_dir,
                    stdout=log,
                    stderr=log,
                    check=True,
                    env=run_env,
                )
                _collect_working_outputs(
                    working_dir,
                    system_run_dir,
                    working_state,
                    config.store,
                    config.cleanup,
                )

                # Check if the simulation finished aka has the expected number of frames
                success.append(
                    check_success(f'{deffnm}.xtc', fn_prod_mdp, steps)
                )
                trajectory_name = (
                    str(deffnm.with_suffix(".xtc").relative_to(campaign_dir))
                    if "xtc" in config.store
                    else None
                )
                stored_inputs: dict[str, str | list[str]] = {}
                for extension in config.store:
                    role = extension.lstrip(".")
                    if role == "xtc":
                        continue
                    matches = sorted(
                        path
                        for path in system_run_dir.iterdir()
                        if path.is_file() and path.suffix == f".{role}"
                    )
                    relative_paths = [
                        str(path.relative_to(campaign_dir)) for path in matches
                    ]
                    if len(relative_paths) == 1:
                        stored_inputs[role] = relative_paths[0]
                    elif relative_paths:
                        stored_inputs[role] = relative_paths
                outputs.append(
                    {
                        "system_id": system.system_id,
                        "trajectory": trajectory_name,
                        "inputs": stored_inputs,
                    }
                )

        if np.all(success):
            status = "completed"
    finally:
        if status != "completed" and config.cleanup:
            for system in config.systems:
                for file in _sample_output_paths(
                    campaign_dir, sample_id, system.system_id
                ):
                    file.unlink(missing_ok=True)
        save_yaml(
            {
                "sample_id": sample_id,
                "status": status,
                "outputs": outputs,
            },
            working_dir / "result.yaml",
        )
