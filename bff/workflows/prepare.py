import argparse
import os
import subprocess
import shutil
import warnings
from dataclasses import dataclass
import numpy as np
from typing import Union, Dict, List, Optional, Sequence

import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.selections.gromacs import SelectionWriter
from MDAnalysis import transformations as trans

from pathlib import Path

from gmxtopology import Topology
from ..domain.bias import BiasSpec
from ..io.commands import build_command
from ..topology import create_box
from ..io.colvars import (
    resolve_distance_bias_metadata as resolve_colvars_distance_bias_metadata,
    write_mdp_with_colvars,
)
from ..io.cp2k import (
    make_cp2k_input,
    make_cp2k_single_point_input,
    write_cp2k_md_slurm_script,
    write_cp2k_single_point_slurm_script,
)
from ..io.plumed import (
    ensure_plumed_kernel,
    resolve_distance_bias_metadata as resolve_plumed_distance_bias_metadata,
)
from ..io.logs import Logger
from .configs import PrepareConfig


PathLike = Union[str, Path]


@dataclass(slots=True)
class EquilibratedTopology:
    """Shared topology state reused across simulation windows."""

    fn_topol_processed: Path
    fn_coord: Path
    universe: mda.Universe
    box: np.ndarray
    maxwarn: int


def check_gmx_available(gmx_cmd: str = 'gmx') -> None:
    """Check if the 'gmx' command is available in the system PATH."""
    try:
        subprocess.run(
            build_command(gmx_cmd, '--version'),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "GROMACS is not available.\n"
            "Please make sure that your Gromacs executable is available "
            "from the command line via e.g. gmx command."
        )


def setup_directories(main_dir: PathLike = Path('./')) -> tuple[Path, Path, Path, Path]:
    """Create required directories for the preparation workflow."""
    main_dir = Path(main_dir).resolve()
    equilibration_dir = main_dir / "equilibration"
    training_dir = main_dir / "training"
    reference_dir = main_dir / "reference"
    reference_md_dir = reference_dir / "md"
    reference_sp_dir = reference_dir / "single-points"

    for directory in [
        equilibration_dir,
        training_dir,
        reference_dir,
        reference_md_dir,
        reference_sp_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return equilibration_dir, training_dir, reference_md_dir, reference_sp_dir


def make_ndx(universe: mda.Universe, selections: List[str], fn_out: str) -> None:
    with SelectionWriter(fn_out, mode='w') as ndx:
        ndx.write(universe.atoms, name='System')
        if selections is not None:
            for sel in selections:
                for atom in sel.split():
                    group = universe.select_atoms('name ' + atom)
                    ndx.write(group, name=atom)


def determine_maxwarn(topol: Topology) -> int:
    """Determine whether charged systems need relaxed ``grompp`` warnings."""
    total_charge = sum(atom.charge for atom in topol.atoms)
    return 1 if not np.isclose(total_charge, 0, atol=1e-4) else 0


def run_md(
    name: str,
    fn_mdp: PathLike,
    fn_topol: PathLike,
    fn_coord: PathLike,
    fn_ndx: Optional[PathLike],
    bias: BiasSpec | None = None,
    gmx_cmd: str = "gmx",
    n_steps: int = -2,
    maxwarn: int = 0,
    fn_log: PathLike = 'gmx.log'
) -> Path:

    fn_mdp_path = Path(fn_mdp).resolve()
    fn_topol = str(Path(fn_topol).resolve())
    fn_coord = str(Path(fn_coord).resolve())
    fn_ndx = str(Path(fn_ndx).resolve()) if fn_ndx else None
    fn_tpr = str(name) + '.tpr'
    fn_mdp_run = fn_mdp_path
    run_cwd = fn_mdp_path.parent
    run_env = None
    mdrun_cmd = [
        *build_command(
            gmx_cmd,
            'mdrun',
            '-deffnm',
            name,
            '-nsteps',
            str(n_steps),
        ),
    ]

    if bias is not None and bias.kind == "colvars" and bias.input_file is not None:
        fn_bias_local = Path(fn_mdp_path.parent) / Path(bias.input_file).name
        shutil.copy2(bias.input_file, fn_bias_local)
        fn_mdp_run = Path(f"{name}-colvars.mdp").resolve()
        run_cwd = fn_mdp_run.parent
        write_mdp_with_colvars(fn_mdp_path, fn_bias_local, fn_mdp_run)
    elif bias is not None and bias.kind == "plumed" and bias.input_file is not None:
        kernel = ensure_plumed_kernel()
        run_env = dict(os.environ)
        run_env.setdefault("PLUMED_KERNEL", str(kernel))
        mdrun_cmd.extend(["-plumed", str(bias.input_file)])

    maxwarn = str(maxwarn)

    grompp_cmd = [
        *build_command(
            gmx_cmd,
            'grompp',
            '-f',
            str(fn_mdp_run),
            '-c',
            fn_coord,
            '-p',
            fn_topol,
            '-o',
            fn_tpr,
            '-maxwarn',
            maxwarn,
        ),
    ]
    if fn_ndx:
        grompp_cmd.extend(['-n', fn_ndx])

    with open(fn_log, "a") as f:
        subprocess.run(grompp_cmd, stdout=f, stderr=f, check=True, cwd=run_cwd)
        subprocess.run(
            mdrun_cmd,
            stdout=f,
            stderr=f,
            check=True,
            env=run_env,
            cwd=run_cwd,
        )

    return fn_mdp_run


def get_average_box(
    universe: mda.Universe,
    start: int = 0,
    stop: int = None,
    step: int = 1
) -> np.ndarray:
    """Compute the average box size over a trajectory."""
    sl = slice(start, stop or universe.trajectory.n_frames, step)

    box = np.zeros((len(universe.trajectory[sl]), 6))
    for i, ts in enumerate(universe.trajectory[sl]):
        box[i] = ts.dimensions
    box_avg = np.round(np.mean(box, axis=0), 4)

    return box_avg


def create_bias_window(
    universe: mda.Universe,
    pair_labels: List[str],
    target_distances: List[float],
    box: np.ndarray,
    fn_out: PathLike,
) -> None:
    """Write the frame closest to the requested bias-window distances.

    Parameters
    ----------
    universe
        Equilibrated universe from which the best-matching frame is selected.
    pair_labels
        Atom-name pairs whose distances define the bias window.
    target_distances
        Target distances in nm corresponding to ``pair_labels``.
    box
        Unit-cell dimensions assigned to the output frame.
    fn_out
        Output coordinate file.
    """
    target_distances = np.array(target_distances, dtype=float)
    ags = [[universe.select_atoms('name ' + x)
            for x in a.split()] for a in pair_labels]
    distances = np.array([
        [distance_array(ag_pair[0], ag_pair[1], box=ts.dimensions)[0, 0]
         for ag_pair in ags]
        for ts in universe.trajectory
    ])

    target_distances_scaled = np.array(target_distances) * 10
    idx = int(np.argmin(np.sum(np.abs(distances - target_distances_scaled), axis=1)))
    with mda.Writer(str(fn_out), 'w') as w:
        ts = universe.trajectory[idx]
        ts.dimensions = box
        w.write(universe.atoms)


def strip_topol(
    fn_topol: PathLike,
    fn_coords: PathLike,
    fn_out_topol: PathLike,
    *fn_out_coords: PathLike
) -> None:
    top = Topology(fn_topol)
    u = mda.Universe(fn_topol, fn_coords, topology_format='ITP')

    # Remove virtual sites from topology
    for mol, _ in top.molecules.values():
        mol.remove_vsites()
    top.write(fn_out_topol, overwrite=True)

    # Remove virtual sites from coordinates
    atoms = u.select_atoms('not mass -1 to 0.5')
    for fn_out in fn_out_coords:
        atoms.write(fn_out, frames=u.trajectory[[-1]])


def prepare_bias_file(
    bias: BiasSpec,
    fn_out: PathLike,
) -> Path | None:
    """Copy a user-provided bias file for one simulation window.

    Parameters
    ----------
    bias
        Bias specification for the current window.
    fn_out
        Output bias file path.

    Returns
    -------
    pathlib.Path or None
        Path to the prepared bias file, or ``None`` for unbiased systems.
    """
    if not bias.is_biased or bias.input_file is None:
        return None

    fn_out = Path(fn_out).resolve()
    shutil.copy2(bias.input_file, fn_out)
    return fn_out


def localize_bias_spec(
    bias: BiasSpec,
    fn_input: Path | None,
) -> BiasSpec:
    """Return a bias spec that points to a prepared local bias input file."""
    if fn_input is None or not bias.is_biased:
        return bias
    if bias.kind == "colvars":
        return BiasSpec(kind="colvars", colvars_file=fn_input)
    if bias.kind == "plumed":
        return BiasSpec(kind="plumed", plumed_file=fn_input)
    return bias


def resolve_bias_window_definition(
    bias: BiasSpec,
    fn_system: PathLike,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Resolve distance-based window metadata from a simple bias file.

    Returns empty tuples for unbiased systems or for bias files that do not
    expose simple distance centers.
    """
    if not bias.is_biased or bias.input_file is None:
        return (), ()

    if bias.kind == "colvars":
        return resolve_colvars_distance_bias_metadata(bias.input_file, fn_system)
    if bias.kind == "plumed":
        return resolve_plumed_distance_bias_metadata(bias.input_file, fn_system)
    return (), ()


def window_name(window_index: int) -> str:
    """Return the canonical zero-padded window name."""
    return f"window-{window_index:03d}"


def topology_name(topology_index: int) -> str:
    """Return the canonical zero-padded topology name."""
    return f"topology-{topology_index:03d}"


def sample_snapshot_indices(n_frames: int, n_snapshots: int) -> np.ndarray:
    """Choose evenly spaced snapshot indices from a trajectory."""
    if n_frames <= 0:
        raise ValueError("Cannot sample snapshots from an empty trajectory.")
    count = min(int(n_snapshots), int(n_frames))
    return np.unique(np.linspace(0, n_frames - 1, num=count, dtype=int))


def write_snapshot_xyz_files(
    universe: mda.Universe,
    *,
    snapshots_dir: Path,
    n_snapshots: int,
) -> list[Path]:
    """Write evenly spaced XYZ snapshots from a trajectory."""
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    indices = sample_snapshot_indices(universe.trajectory.n_frames, n_snapshots)
    atoms = universe.select_atoms("not mass -1 to 0.5")
    written: list[Path] = []

    for output_index, frame_index in enumerate(indices):
        fn_snapshot = snapshots_dir / f"snapshot-{output_index:04d}.xyz"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Reader has no dt information, set to 1.0 ps.*",
                category=UserWarning,
            )
            atoms.write(fn_snapshot, frames=universe.trajectory[[frame_index]])
        written.append(fn_snapshot)

    return written


def write_window_coordinates(
    universe: mda.Universe,
    *,
    box: np.ndarray,
    fn_out: PathLike,
    pair_labels: Sequence[str],
    target_distances: Sequence[float],
) -> None:
    """Write coordinates for one NVT window."""
    if pair_labels and target_distances:
        create_bias_window(
            universe,
            list(pair_labels),
            list(target_distances),
            box,
            fn_out,
        )
        return

    with mda.Writer(fn_out, "w") as writer:
        ts = universe.trajectory[-1]
        ts.dimensions = box
        writer.write(universe.atoms)


def prepare_equilibrated_topology(
    *,
    topol_index: int,
    fn_topol: Path,
    fn_mol: Path,
    box: list[float] | None,
    fn_mdp_em: Path,
    fn_mdp_npt: Path,
    nsteps_npt: int,
    equilibration_dir: Path,
    gmx_cmd: str,
    fn_gmx_log: Path,
    logger: Logger,
) -> EquilibratedTopology:
    """Create, minimize, and optionally NpT-equilibrate one unique topology."""
    topology_label = topology_name(topol_index)
    fn_coord_box = equilibration_dir / f"{topology_label}-box.gro"

    logger.info("Creating box: in progress...", overwrite=True, level=2)
    universe, topol = create_box(fn_topol, fn_mol, fn_out=fn_coord_box, box=box)
    logger.info("Creating box: Done.", level=2)

    fn_topol_processed = fn_coord_box.with_suffix(".top")
    topol.write(fn_topol_processed, overwrite=True)
    maxwarn = determine_maxwarn(topol)

    deffnm_em = equilibration_dir / f"{topology_label}-em"
    logger.info("Energy minimization: in progress...", overwrite=True, level=2)
    run_md(
        deffnm_em,
        fn_mdp_em,
        fn_topol_processed,
        fn_coord_box,
        fn_ndx=None,
        gmx_cmd=gmx_cmd,
        n_steps=-2,
        maxwarn=maxwarn,
        fn_log=fn_gmx_log,
    )
    logger.info("Energy minimization: Done.", level=2)

    if nsteps_npt <= 0:
        logger.info("NpT equilibration: skipped (box defined)", level=2)
        return EquilibratedTopology(
            fn_topol_processed=fn_topol_processed,
            fn_coord=deffnm_em.with_suffix(".gro"),
            universe=mda.Universe(
                deffnm_em.with_suffix(".gro"),
                to_guess=("elements", "masses"),
            ),
            box=np.asarray(universe.dimensions, dtype=float),
            maxwarn=maxwarn,
        )

    deffnm_npt = equilibration_dir / f"{topology_label}-npt"
    logger.info("NpT equilibration: in progress...", overwrite=True, level=2)
    run_md(
        deffnm_npt,
        fn_mdp_npt,
        fn_topol_processed,
        deffnm_em.with_suffix(".gro"),
        fn_ndx=None,
        gmx_cmd=gmx_cmd,
        n_steps=nsteps_npt,
        maxwarn=maxwarn,
        fn_log=fn_gmx_log,
    )
    universe = mda.Universe(
        deffnm_npt.with_suffix(".gro"),
        deffnm_npt.with_suffix(".xtc"),
        to_guess=("elements", "masses"),
    )
    universe.trajectory.add_transformations(trans.unwrap(universe.atoms))
    discard = int(universe.trajectory.n_frames * 0.2)
    box_avg = get_average_box(universe, start=discard)
    logger.info("NpT equilibration: Done.", level=2)

    return EquilibratedTopology(
        fn_topol_processed=fn_topol_processed,
        fn_coord=deffnm_npt.with_suffix(".gro"),
        universe=universe,
        box=box_avg,
        maxwarn=maxwarn,
    )


def save_training_artifacts(
    *,
    window_index: int,
    training_dir: Path,
    fn_mdp_nvt: Path,
    fn_coord: Path,
    fn_ndx: Path,
    fn_topol: Path,
    fn_bias_input: Path | None,
    bias: BiasSpec,
) -> None:
    """Save all training artifacts produced for one window."""
    window_label = window_name(window_index)
    for src, dst in [
        (fn_mdp_nvt, training_dir / f"{window_label}.mdp"),
        (fn_coord, training_dir / f"{window_label}.gro"),
        (fn_ndx, training_dir / f"{window_label}.ndx"),
        (fn_topol, training_dir / f"{window_label}.top"),
    ]:
        shutil.copy2(src, dst)

    for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
        stale_file = training_dir / f"{window_label}.{suffix}"
        if stale_file.exists():
            stale_file.unlink()

    if fn_bias_input is not None and bias.input_filename is not None:
        fn_bias_train = training_dir / f"{window_label}.{bias.input_filename}"
        shutil.copy2(fn_bias_input, fn_bias_train)


def write_reference_md_assets(
    *,
    reference_md_dir: Path,
    window_index: int,
    project_name: str,
    charge: int,
    mult: int,
    box: np.ndarray,
    fn_topol_processed: Path,
    fn_coord_nvt: Path,
    bias: BiasSpec,
    logger: Logger,
) -> None:
    """Write CP2K MD assets for one simulation window."""
    logger.info("Reference MD assets: in progress...", overwrite=True, level=2)
    window_dir = reference_md_dir / window_name(window_index)
    window_dir.mkdir(parents=True, exist_ok=True)

    strip_topol(
        fn_topol_processed,
        fn_coord_nvt,
        window_dir / "system.top",
        window_dir / "system.gro",
        window_dir / "system.xyz",
    )
    fn_plumed = None
    stale_plumed = window_dir / "plumed.dat"
    if stale_plumed.exists():
        stale_plumed.unlink()
    if bias.kind == "plumed" and bias.input_file is not None:
        fn_plumed = window_dir / "plumed.dat"
        shutil.copy2(bias.input_file, fn_plumed)

    for cp2k_specs in [
        {"eq": True, "restart": False, "fn": "md-eq-start.inp"},
        {"eq": True, "restart": True, "fn": "md-eq-restart.inp"},
        {"eq": False, "restart": True, "fn": "md-prod.inp"},
    ]:
        make_cp2k_input(
            project_name,
            charge,
            mult,
            box[:3].astype(float).tolist(),
            window_dir / "system.xyz",
            cp2k_specs["eq"],
            cp2k_specs["restart"],
            window_dir / cp2k_specs["fn"],
            plumed_input_file=fn_plumed,
        )

    fn_submit = window_dir / "submit-slurm.sh"
    write_cp2k_md_slurm_script(fn_submit, uses_plumed=fn_plumed is not None)
    fn_submit.chmod(0o755)
    logger.info("Reference MD assets: Done.", level=2)


def write_reference_single_point_assets(
    *,
    reference_sp_dir: Path,
    window_index: int,
    project_name: str,
    charge: int,
    mult: int,
    box: np.ndarray,
    fn_topol_processed: Path,
    fn_coord_nvt: Path,
    fn_trj_nvt: Path,
    n_snapshots: int,
    logger: Logger,
) -> None:
    """Write CP2K single-point assets for one simulation window."""
    logger.info(
        "Reference single-point assets: in progress...",
        overwrite=True,
        level=2,
    )
    window_dir = reference_sp_dir / window_name(window_index)
    snapshots_dir = window_dir / "snapshots"
    runs_dir = window_dir / "runs"
    window_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Reader has no dt information, set to 1.0 ps.*",
            category=UserWarning,
        )
        universe = mda.Universe(
            fn_topol_processed,
            fn_coord_nvt,
            topology_format="ITP",
        )
        universe.load_new(fn_trj_nvt, dt=1)
        snapshot_files = write_snapshot_xyz_files(
            universe,
            snapshots_dir=snapshots_dir,
            n_snapshots=n_snapshots,
        )
        fn_template_xyz = window_dir / "snapshot.xyz"
        shutil.copy2(snapshot_files[0], fn_template_xyz)
        make_cp2k_single_point_input(
            project_name,
            charge,
            mult,
            box[:3].astype(float).tolist(),
            fn_template_xyz,
            window_dir / "single-point.inp",
        )
    fn_submit = window_dir / "submit-slurm.sh"
    write_cp2k_single_point_slurm_script(fn_submit)
    fn_submit.chmod(0o755)
    logger.info("Reference single-point assets: Done.", level=2)


def main(fn_config: PathLike) -> None:
    config = PrepareConfig.load(fn_config)
    check_gmx_available(config.gmx_cmd)
    if any(system.bias.kind == "plumed" for system in config.systems):
        ensure_plumed_kernel()

    fn_gmx_log = config.project_dir / "gmx.log"
    fn_gmx_log.parent.mkdir(parents=True, exist_ok=True)

    logger = Logger("prepare", str(config.fn_log) if config.fn_log else None)
    logger.info("", level=0)
    logger.info(f"=== Preparing project: {config.project_dir.name} ===\n", level=0)

    (
        equilibration_dir,
        training_dir,
        reference_md_dir,
        reference_sp_dir,
    ) = setup_directories(config.project_dir)
    shutil.copy2(config.systems[0].fn_mdp_em, training_dir / "em.mdp")

    equilibrated_topologies: Dict[str, EquilibratedTopology] = {}

    n_total = len(config.systems)
    for i, system in enumerate(config.systems):
        logger.info(f"System: {i + 1}/{n_total}", level=1)

        topology_key = str(system.fn_topol)
        if topology_key not in equilibrated_topologies:
            topol_index = next(
                j
                for j, candidate in enumerate(config.systems)
                if candidate.fn_topol == system.fn_topol
            )
            equilibrated_topologies[topology_key] = prepare_equilibrated_topology(
                topol_index=topol_index,
                fn_topol=system.fn_topol,
                fn_mol=system.fn_mol,
                box=system.box,
                fn_mdp_em=system.fn_mdp_em,
                fn_mdp_npt=system.fn_mdp_npt,
                nsteps_npt=system.nsteps_npt,
                equilibration_dir=equilibration_dir,
                gmx_cmd=config.gmx_cmd,
                fn_gmx_log=fn_gmx_log,
                logger=logger,
            )

        topology_state = equilibrated_topologies[topology_key]

        window_label = window_name(i)
        fn_mdp_nvt = system.fn_mdp_nvt
        fn_nvt_local = equilibration_dir / f"{window_label}.mdp"
        fn_ndx = equilibration_dir / f"{window_label}.ndx"
        fn_coord = equilibration_dir / f"{window_label}.gro"
        fn_bias_local = (
            None
            if system.bias.input_filename is None
            else equilibration_dir / f"{window_label}.{system.bias.input_filename}"
        )
        fn_topol_local = equilibration_dir / f"{window_label}.top"
        fn_topol_local.write_text(topology_state.fn_topol_processed.read_text())
        fn_bias_input = None if fn_bias_local is None else prepare_bias_file(
            system.bias,
            fn_bias_local,
        )
        bias_run = localize_bias_spec(system.bias, fn_bias_input)
        pair_labels, target_distances = resolve_bias_window_definition(
            system.bias,
            topology_state.fn_coord,
        )

        write_window_coordinates(
            topology_state.universe,
            box=topology_state.box,
            fn_out=fn_coord,
            pair_labels=pair_labels,
            target_distances=target_distances,
        )
        make_ndx(topology_state.universe, None, fn_out=fn_ndx)
        shutil.copy2(fn_mdp_nvt, fn_nvt_local)

        deffnm_nvt = equilibration_dir / f"{window_label}-nvt"
        logger.info("NVT equilibration: in progress...", overwrite=True, level=2)
        fn_nvt_run = run_md(
            deffnm_nvt,
            fn_nvt_local,
            fn_topol_local,
            fn_coord,
            fn_ndx,
            bias=bias_run,
            gmx_cmd=config.gmx_cmd,
            n_steps=system.nsteps_nvt,
            maxwarn=topology_state.maxwarn,
            fn_log=fn_gmx_log,
        )
        logger.info("NVT equilibration: Done.", level=2)

        save_training_artifacts(
            window_index=i,
            training_dir=training_dir,
            fn_mdp_nvt=fn_nvt_run,
            fn_coord=fn_coord,
            fn_ndx=fn_ndx,
            fn_topol=fn_topol_local,
            fn_bias_input=fn_bias_input,
            bias=system.bias,
        )
        write_reference_md_assets(
            reference_md_dir=reference_md_dir,
            window_index=i,
            project_name=config.project_dir.name,
            charge=system.charge,
            mult=system.mult,
            box=topology_state.box,
            fn_topol_processed=topology_state.fn_topol_processed,
            fn_coord_nvt=deffnm_nvt.with_suffix(".gro"),
            bias=system.bias,
            logger=logger,
        )
        write_reference_single_point_assets(
            reference_sp_dir=reference_sp_dir,
            window_index=i,
            project_name=config.project_dir.name,
            charge=system.charge,
            mult=system.mult,
            box=topology_state.box,
            fn_topol_processed=topology_state.fn_topol_processed,
            fn_coord_nvt=deffnm_nvt.with_suffix(".gro"),
            fn_trj_nvt=deffnm_nvt.with_suffix(".xtc"),
            n_snapshots=config.n_single_point_snapshots,
            logger=logger,
        )
        logger.info("", level=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare simulation assets.")
    parser.add_argument('fn_config', help='Path to the config file [YAML].')
    args = parser.parse_args()
    main(args.fn_config)
