import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis import transformations as trans
from MDAnalysis.selections.gromacs import SelectionWriter

from ...domain.bias import BiasSpec
from ...io.colvars import write_mdp_with_colvars
from ...io.commands import build_command
from ...io.cp2k import (
    get_cp2k_elements,
    get_cp2k_single_atom_directory_name,
    make_cp2k_input,
    make_cp2k_isolated_atom_input,
)
from ...io.extxyz import write_extxyz_frame
from ...io.logs import Logger
from ...io.plumed import ensure_plumed_kernel
from ...topology import create_box
from .config import BuildConfig

PathLike = Union[str, Path]
CP2K_SNAPSHOT_MD_STEPS = 100


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


def setup_directories(main_dir: PathLike = Path('./')) -> tuple[Path, Path, Path]:
    """Create required directories for the build workflow."""
    main_dir = Path(main_dir).resolve()
    equilibration_dir = main_dir / "equilibration"
    ffmd_dir = main_dir / "ffmd"
    reference_dir = main_dir / "reference"

    for directory in [
        equilibration_dir,
        ffmd_dir,
        reference_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return equilibration_dir, ffmd_dir, reference_dir


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
        if Path(bias.input_file).resolve() != fn_bias_local.resolve():
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
    ts = u.trajectory[-1]
    for fn_out in fn_out_coords:
        fn_out = Path(fn_out)
        if fn_out.suffix.lower() == ".xyz":
            write_extxyz_frame(atoms, fn_out, dimensions=ts.dimensions)
        else:
            atoms.write(fn_out, frames=u.trajectory[[-1]])


def system_name(system_index: int) -> str:
    """Return the canonical zero-padded prepared-system name."""
    return f"system-{system_index:03d}"


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
        ts = universe.trajectory[frame_index]
        write_extxyz_frame(atoms, fn_snapshot, dimensions=ts.dimensions)
        written.append(fn_snapshot)

    return written


def system_run_name(system_index: int) -> str:
    """Return the canonical filename stem for the prepared production run."""
    return f"{system_name(system_index)}-prod"


def build_equilibrated_topology(
    *,
    topol_index: int,
    fn_topol: Path,
    templates: dict[str, Path],
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

    logger.status("Creating box", "in progress...", overwrite=True, level=2)
    universe, topol = create_box(
        fn_topol,
        templates,
        fn_out=fn_coord_box,
        box=box,
    )
    logger.done("Creating box", level=2)

    fn_topol_processed = fn_coord_box.with_suffix(".top")
    topol.write(fn_topol_processed, overwrite=True)
    maxwarn = determine_maxwarn(topol)
    logger.warn_if(
        maxwarn > 0,
        "Non-neutral topology detected; GROMACS preprocessing will use -maxwarn 1.",
        level=2,
    )

    deffnm_em = equilibration_dir / f"{topology_label}-em"
    logger.status(
        "Energy minimization",
        "in progress...",
        overwrite=True,
        level=2,
    )
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
    logger.done("Energy minimization", level=2)

    if nsteps_npt <= 0:
        logger.warn(
            "Skipping NpT equilibration because nsteps_npt <= 0; using the "
            "box from the constructed system.",
            level=2,
        )
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
    logger.status("NpT equilibration", "in progress...", overwrite=True, level=2)
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
        fn_topol_processed,
        deffnm_npt.with_suffix(".xtc"),
        topology_format="ITP",
        to_guess=("elements", "masses"),
    )
    universe.trajectory.add_transformations(trans.unwrap(universe.atoms))
    discard = int(universe.trajectory.n_frames * 0.2)
    box_avg = get_average_box(universe, start=discard)
    logger.done("NpT equilibration", level=2)

    return EquilibratedTopology(
        fn_topol_processed=fn_topol_processed,
        fn_coord=deffnm_npt.with_suffix(".gro"),
        universe=universe,
        box=box_avg,
        maxwarn=maxwarn,
    )


def save_ffmd_artifacts(
    *,
    window_index: int,
    ffmd_dir: Path,
    fn_mdp_em: Path,
    fn_mdp_npt: Path,
    fn_mdp_prod: Path,
    fn_coord: Path,
    fn_ndx: Path,
    fn_topol: Path,
    fn_bias_input: Path | None,
    bias: BiasSpec,
) -> None:
    """Save all FFMD artifacts produced for one window."""
    system_label = system_name(window_index)
    asset_dir = ffmd_dir / system_label
    asset_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (fn_mdp_em, asset_dir / f"{system_label}.em.mdp"),
        (fn_mdp_npt, asset_dir / f"{system_label}.npt.mdp"),
        (fn_mdp_prod, asset_dir / f"{system_label}.mdp"),
        (fn_coord, asset_dir / f"{system_label}.gro"),
        (fn_ndx, asset_dir / f"{system_label}.ndx"),
        (fn_topol, asset_dir / f"{system_label}.top"),
    ]:
        shutil.copy2(src, dst)

    for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
        stale_file = asset_dir / f"{system_label}.{suffix}"
        if stale_file.exists():
            stale_file.unlink()

    if fn_bias_input is not None and bias.input_filename is not None:
        fn_bias_train = asset_dir / f"{system_label}.{bias.input_filename}"
        shutil.copy2(fn_bias_input, fn_bias_train)


def write_reference_assets(
    *,
    reference_dir: Path,
    window_index: int,
    charge: int,
    mult: int,
    box: np.ndarray,
    fn_topol_processed: Path,
    fn_coord_nvt: Path,
    fn_trj_nvt: Path,
    bias: BiasSpec,
    n_snapshots: int,
    logger: Logger,
) -> None:
    """Write the full CP2K reference tree for one simulation window."""
    logger.status("Reference assets", "in progress...", overwrite=True, level=2)
    window_dir = reference_dir / system_name(window_index)
    md_dir = window_dir / "md"
    snapshots_dir = window_dir / "snapshots"
    single_atoms_dir = window_dir / "single-atoms"
    snapshot_xyz_dir = snapshots_dir / "xyz"

    for directory in [
        window_dir,
        md_dir,
        single_atoms_dir,
        snapshot_xyz_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    fn_system_top = window_dir / "system.top"
    fn_system_gro = window_dir / "system.gro"
    fn_system_xyz = window_dir / "system.xyz"
    strip_topol(
        fn_topol_processed,
        fn_coord_nvt,
        fn_system_top,
        fn_system_gro,
        fn_system_xyz,
    )
    single_atom_elements = get_cp2k_elements(fn_system_xyz)

    fn_md_pos = md_dir / "pos.xyz"
    shutil.copy2(fn_system_xyz, fn_md_pos)

    for element in single_atom_elements:
        atom_dir = single_atoms_dir / get_cp2k_single_atom_directory_name(element)
        atom_dir.mkdir(parents=True, exist_ok=True)
        make_cp2k_isolated_atom_input(element, atom_dir / "input.inp")

    fn_plumed = None
    stale_plumed = md_dir / "plumed.dat"
    if stale_plumed.exists():
        stale_plumed.unlink()
    if bias.kind == "plumed" and bias.input_file is not None:
        fn_plumed = md_dir / "plumed.dat"
        shutil.copy2(bias.input_file, fn_plumed)

    stale_snapshot_pos = snapshots_dir / "pos.xyz"
    if stale_snapshot_pos.exists():
        stale_snapshot_pos.unlink()

    make_cp2k_input(
        "md",
        charge,
        mult,
        box[:3].astype(float).tolist(),
        fn_md_pos,
        md_dir / "md.inp",
        plumed_input_file=fn_plumed,
    )
    make_cp2k_input(
        "md",
        charge,
        mult,
        box[:3].astype(float).tolist(),
        fn_md_pos,
        md_dir / "md-restart.inp",
        restart=True,
        plumed_input_file=fn_plumed,
    )

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
            snapshots_dir=snapshot_xyz_dir,
            n_snapshots=n_snapshots,
        )

    make_cp2k_input(
        "sp",
        charge,
        mult,
        box[:3].astype(float).tolist(),
        snapshot_files[0],
        snapshots_dir / "sp.inp",
        kind="single_point",
        coord_filename="md-pos-1.xyz",
    )
    make_cp2k_input(
        "md",
        charge,
        mult,
        box[:3].astype(float).tolist(),
        snapshot_files[0],
        snapshots_dir / "md.inp",
        kind="xtb_md",
        steps=CP2K_SNAPSHOT_MD_STEPS,
        coord_filename="pos.xyz",
    )
    logger.done("Reference assets", level=2)


def main(fn_config: PathLike) -> None:
    config = BuildConfig.load(fn_config)
    check_gmx_available(config.gmx_cmd)
    if any(system.bias.kind == "plumed" for system in config.systems):
        ensure_plumed_kernel()

    fn_gmx_log = config.project_dir / "gmx.log"
    fn_gmx_log.parent.mkdir(parents=True, exist_ok=True)

    logger = Logger(
        "build",
        str(config.fn_log) if config.fn_log else None,
        mode="w",
    )
    logger.section(f"Build: {config.project_dir.name}")
    logger.kv("Config", Path(fn_config).resolve())
    logger.kv("Project directory", config.project_dir.resolve())
    logger.kv("Systems", len(config.systems))
    logger.kv("GROMACS command", config.gmx_cmd)
    logger.kv(
        "Single-point snapshots per window",
        config.n_single_point_snapshots,
    )
    logger.warn_if(
        config.n_single_point_snapshots < 10,
        "Very few CP2K reference snapshots are requested; train/valid splits "
        "may be noisy.",
    )
    logger.warn_if(
        any(system.bias.is_biased for system in config.systems),
        "Bias files are staged verbatim. For strong restraints, supply a "
        "user-prepared ramp-up stage or starting structures already near the "
        "intended region.",
    )
    if config.fn_log is not None:
        logger.kv("Log file", config.fn_log.resolve())
    logger.blank()

    (
        equilibration_dir,
        ffmd_dir,
        reference_dir,
    ) = setup_directories(config.project_dir)

    equilibrated_topologies: Dict[str, EquilibratedTopology] = {}

    n_total = len(config.systems)
    for i, system in enumerate(config.systems):
        logger.info(f"System {i + 1}/{n_total}", level=1)

        topology_key = str(system.fn_topol)
        if topology_key not in equilibrated_topologies:
            topol_index = next(
                j
                for j, candidate in enumerate(config.systems)
                if candidate.fn_topol == system.fn_topol
            )
            equilibrated_topologies[topology_key] = build_equilibrated_topology(
                topol_index=topol_index,
                fn_topol=system.fn_topol,
                templates=system.templates,
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

        system_label = system_name(i)
        fn_mdp_prod = system.fn_mdp_prod
        fn_prod_local = equilibration_dir / f"{system_label}.mdp"
        fn_ndx = equilibration_dir / f"{system_label}.ndx"
        fn_coord = equilibration_dir / f"{system_label}.gro"
        fn_topol_local = equilibration_dir / f"{system_label}.top"
        fn_topol_local.write_text(topology_state.fn_topol_processed.read_text())

        fn_bias_input = None
        if system.bias.input_file is not None and system.bias.input_filename is not None:
            fn_bias_input = (
                equilibration_dir / f"{system_label}.{system.bias.input_filename}"
            )
            shutil.copy2(system.bias.input_file, fn_bias_input)

        if fn_bias_input is None or not system.bias.is_biased:
            bias_run = system.bias
        elif system.bias.kind == "colvars":
            bias_run = BiasSpec(kind="colvars", colvars_file=fn_bias_input)
        else:
            bias_run = BiasSpec(kind="plumed", plumed_file=fn_bias_input)

        with mda.Writer(fn_coord, "w") as writer:
            ts = topology_state.universe.trajectory[-1]
            ts.dimensions = topology_state.box
            writer.write(topology_state.universe.atoms)
        make_ndx(topology_state.universe, None, fn_out=fn_ndx)
        shutil.copy2(fn_mdp_prod, fn_prod_local)

        deffnm_prod = equilibration_dir / system_run_name(i)
        logger.status(
            "Production build run",
            "in progress...",
            overwrite=True,
            level=2,
        )
        run_md(
            deffnm_prod,
            fn_prod_local,
            fn_topol_local,
            fn_coord,
            fn_ndx,
            bias=bias_run,
            gmx_cmd=config.gmx_cmd,
            n_steps=system.nsteps_prod,
            maxwarn=topology_state.maxwarn,
            fn_log=fn_gmx_log,
        )
        logger.done("Production build run", level=2)

        logger.status("FFMD assets", "in progress...", overwrite=True, level=2)
        save_ffmd_artifacts(
            window_index=i,
            ffmd_dir=ffmd_dir,
            fn_mdp_em=system.fn_mdp_em,
            fn_mdp_npt=system.fn_mdp_npt,
            fn_mdp_prod=system.fn_mdp_prod,
            fn_coord=fn_coord,
            fn_ndx=fn_ndx,
            fn_topol=fn_topol_local,
            fn_bias_input=fn_bias_input,
            bias=system.bias,
        )
        logger.done("FFMD assets", level=2)
        write_reference_assets(
            reference_dir=reference_dir,
            window_index=i,
            charge=system.charge,
            mult=system.mult,
            box=topology_state.box,
            fn_topol_processed=topology_state.fn_topol_processed,
            fn_coord_nvt=deffnm_prod.with_suffix(".gro"),
            bias=system.bias,
            fn_trj_nvt=deffnm_prod.with_suffix(".xtc"),
            n_snapshots=config.n_single_point_snapshots,
            logger=logger,
        )
        logger.blank()

