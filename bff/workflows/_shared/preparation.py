from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis.selections.gromacs import SelectionWriter

from ...domain.bias import BiasSpec
from ...domain.systems import (
    load_build_system_metadata,
    validate_system_id,
)
from ...io.colvars import write_mdp_with_colvars
from ...io.commands import build_command
from ...io.plumed import ensure_plumed_kernel

PathLike = str | Path

# MDAnalysis emits this while transitioning ITP element guessing APIs.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"MDAnalysis\.topology\.ITPParser",
)


@dataclass(frozen=True, slots=True)
class PreparedSystem:
    system_id: str
    system_name: str | None
    charge: int
    multiplicity: int
    box: np.ndarray
    topology_path: Path
    coordinates_path: Path
    mdp_em_path: Path
    mdp_npt_path: Path
    mdp_production_path: Path
    index_path: Path
    production_coordinates_path: Path
    production_trajectory_path: Path
    bias: BiasSpec
    nsteps_prod: int
    maxwarn: int


def _required_file(system_dir: Path, name: str, *, system_id: str) -> Path:
    path = system_dir / name
    if not path.is_file():
        raise FileNotFoundError(
            f"system {system_id!r}: expected build output {name!r} at {path}; "
            "regenerate the build stage or correct 'source'."
        )
    return path


def load_build_system(source: PathLike, system_id: str) -> PreparedSystem:
    """Resolve one system from the documented build-directory contract."""
    source = Path(source).resolve()
    system_id = validate_system_id(system_id)
    system_dir = source / "systems" / system_id
    metadata = load_build_system_metadata(source, system_id)
    colvars = system_dir / "bias.colvars.dat"
    plumed = system_dir / "bias.plumed.dat"
    if colvars.is_file() and plumed.is_file():
        raise ValueError(
            f"system {system_id!r}: both reserved bias files exist ({colvars} and "
            f"{plumed}); retain exactly one."
        )
    if colvars.is_file():
        bias = BiasSpec(kind="colvars", colvars_file=colvars)
    elif plumed.is_file():
        bias = BiasSpec(kind="plumed", plumed_file=plumed)
    else:
        bias = BiasSpec()
    return PreparedSystem(
        system_id=system_id,
        system_name=metadata.system_name,
        charge=metadata.charge,
        multiplicity=metadata.multiplicity,
        box=np.asarray(metadata.box, dtype=float),
        topology_path=_required_file(
            system_dir, "topology.top", system_id=system_id
        ),
        coordinates_path=_required_file(
            system_dir, "coordinates.gro", system_id=system_id
        ),
        mdp_em_path=_required_file(system_dir, "em.mdp", system_id=system_id),
        mdp_npt_path=_required_file(system_dir, "npt.mdp", system_id=system_id),
        mdp_production_path=_required_file(
            system_dir, "production.mdp", system_id=system_id
        ),
        index_path=_required_file(system_dir, "index.ndx", system_id=system_id),
        production_coordinates_path=_required_file(
            system_dir, "production.gro", system_id=system_id
        ),
        production_trajectory_path=_required_file(
            system_dir, "production.xtc", system_id=system_id
        ),
        bias=bias,
        nsteps_prod=metadata.production_steps,
        maxwarn=metadata.maxwarn,
    )


def topology_name(topology_index: int) -> str:
    return f"topology-{topology_index:03d}"


def check_gmx_available(gmx_cmd: str = "gmx") -> None:
    try:
        subprocess.run(
            build_command(gmx_cmd, "--version"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"GROMACS command {gmx_cmd!r} is not available.\n"
            "Make sure the executable is on PATH in the job environment."
        ) from exc


def make_ndx(
    universe: mda.Universe,
    selections: list[str] | None,
    fn_out: PathLike,
) -> None:
    with SelectionWriter(str(fn_out), mode="w") as ndx:
        ndx.write(universe.atoms, name="System")
        if selections is None:
            return
        for selection in selections:
            for atom in selection.split():
                ndx.write(universe.select_atoms("name " + atom), name=atom)


def determine_maxwarn(topol: Topology) -> int:
    total_charge = sum(atom.charge for atom in topol.atoms)
    return 1 if not np.isclose(total_charge, 0, atol=1e-4) else 0


def run_md(
    name: PathLike,
    fn_mdp: PathLike,
    fn_topol: PathLike,
    fn_coord: PathLike,
    fn_ndx: PathLike | None,
    *,
    bias: BiasSpec | None = None,
    gmx_cmd: str = "gmx",
    n_steps: int = -2,
    maxwarn: int = 0,
    fn_log: PathLike = "gmx.log",
) -> Path:
    fn_mdp_path = Path(fn_mdp).resolve()
    fn_topol = str(Path(fn_topol).resolve())
    fn_coord = str(Path(fn_coord).resolve())
    fn_ndx = str(Path(fn_ndx).resolve()) if fn_ndx else None
    fn_tpr = str(name) + ".tpr"
    fn_mdp_run = fn_mdp_path
    run_cwd = fn_mdp_path.parent
    run_env = None
    mdrun_cmd = build_command(
        gmx_cmd,
        "mdrun",
        "-deffnm",
        str(name),
        "-nsteps",
        str(n_steps),
    )

    if bias is not None and bias.kind == "colvars" and bias.input_file is not None:
        fn_bias_local = fn_mdp_path.parent / Path(bias.input_file).name
        if Path(bias.input_file).resolve() != fn_bias_local.resolve():
            shutil.copy2(bias.input_file, fn_bias_local)
        fn_mdp_run = Path(f"{name}-colvars.mdp").resolve()
        run_cwd = fn_mdp_run.parent
        write_mdp_with_colvars(
            fn_mdp_path,
            fn_bias_local,
            fn_mdp_run,
            working_dir=run_cwd,
        )
    elif bias is not None and bias.kind == "plumed" and bias.input_file is not None:
        kernel = ensure_plumed_kernel()
        run_env = dict(os.environ)
        run_env.setdefault("PLUMED_KERNEL", str(kernel))
        mdrun_cmd.extend(["-plumed", str(bias.input_file)])

    grompp_cmd = build_command(
        gmx_cmd,
        "grompp",
        "-f",
        str(fn_mdp_run),
        "-c",
        fn_coord,
        "-p",
        fn_topol,
        "-o",
        fn_tpr,
        "-maxwarn",
        str(maxwarn),
    )
    if fn_ndx:
        grompp_cmd.extend(["-n", fn_ndx])

    with open(fn_log, "a", encoding="utf-8") as log:
        subprocess.run(grompp_cmd, stdout=log, stderr=log, check=True, cwd=run_cwd)
        subprocess.run(
            mdrun_cmd,
            stdout=log,
            stderr=log,
            check=True,
            env=run_env,
            cwd=run_cwd,
        )

    return fn_mdp_run


def get_average_box(
    universe: mda.Universe,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> np.ndarray:
    frames = universe.trajectory[
        slice(start, stop or universe.trajectory.n_frames, step)
    ]
    box = np.zeros((len(frames), 6))
    for i, ts in enumerate(frames):
        box[i] = ts.dimensions
    return np.round(np.mean(box, axis=0), 4)


def write_reference_system(
    fn_topol: PathLike,
    fn_coords: PathLike,
    fn_out_topol: PathLike,
    fn_out_coords: PathLike,
) -> int:
    """Write matching topology and coordinates with virtual sites removed."""
    top = Topology(fn_topol)
    universe = mda.Universe(fn_topol, fn_coords, topology_format="ITP")

    virtual_site_indices: list[int] = []
    atom_offset = 0
    for mol, count in top.molecules.values():
        molecule_virtual_sites = {
            virtual_site.ai.nr - 1
            for section in mol.VSITE_SECTIONS
            for virtual_site in getattr(mol, section)
        }
        for molecule_index in range(count):
            molecule_offset = atom_offset + molecule_index * len(mol.atoms)
            virtual_site_indices.extend(
                molecule_offset + atom_index
                for atom_index in molecule_virtual_sites
            )
        atom_offset += count * len(mol.atoms)
        mol.remove_vsites()

    if atom_offset != len(universe.atoms):
        raise ValueError(
            f"Topology {fn_topol} expands to {atom_offset} atoms, but coordinates "
            f"{fn_coords} contain {len(universe.atoms)} atoms."
        )

    keep = np.ones(len(universe.atoms), dtype=bool)
    keep[virtual_site_indices] = False
    atoms = universe.atoms[keep]

    fn_out_topol = Path(fn_out_topol)
    fn_out_coords = Path(fn_out_coords)
    fn_out_topol.parent.mkdir(parents=True, exist_ok=True)
    fn_out_coords.parent.mkdir(parents=True, exist_ok=True)
    top.write(fn_out_topol, overwrite=True)

    universe.trajectory[-1]
    with mda.Writer(fn_out_coords, n_atoms=len(atoms)) as writer:
        writer.write(atoms)

    reference = mda.Universe(
        fn_out_topol,
        fn_out_coords,
        topology_format="ITP",
    )
    if len(reference.atoms) != len(atoms):
        raise ValueError(
            f"Generated reference topology {fn_out_topol} contains "
            f"{len(reference.atoms)} atoms, but {fn_out_coords} contains "
            f"{len(atoms)} atoms."
        )
    if not np.array_equal(reference.atoms.names, atoms.names):
        raise ValueError(
            "Generated reference topology and coordinates have different atom "
            "ordering."
        )
    return len(virtual_site_indices)


def sample_snapshot_indices(n_frames: int, n_snapshots: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("Cannot sample snapshots from an empty trajectory.")
    count = min(int(n_snapshots), int(n_frames))
    return np.unique(np.linspace(0, n_frames - 1, num=count, dtype=int))
