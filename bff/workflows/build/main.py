"""Build, equilibrate, and seed simulation systems from GROMACS topologies."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis import transformations
from MDAnalysis.selections.gromacs import SelectionWriter

from ...domain.bias import BiasSpec
from ...domain.systems import BuildSystemMetadata, write_build_system_metadata
from ...gromacs import check_gmx_available, run_md
from ...io.logs import Logger
from ...io.plumed import ensure_plumed_kernel
from .box import create_box
from .config import BuildConfig, BuildSystemConfig

PathLike = str | Path


@dataclass(slots=True)
class EquilibratedTopology:
    topology_path: Path
    universe: mda.Universe
    box: np.ndarray
    maxwarn: int


def _equilibration_key(system: BuildSystemConfig) -> tuple:
    """Systems sharing every equilibration input share one equilibration."""
    return (
        system.topology_path,
        tuple(sorted(system.templates.items())),
        None if system.box is None else tuple(system.box),
        system.mdp_em_path,
        system.mdp_npt_path,
        system.nsteps_npt,
    )


def equilibrate(
    system: BuildSystemConfig,
    *,
    label: str,
    equilibration_dir: Path,
    gmx_cmd: str,
    fn_gmx_log: Path,
    logger: Logger,
) -> EquilibratedTopology:
    """Fill a box, minimize its energy, and run NpT equilibration."""
    fn_coord_box = equilibration_dir / f"{label}-box.gro"
    logger.status("Creating box", "in progress...", overwrite=True, level=2)
    _, topol = create_box(
        system.topology_path, system.templates, fn_out=fn_coord_box, box=system.box
    )
    logger.done("Creating box", level=2)

    fn_topol = fn_coord_box.with_suffix(".top")
    topol.write(fn_topol, overwrite=True)
    maxwarn = (
        0 if np.isclose(sum(atom.charge for atom in topol.atoms), 0, atol=1e-4) else 1
    )
    if maxwarn:
        logger.warn(
            "Non-neutral topology detected; GROMACS preprocessing will use -maxwarn 1.",
            level=2,
        )

    deffnm_em = equilibration_dir / f"{label}-em"
    logger.status("Energy minimization", "in progress...", overwrite=True, level=2)
    run_md(
        deffnm_em,
        mdp=system.mdp_em_path,
        topology=fn_topol,
        coordinates=fn_coord_box,
        gmx_cmd=gmx_cmd,
        maxwarn=maxwarn,
        log=fn_gmx_log,
    )
    logger.done("Energy minimization", level=2)

    if system.nsteps_npt <= 0:
        logger.warn(
            "Skipping NpT equilibration because nsteps_npt <= 0; using the "
            "box from the constructed system.",
            level=2,
        )
        universe = mda.Universe(
            deffnm_em.with_suffix(".gro"), to_guess=("elements", "masses")
        )
        return EquilibratedTopology(
            fn_topol, universe, np.asarray(universe.dimensions, dtype=float), maxwarn
        )

    deffnm_npt = equilibration_dir / f"{label}-npt"
    logger.status("NpT equilibration", "in progress...", overwrite=True, level=2)
    run_md(
        deffnm_npt,
        mdp=system.mdp_npt_path,
        topology=fn_topol,
        coordinates=deffnm_em.with_suffix(".gro"),
        gmx_cmd=gmx_cmd,
        n_steps=system.nsteps_npt,
        maxwarn=maxwarn,
        log=fn_gmx_log,
    )
    universe = mda.Universe(
        fn_topol,
        deffnm_npt.with_suffix(".xtc"),
        topology_format="ITP",
        to_guess=("elements", "masses"),
    )
    # Average the box over the last 80% of the equilibration.
    discard = int(universe.trajectory.n_frames * 0.2)
    boxes = [ts.dimensions for ts in universe.trajectory[discard:]]
    box = np.round(np.mean(boxes, axis=0), 4)
    universe.trajectory.add_transformations(transformations.unwrap(universe.atoms))
    logger.done("NpT equilibration", level=2)
    return EquilibratedTopology(fn_topol, universe, box, maxwarn)


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
                molecule_offset + atom_index for atom_index in molecule_virtual_sites
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
            "Generated reference topology and coordinates have different atom ordering."
        )
    return len(virtual_site_indices)


def main(fn_config: PathLike) -> None:
    config = BuildConfig.load(fn_config)
    check_gmx_available(config.gmx_cmd)
    if any(system.bias.kind == "plumed" for system in config.systems):
        ensure_plumed_kernel()

    project_dir = config.project_dir.resolve()
    equilibration_dir = project_dir / "equilibration"
    equilibration_dir.mkdir(parents=True, exist_ok=True)
    systems_dir = project_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    fn_gmx_log = project_dir / "gromacs.log"
    logger = Logger("build", str(config.fn_log) if config.fn_log else None, mode="w")
    logger.section(f"Build: {project_dir.name}")
    logger.kv("Config", Path(fn_config).resolve())
    logger.kv("Project directory", project_dir)
    logger.kv("Equilibration directory", equilibration_dir)
    logger.kv("Systems directory", systems_dir)
    logger.kv("Systems", len(config.systems))
    logger.kv("GROMACS command", config.gmx_cmd)
    if any(system.bias.is_biased for system in config.systems):
        logger.warn(
            "Bias files are staged verbatim. For strong restraints, supply a "
            "user-prepared ramp-up stage or starting structures already near the "
            "intended region.",
        )
    if config.fn_log is not None:
        logger.kv("Log file", config.fn_log.resolve())
    logger.blank()

    keys = [_equilibration_key(system) for system in config.systems]
    equilibrated: dict[tuple, EquilibratedTopology] = {}
    for i, (system, key) in enumerate(zip(config.systems, keys)):
        logger.info(
            f"System {i + 1}/{len(config.systems)}: {system.system_id}", level=1
        )
        if key not in equilibrated:
            equilibrated[key] = equilibrate(
                system,
                label=f"topology-{keys.index(key):03d}",
                equilibration_dir=equilibration_dir,
                gmx_cmd=config.gmx_cmd,
                fn_gmx_log=fn_gmx_log,
                logger=logger,
            )
        state = equilibrated[key]

        system_dir = systems_dir / system.system_id
        system_dir.mkdir(parents=True, exist_ok=True)
        fn_topol = system_dir / "topology.top"
        fn_coord = system_dir / "coordinates.gro"
        fn_ndx = system_dir / "index.ndx"
        fn_mdp_prod = system_dir / "production.mdp"
        shutil.copy2(state.topology_path, fn_topol)
        shutil.copy2(system.mdp_em_path, system_dir / "em.mdp")
        shutil.copy2(system.mdp_npt_path, system_dir / "npt.mdp")
        shutil.copy2(system.mdp_production_path, fn_mdp_prod)
        with mda.Writer(fn_coord, "w") as writer:
            ts = state.universe.trajectory[-1]
            ts.dimensions = state.box
            writer.write(state.universe.atoms)
        with SelectionWriter(str(fn_ndx), mode="w") as ndx:
            ndx.write(state.universe.atoms, name="System")

        for name in ("bias.colvars.dat", "bias.plumed.dat"):
            (system_dir / name).unlink(missing_ok=True)
        bias = system.bias
        if bias.is_biased:
            fn_bias = system_dir / bias.input_filename
            shutil.copy2(bias.input_file, fn_bias)
            bias = BiasSpec.load(fn_bias)

        logger.status("Production seed run", "in progress...", overwrite=True, level=2)
        run_md(
            system_dir / "production",
            mdp=fn_mdp_prod,
            topology=fn_topol,
            coordinates=fn_coord,
            index=fn_ndx,
            bias=bias,
            gmx_cmd=config.gmx_cmd,
            n_steps=system.nsteps_prod,
            maxwarn=state.maxwarn,
            log=fn_gmx_log,
        )
        logger.done("Production seed run", level=2)

        removed = write_reference_system(
            fn_topol,
            system_dir / "production.gro",
            system_dir / "reference" / "topology.top",
            system_dir / "reference" / "coordinates.gro",
        )
        logger.done(
            "Reference-compatible system",
            detail=f"removed {removed} virtual sites",
            level=2,
        )
        write_build_system_metadata(
            project_dir,
            system.system_id,
            BuildSystemMetadata(
                system_name=system.system_name,
                charge=system.charge,
                multiplicity=system.mult,
                box=tuple(float(value) for value in state.box),
                maxwarn=state.maxwarn,
                production_steps=system.nsteps_prod,
            ),
        )
        logger.done("System metadata", detail=str(system_dir / "system.yaml"), level=2)
        logger.blank()
