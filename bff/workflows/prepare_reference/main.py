from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import MDAnalysis as mda

from ...domain.systems import (
    ReferenceSystemMetadata,
    write_reference_system_metadata,
)
from ...io.cp2k import (
    get_cp2k_elements,
    get_cp2k_single_atom_directory_name,
    make_cp2k_input,
    make_cp2k_isolated_atom_input,
)
from ...io.logs import Logger
from .._shared.preparation import PreparedSystem, strip_topol, write_snapshot_xyz_files
from .config import PrepareReferenceConfig

CP2K_SNAPSHOT_MD_STEPS = 100


def main(fn_config: str | Path) -> None:
    config = PrepareReferenceConfig.load(fn_config)
    systems_dir = config.output_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger("prepare-reference", str(config.log), mode="w")
    logger.section("Prepare Reference")
    logger.kv("Config", config.fn_config)
    logger.kv("Build source", config.source)
    logger.kv("Output directory", config.output_dir)
    logger.kv("Systems", len(config.systems))
    logger.kv("Single-point snapshots per system", config.n_single_point_snapshots)
    if config.n_single_point_snapshots < 10:
        logger.warn(
            "Very few CP2K reference snapshots are requested; train/valid splits "
            "may be noisy."
        )
    logger.blank()

    for index, system in enumerate(config.systems, start=1):
        logger.info(
            f"System {index}/{len(config.systems)}: {system.system_id}", level=1
        )
        snapshot_count, elements = write_reference_inputs(
            system,
            systems_dir,
            config.n_single_point_snapshots,
            logger,
        )
        write_reference_system_metadata(
            config.output_dir,
            system.system_id,
            ReferenceSystemMetadata(
                system_name=system.system_name,
                charge=system.charge,
                multiplicity=system.multiplicity,
                box=tuple(float(value) for value in system.box),
                snapshot_count=snapshot_count,
                elements=elements,
            ),
        )
        logger.done(
            "System metadata",
            detail=str(systems_dir / system.system_id / "system.yaml"),
            level=2,
        )
        logger.blank()


def write_reference_inputs(
    system: PreparedSystem,
    systems_dir: Path,
    n_snapshots: int,
    logger: Logger,
) -> tuple[int, tuple[str, ...]]:
    logger.status("Reference inputs", "in progress...", overwrite=True, level=2)
    system_dir = systems_dir / system.system_id
    md_dir = system_dir / "md"
    snapshots_dir = system_dir / "snapshots"
    single_atoms_dir = system_dir / "single-atoms"
    snapshot_xyz_dir = snapshots_dir / "xyz"

    for directory in (md_dir, snapshots_dir, single_atoms_dir):
        if directory.exists():
            shutil.rmtree(directory)
    for directory in (system_dir, md_dir, single_atoms_dir, snapshot_xyz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    system_top = system_dir / "system.top"
    system_gro = system_dir / "system.gro"
    system_xyz = system_dir / "system.xyz"
    strip_topol(
        system.topology_path,
        system.production_coordinates_path,
        system_top,
        system_gro,
        system_xyz,
    )

    md_position = md_dir / "pos.xyz"
    shutil.copy2(system_xyz, md_position)
    elements = tuple(sorted(get_cp2k_elements(system_xyz)))
    for element in elements:
        atom_dir = single_atoms_dir / get_cp2k_single_atom_directory_name(element)
        atom_dir.mkdir(parents=True, exist_ok=True)
        make_cp2k_isolated_atom_input(element, atom_dir / "input.inp")

    plumed = None
    if system.bias.kind == "plumed" and system.bias.input_file is not None:
        plumed = md_dir / "plumed.dat"
        shutil.copy2(system.bias.input_file, plumed)

    box = system.box[:3].astype(float).tolist()
    make_cp2k_input(
        "md", system.charge, system.multiplicity, box, md_position, md_dir / "md.inp",
        plumed_input_file=plumed,
    )
    make_cp2k_input(
        "md", system.charge, system.multiplicity, box, md_position,
        md_dir / "md-restart.inp", restart=True, plumed_input_file=plumed,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Reader has no dt information, set to 1.0 ps.*",
            category=UserWarning,
        )
        universe = mda.Universe(
            system.topology_path,
            system.production_coordinates_path,
            topology_format="ITP",
        )
        universe.load_new(system.production_trajectory_path, dt=1)
        snapshots = write_snapshot_xyz_files(
            universe, snapshots_dir=snapshot_xyz_dir, n_snapshots=n_snapshots
        )

    make_cp2k_input(
        "sp", system.charge, system.multiplicity, box, snapshots[0],
        snapshots_dir / "sp.inp", kind="single_point",
        coord_filename="md-pos-1.xyz",
    )
    make_cp2k_input(
        "md", system.charge, system.multiplicity, box, snapshots[0],
        snapshots_dir / "md.inp", kind="xtb_md", steps=CP2K_SNAPSHOT_MD_STEPS,
        coord_filename="pos.xyz",
    )
    logger.done(
        "Reference inputs",
        detail=f"{system_dir.resolve()} ({len(snapshots)} snapshots)",
        level=2,
    )
    return len(snapshots), elements
