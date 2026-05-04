from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import MDAnalysis as mda

from ...io.cp2k import (
    get_cp2k_elements,
    get_cp2k_single_atom_directory_name,
    make_cp2k_input,
    make_cp2k_isolated_atom_input,
)
from ...io.logs import Logger
from .._shared.preparation import (
    PreparedSystem,
    strip_topol,
    system_name,
    write_snapshot_xyz_files,
)
from .config import PrepareAssetsConfig

CP2K_SNAPSHOT_MD_STEPS = 100


def main(fn_config: str | Path) -> None:
    config = PrepareAssetsConfig.load(fn_config)
    config.ffmd_dir.mkdir(parents=True, exist_ok=True)
    config.reference_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger("prepare-assets")
    logger.section("Prepare Assets")
    logger.kv("Config", config.fn_config)
    logger.kv("Build manifest", config.manifest.fn_manifest)
    logger.kv("FFMD directory", config.ffmd_dir.resolve())
    logger.kv("Reference directory", config.reference_dir.resolve())
    logger.kv("Systems", len(config.systems))
    logger.kv("Single-point snapshots per system", config.n_single_point_snapshots)
    if config.n_single_point_snapshots < 10:
        logger.warn(
            "Very few CP2K reference snapshots are requested; train/valid splits "
            "may be noisy.",
        )
    logger.blank()

    for index, system in enumerate(config.systems, start=1):
        logger.info(f"System {index}/{len(config.systems)}", level=1)
        write_ffmd_assets(system, config.ffmd_dir, logger)
        write_reference_assets(
            system,
            config.reference_dir,
            config.n_single_point_snapshots,
            logger,
        )
        logger.blank()


def write_ffmd_assets(
    system: PreparedSystem,
    ffmd_dir: Path,
    logger: Logger,
) -> None:
    logger.status("FFMD assets", "in progress...", overwrite=True, level=2)
    system_label = system_name(system.system_id)
    asset_dir = ffmd_dir / system_label
    asset_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
        stale = asset_dir / f"{system_label}.{suffix}"
        if stale.exists():
            stale.unlink()

    for src, dst in [
        (system.fn_mdp_em, asset_dir / f"{system_label}.em.mdp"),
        (system.fn_mdp_npt, asset_dir / f"{system_label}.npt.mdp"),
        (system.fn_mdp_prod, asset_dir / f"{system_label}.mdp"),
        (system.fn_prod_coord, asset_dir / f"{system_label}.gro"),
        (system.fn_ndx, asset_dir / f"{system_label}.ndx"),
        (system.fn_topol, asset_dir / f"{system_label}.top"),
    ]:
        shutil.copy2(src, dst)

    if system.bias.input_file is not None and system.bias.input_filename is not None:
        shutil.copy2(
            system.bias.input_file,
            asset_dir / f"{system_label}.{system.bias.input_filename}",
        )
    logger.done("FFMD assets", detail=str(asset_dir.resolve()), level=2)


def write_reference_assets(
    system: PreparedSystem,
    reference_dir: Path,
    n_snapshots: int,
    logger: Logger,
) -> None:
    logger.status("Reference assets", "in progress...", overwrite=True, level=2)
    system_label = system_name(system.system_id)
    system_dir = reference_dir / system_label
    md_dir = system_dir / "md"
    snapshots_dir = system_dir / "snapshots"
    single_atoms_dir = system_dir / "single-atoms"
    snapshot_xyz_dir = snapshots_dir / "xyz"

    for directory in (md_dir, snapshots_dir, single_atoms_dir):
        if directory.exists():
            shutil.rmtree(directory)
    for directory in (system_dir, md_dir, single_atoms_dir, snapshot_xyz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fn_system_top = system_dir / "system.top"
    fn_system_gro = system_dir / "system.gro"
    fn_system_xyz = system_dir / "system.xyz"
    strip_topol(
        system.fn_topol,
        system.fn_prod_coord,
        fn_system_top,
        fn_system_gro,
        fn_system_xyz,
    )

    fn_md_pos = md_dir / "pos.xyz"
    shutil.copy2(fn_system_xyz, fn_md_pos)

    for element in get_cp2k_elements(fn_system_xyz):
        atom_dir = single_atoms_dir / get_cp2k_single_atom_directory_name(element)
        atom_dir.mkdir(parents=True, exist_ok=True)
        make_cp2k_isolated_atom_input(element, atom_dir / "input.inp")

    fn_plumed = None
    if system.bias.kind == "plumed" and system.bias.input_file is not None:
        fn_plumed = md_dir / "plumed.dat"
        shutil.copy2(system.bias.input_file, fn_plumed)

    make_cp2k_input(
        "md",
        system.charge,
        system.multiplicity,
        system.box[:3].astype(float).tolist(),
        fn_md_pos,
        md_dir / "md.inp",
        plumed_input_file=fn_plumed,
    )
    make_cp2k_input(
        "md",
        system.charge,
        system.multiplicity,
        system.box[:3].astype(float).tolist(),
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
            system.fn_topol,
            system.fn_prod_coord,
            topology_format="ITP",
        )
        universe.load_new(system.fn_prod_trj, dt=1)
        snapshot_files = write_snapshot_xyz_files(
            universe,
            snapshots_dir=snapshot_xyz_dir,
            n_snapshots=n_snapshots,
        )

    make_cp2k_input(
        "sp",
        system.charge,
        system.multiplicity,
        system.box[:3].astype(float).tolist(),
        snapshot_files[0],
        snapshots_dir / "sp.inp",
        kind="single_point",
        coord_filename="md-pos-1.xyz",
    )
    make_cp2k_input(
        "md",
        system.charge,
        system.multiplicity,
        system.box[:3].astype(float).tolist(),
        snapshot_files[0],
        snapshots_dir / "md.inp",
        kind="xtb_md",
        steps=CP2K_SNAPSHOT_MD_STEPS,
        coord_filename="pos.xyz",
    )
    logger.done(
        "Reference assets",
        detail=f"{system_dir.resolve()} ({len(snapshot_files)} snapshots)",
        level=2,
    )
