import shutil
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis import transformations as trans

from ...domain.bias import BiasSpec
from ...domain.systems import BuildSystemMetadata, write_build_system_metadata
from ...io.logs import Logger
from ...io.plumed import ensure_plumed_kernel
from ...topology import create_box
from .._shared.preparation import (
    check_gmx_available,
    determine_maxwarn,
    get_average_box,
    make_ndx,
    run_md,
    topology_name,
    write_reference_system,
)
from .config import BuildConfig

PathLike = str | Path


@dataclass(slots=True)
class EquilibratedTopology:
    topology_path: Path
    universe: mda.Universe
    box: np.ndarray
    maxwarn: int


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
    if maxwarn > 0:
        logger.warn(
            "Non-neutral topology detected; GROMACS preprocessing will use "
            "-maxwarn 1.",
            level=2,
        )

    deffnm_em = equilibration_dir / f"{topology_label}-em"
    logger.status("Energy minimization", "in progress...", overwrite=True, level=2)
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
        em_universe = mda.Universe(
            deffnm_em.with_suffix(".gro"),
            to_guess=("elements", "masses"),
        )
        return EquilibratedTopology(
            topology_path=fn_topol_processed,
            universe=em_universe,
            box=np.asarray(em_universe.dimensions, dtype=float),
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
        topology_path=fn_topol_processed,
        universe=universe,
        box=box_avg,
        maxwarn=maxwarn,
    )


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
    fn_gmx_log.parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(
        "build",
        str(config.fn_log) if config.fn_log else None,
        mode="w",
    )
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

    equilibrated_topologies: dict[str, EquilibratedTopology] = {}

    n_total = len(config.systems)
    for i, system in enumerate(config.systems):
        logger.info(
            f"System {i + 1}/{n_total}: {system.system_id}", level=1
        )

        topology_key = str(system.topology_path)
        if topology_key not in equilibrated_topologies:
            topol_index = next(
                j
                for j, candidate in enumerate(config.systems)
                if candidate.topology_path == system.topology_path
            )
            equilibrated_topologies[topology_key] = build_equilibrated_topology(
                topol_index=topol_index,
                fn_topol=system.topology_path,
                templates=system.templates,
                box=system.box,
                fn_mdp_em=system.mdp_em_path,
                fn_mdp_npt=system.mdp_npt_path,
                nsteps_npt=system.nsteps_npt,
                equilibration_dir=equilibration_dir,
                gmx_cmd=config.gmx_cmd,
                fn_gmx_log=fn_gmx_log,
                logger=logger,
            )

        topology_state = equilibrated_topologies[topology_key]
        system_dir = systems_dir / system.system_id
        system_dir.mkdir(parents=True, exist_ok=True)
        fn_topol_local = system_dir / "topology.top"
        fn_coord = system_dir / "coordinates.gro"
        fn_ndx = system_dir / "index.ndx"
        fn_mdp_em = system_dir / "em.mdp"
        fn_mdp_npt = system_dir / "npt.mdp"
        fn_mdp_prod = system_dir / "production.mdp"

        shutil.copy2(topology_state.topology_path, fn_topol_local)
        shutil.copy2(system.mdp_em_path, fn_mdp_em)
        shutil.copy2(system.mdp_npt_path, fn_mdp_npt)
        shutil.copy2(system.mdp_production_path, fn_mdp_prod)

        with mda.Writer(fn_coord, "w") as writer:
            ts = topology_state.universe.trajectory[-1]
            ts.dimensions = topology_state.box
            writer.write(topology_state.universe.atoms)
        make_ndx(topology_state.universe, None, fn_out=fn_ndx)

        fn_bias_input = None
        for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
            stale = system_dir / suffix
            if stale.exists():
                stale.unlink()
        if (
            system.bias.input_file is not None
            and system.bias.input_filename is not None
        ):
            fn_bias_input = (
                system_dir / system.bias.input_filename
            )
            shutil.copy2(system.bias.input_file, fn_bias_input)

        if fn_bias_input is None or not system.bias.is_biased:
            bias_run = system.bias
        elif system.bias.kind == "colvars":
            bias_run = BiasSpec(kind="colvars", colvars_file=fn_bias_input)
        else:
            bias_run = BiasSpec(kind="plumed", plumed_file=fn_bias_input)

        deffnm_prod = system_dir / "production"
        logger.status("Production seed run", "in progress...", overwrite=True, level=2)
        run_md(
            deffnm_prod,
            fn_mdp_prod,
            fn_topol_local,
            fn_coord,
            fn_ndx,
            bias=bias_run,
            gmx_cmd=config.gmx_cmd,
            n_steps=system.nsteps_prod,
            maxwarn=topology_state.maxwarn,
            fn_log=fn_gmx_log,
        )
        logger.done("Production seed run", level=2)

        reference_dir = system_dir / "reference"
        removed_virtual_sites = write_reference_system(
            fn_topol_local,
            deffnm_prod.with_suffix(".gro"),
            reference_dir / "topology.top",
            reference_dir / "coordinates.gro",
        )
        logger.done(
            "Reference-compatible system",
            detail=f"removed {removed_virtual_sites} virtual sites",
            level=2,
        )

        write_build_system_metadata(
            project_dir,
            system.system_id,
            BuildSystemMetadata(
                system_name=system.system_name,
                charge=system.charge,
                multiplicity=system.mult,
                box=tuple(float(value) for value in topology_state.box),
                maxwarn=topology_state.maxwarn,
                production_steps=system.nsteps_prod,
            ),
        )
        logger.done("System metadata", detail=str(system_dir / "system.yaml"), level=2)
        logger.blank()
