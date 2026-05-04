import shutil
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis import transformations as trans

from ...domain.bias import BiasSpec
from ...io.logs import Logger
from ...io.plumed import ensure_plumed_kernel
from ...io.utils import save_yaml
from ...topology import create_box
from .._shared.preparation import (
    check_gmx_available,
    determine_maxwarn,
    get_average_box,
    make_ndx,
    run_md,
    system_name,
    system_run_name,
    topology_name,
)
from .config import BuildConfig

PathLike = str | Path


@dataclass(slots=True)
class EquilibratedTopology:
    fn_topol_processed: Path
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
            fn_topol_processed=fn_topol_processed,
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
        fn_topol_processed=fn_topol_processed,
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

    fn_gmx_log = project_dir / "gmx.log"
    fn_gmx_log.parent.mkdir(parents=True, exist_ok=True)
    fn_manifest = project_dir / "build-manifest.yaml"

    logger = Logger(
        "build",
        str(config.fn_log) if config.fn_log else None,
        mode="w",
    )
    logger.section(f"Build: {project_dir.name}")
    logger.kv("Config", Path(fn_config).resolve())
    logger.kv("Project directory", project_dir)
    logger.kv("Equilibration directory", equilibration_dir)
    logger.kv("Build manifest", fn_manifest)
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

    def rel(path: Path) -> str:
        return str(path.resolve().relative_to(project_dir))

    equilibrated_topologies: dict[str, EquilibratedTopology] = {}
    manifest_systems: list[dict[str, object]] = []

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
        fn_topol_local = equilibration_dir / f"{system_label}.top"
        fn_coord = equilibration_dir / f"{system_label}.gro"
        fn_ndx = equilibration_dir / f"{system_label}.ndx"
        fn_mdp_em = equilibration_dir / f"{system_label}.em.mdp"
        fn_mdp_npt = equilibration_dir / f"{system_label}.npt.mdp"
        fn_mdp_prod = equilibration_dir / f"{system_label}.mdp"

        shutil.copy2(topology_state.fn_topol_processed, fn_topol_local)
        shutil.copy2(system.fn_mdp_em, fn_mdp_em)
        shutil.copy2(system.fn_mdp_npt, fn_mdp_npt)
        shutil.copy2(system.fn_mdp_prod, fn_mdp_prod)

        with mda.Writer(fn_coord, "w") as writer:
            ts = topology_state.universe.trajectory[-1]
            ts.dimensions = topology_state.box
            writer.write(topology_state.universe.atoms)
        make_ndx(topology_state.universe, None, fn_out=fn_ndx)

        fn_bias_input = None
        for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
            stale = equilibration_dir / f"{system_label}.{suffix}"
            if stale.exists():
                stale.unlink()
        if (
            system.bias.input_file is not None
            and system.bias.input_filename is not None
        ):
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

        deffnm_prod = equilibration_dir / system_run_name(i)
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

        manifest_systems.append(
            {
                "system_id": f"{i:03d}",
                "topology": rel(fn_topol_local),
                "coordinates": rel(fn_coord),
                "index": rel(fn_ndx),
                "mdp": {
                    "em": rel(fn_mdp_em),
                    "npt": rel(fn_mdp_npt),
                    "prod": rel(fn_mdp_prod),
                },
                "bias": None if fn_bias_input is None else rel(fn_bias_input),
                "charge": system.charge,
                "multiplicity": system.mult,
                "box": topology_state.box.tolist(),
                "maxwarn": topology_state.maxwarn,
                "production": {
                    "coordinates": rel(deffnm_prod.with_suffix(".gro")),
                    "trajectory": rel(deffnm_prod.with_suffix(".xtc")),
                    "n_steps": system.nsteps_prod,
                },
            }
        )
        logger.blank()

    save_yaml(
        {
            "version": 1,
            "gmx_cmd": config.gmx_cmd,
            "systems": manifest_systems,
        },
        fn_manifest,
    )
    logger.done("Build manifest", detail=str(fn_manifest), level=1)
