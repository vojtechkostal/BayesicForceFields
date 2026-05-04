from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis.selections.gromacs import SelectionWriter

from ...domain.bias import BiasSpec
from ...io.colvars import write_mdp_with_colvars
from ...io.commands import build_command
from ...io.extxyz import write_extxyz_frame
from ...io.plumed import ensure_plumed_kernel
from ...io.utils import load_yaml
from .config import _resolve_path

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class PreparedSystem:
    system_id: str
    charge: int
    multiplicity: int
    box: np.ndarray
    fn_topol: Path
    fn_coordinates: Path
    fn_mdp_em: Path
    fn_mdp_npt: Path
    fn_mdp_prod: Path
    fn_ndx: Path
    fn_prod_coord: Path
    fn_prod_trj: Path
    bias: BiasSpec
    nsteps_prod: int
    maxwarn: int


@dataclass(frozen=True, slots=True)
class BuildManifest:
    fn_manifest: Path
    gmx_cmd: str
    systems: list[PreparedSystem]

    @classmethod
    def load(cls, fn_manifest: PathLike) -> "BuildManifest":
        fn_manifest = Path(fn_manifest).resolve()
        data = load_yaml(fn_manifest)
        if not isinstance(data, dict):
            raise ValueError("Build manifest must contain a mapping.")
        if data.get("version") != 1:
            raise ValueError("Unsupported build manifest version.")

        base_dir = fn_manifest.parent
        systems_raw = data.get("systems")
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("Build manifest must contain non-empty 'systems'.")

        systems: list[PreparedSystem] = []
        for index, system in enumerate(systems_raw):
            if not isinstance(system, dict):
                raise ValueError(
                    f"systems[{index}] in build manifest must be a mapping."
                )
            mdp = system.get("mdp")
            production = system.get("production")
            if not isinstance(mdp, dict):
                raise ValueError(f"systems[{index}].mdp must be a mapping.")
            if not isinstance(production, dict):
                raise ValueError(f"systems[{index}].production must be a mapping.")

            system_id = normalize_system_id(system.get("system_id", f"{index:03d}"))
            systems.append(
                PreparedSystem(
                    system_id=system_id,
                    charge=int(system["charge"]),
                    multiplicity=int(system["multiplicity"]),
                    box=np.asarray(system["box"], dtype=float),
                    fn_topol=_resolve_path(
                        base_dir,
                        system["topology"],
                        kind=f"manifest system {system_id} topology",
                    ),
                    fn_coordinates=_resolve_path(
                        base_dir,
                        system["coordinates"],
                        kind=f"manifest system {system_id} coordinates",
                    ),
                    fn_mdp_em=_resolve_path(
                        base_dir,
                        mdp["em"],
                        kind=f"manifest system {system_id} EM MDP",
                    ),
                    fn_mdp_npt=_resolve_path(
                        base_dir,
                        mdp["npt"],
                        kind=f"manifest system {system_id} NpT MDP",
                    ),
                    fn_mdp_prod=_resolve_path(
                        base_dir,
                        mdp["prod"],
                        kind=f"manifest system {system_id} production MDP",
                    ),
                    fn_ndx=_resolve_path(
                        base_dir,
                        system["index"],
                        kind=f"manifest system {system_id} index",
                    ),
                    fn_prod_coord=_resolve_path(
                        base_dir,
                        production["coordinates"],
                        kind=f"manifest system {system_id} production coordinates",
                    ),
                    fn_prod_trj=_resolve_path(
                        base_dir,
                        production["trajectory"],
                        kind=f"manifest system {system_id} production trajectory",
                    ),
                    bias=BiasSpec.from_any(system.get("bias"), base_dir=base_dir),
                    nsteps_prod=int(production["n_steps"]),
                    maxwarn=int(system.get("maxwarn", 0)),
                )
            )

        return cls(
            fn_manifest=fn_manifest,
            gmx_cmd=str(data.get("gmx_cmd", "gmx")),
            systems=systems,
        )


def system_name(system_index: int | str) -> str:
    return f"system-{int(system_index):03d}"


def normalize_system_id(value: int | str) -> str:
    text = str(value)
    if text.startswith("system-"):
        text = text.split("-", maxsplit=1)[1]
    return f"{int(text):03d}"


def select_prepared_systems(
    systems: list[PreparedSystem],
    system_ids: tuple[str, ...] | None,
) -> list[PreparedSystem]:
    if system_ids is None:
        return list(systems)

    by_id = {system.system_id: system for system in systems}
    missing = [system_id for system_id in system_ids if system_id not in by_id]
    if missing:
        raise ValueError(
            "Build manifest does not contain requested system id(s): "
            + ", ".join(missing)
        )
    return [by_id[system_id] for system_id in system_ids]


def load_manifest_system_ids(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("'systems' must be a non-empty list when provided.")

    system_ids: list[str] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            if "system_id" not in item:
                raise ValueError(f"systems[{index}] must define 'system_id'.")
            item = item["system_id"]
        if not isinstance(item, (int, str)):
            raise ValueError(f"systems[{index}] must be a system id.")
        system_ids.append(normalize_system_id(item))
    return tuple(system_ids)


def topology_name(topology_index: int) -> str:
    return f"topology-{topology_index:03d}"


def system_run_name(system_index: int | str) -> str:
    return f"{system_name(system_index)}-prod"


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
        write_mdp_with_colvars(fn_mdp_path, fn_bias_local, fn_mdp_run)
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


def strip_topol(
    fn_topol: PathLike,
    fn_coords: PathLike,
    fn_out_topol: PathLike,
    *fn_out_coords: PathLike,
) -> None:
    top = Topology(fn_topol)
    universe = mda.Universe(fn_topol, fn_coords, topology_format="ITP")

    for mol, _ in top.molecules.values():
        mol.remove_vsites()
    top.write(fn_out_topol, overwrite=True)

    atoms = universe.select_atoms("not mass -1 to 0.5")
    ts = universe.trajectory[-1]
    for fn_out in fn_out_coords:
        fn_out = Path(fn_out)
        if fn_out.suffix.lower() == ".xyz":
            write_extxyz_frame(atoms, fn_out, dimensions=ts.dimensions)
        else:
            atoms.write(fn_out, frames=universe.trajectory[[-1]])


def sample_snapshot_indices(n_frames: int, n_snapshots: int) -> np.ndarray:
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
