"""Workflow entry point for staged CP2K reference calculations."""

from __future__ import annotations

import os
import random
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ..io.logs import Logger
from ..io.utils import load_yaml, save_yaml
from .configs import ReferenceConfig, ReferenceSystemConfig
from .runsims import (
    bff_cli_command,
    build_slurm_cli_job,
    control_jobs,
    wait_for_scheduler_slot,
)

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
ENERGY_RE = re.compile(
    r"ENERGY\| Total FORCE_EVAL .*?([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*$",
    re.MULTILINE,
)
FORCE_RE = re.compile(
    r"^\s*FORCES\|\s+\d+\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s+"
    r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?\s*$",
    re.MULTILINE,
)
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


@dataclass(frozen=True)
class ReferenceJobConfig:
    kind: Literal["snapshot", "single_atom"]
    run_dir: Path
    cp2k_cmd: str

    @classmethod
    def load(cls, fn_config: str | Path) -> "ReferenceJobConfig":
        data = load_yaml(fn_config)
        if not isinstance(data, dict):
            raise ValueError("Reference job config must contain a mapping.")

        kind = data.get("kind")
        if kind not in {"snapshot", "single_atom"}:
            raise ValueError(
                "Reference job config 'kind' must be 'snapshot' or "
                "'single_atom'."
            )

        run_dir = Path(data.get("run_dir", "")).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Reference job directory not found: {run_dir}")

        cp2k_cmd = str(data.get("cp2k_cmd", "cp2k.psmp"))
        parts = shlex.split(cp2k_cmd)
        if len(parts) != 1:
            raise ValueError("'cp2k_cmd' must be a single executable name or path.")

        return cls(kind=kind, run_dir=run_dir, cp2k_cmd=cp2k_cmd)


def check_cp2k_available(cp2k_cmd: str) -> str:
    """Validate the local CP2K executable name or path."""
    parts = shlex.split(cp2k_cmd)
    if len(parts) != 1:
        raise ValueError(
            "'cp2k_cmd' must be a single executable name or path for local runs."
        )

    executable = parts[0]
    if "/" in executable:
        resolved = Path(executable).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"CP2K executable not found: {resolved}")
        return str(resolved)

    resolved = shutil.which(executable)
    if resolved is None:
        raise FileNotFoundError(
            f"CP2K executable '{executable}' was not found on PATH."
        )
    return resolved


def print_reference_summary(config: ReferenceConfig, logger: Logger) -> None:
    """Print a concise reference-workflow summary."""
    logger.section("Reference Data Generation")
    logger.kv("Reference directory", config.reference_dir.resolve())
    logger.kv("Systems", len(config.systems))
    logger.kv("Scheduler", config.job_scheduler)
    logger.kv("CP2K command", config.cp2k_cmd)
    logger.kv("Single-atom energies", "yes" if config.single_atoms else "no")
    logger.kv(
        "Snapshot MD steps",
        (
            config.snapshot_md_steps
            if config.snapshot_md_steps is not None
            else "from staged md.inp"
        ),
    )
    logger.kv("Train fraction", config.train_fraction)
    logger.kv("Shuffle seed", config.seed)
    if config.job_scheduler == "slurm" and config.slurm is not None:
        logger.kv("Max parallel jobs", config.slurm.max_parallel_jobs)
    logger.warn_if(
        not config.single_atoms,
        "Single-atom reference energies are disabled for this run.",
    )
    logger.blank()


def _write_snapshot_md_input(
    *,
    src: Path,
    dst: Path,
    steps: int | None,
) -> None:
    """Copy one staged short-MD input and optionally override its step count."""
    text = src.read_text(encoding="utf-8")
    if steps is None:
        dst.write_text(text, encoding="utf-8")
        return

    updated, n_replaced = re.subn(
        r"(?m)^(\s*STEPS\s+)\d+\s*$",
        rf"\g<1>{steps}",
        text,
        count=1,
    )
    if n_replaced != 1:
        raise ValueError(f"Could not override STEPS in staged CP2K input {src}.")
    dst.write_text(updated, encoding="utf-8")


def stage_system(
    system: ReferenceSystemConfig,
    config: ReferenceConfig,
) -> tuple[Path, list[Path], list[Path]]:
    """Stage one prepared reference system into the output tree."""
    system_dir = config.reference_dir.resolve() / system.assets_dir.name
    system_dir.mkdir(parents=True, exist_ok=True)
    for stale in (system_dir / "train.extxyz", system_dir / "valid.extxyz"):
        if stale.exists():
            stale.unlink()

    shutil.copy2(system.fn_system_top, system_dir / "system.top")
    shutil.copy2(system.fn_system_gro, system_dir / "system.gro")
    shutil.copy2(system.fn_system_xyz, system_dir / "system.xyz")

    snapshot_files = sorted(system.snapshot_xyz_dir.glob("snapshot-*.xyz"))
    if not snapshot_files:
        raise FileNotFoundError(
            f"No snapshot XYZ files found in {system.snapshot_xyz_dir}"
        )

    snapshots_dir = system_dir / "snapshots"
    if snapshots_dir.exists():
        shutil.rmtree(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_setup_env = system.snapshots_dir / "setup-env.sh"
    if not snapshot_setup_env.exists():
        snapshot_setup_env = None

    snapshot_run_dirs: list[Path] = []
    for xyz in snapshot_files:
        run_dir = snapshots_dir / xyz.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_snapshot_md_input(
            src=system.fn_snapshot_md,
            dst=run_dir / "md.inp",
            steps=config.snapshot_md_steps,
        )
        shutil.copy2(system.fn_snapshot_sp, run_dir / "sp.inp")
        shutil.copy2(xyz, run_dir / "pos.xyz")
        if snapshot_setup_env is not None:
            shutil.copy2(snapshot_setup_env, run_dir / "setup-env.sh")
        snapshot_run_dirs.append(run_dir)

    single_atom_run_dirs: list[Path] = []
    if config.single_atoms:
        single_atoms_dir = system_dir / "single-atoms"
        if single_atoms_dir.exists():
            shutil.rmtree(single_atoms_dir)
        single_atoms_dir.mkdir(parents=True, exist_ok=True)

        atom_dirs = sorted(
            path for path in system.single_atoms_dir.iterdir() if path.is_dir()
        )
        if not atom_dirs:
            raise FileNotFoundError(
                "No isolated-atom directories found in "
                f"{system.single_atoms_dir}"
            )

        single_atom_setup_env = system.single_atoms_dir / "setup-env.sh"
        if not single_atom_setup_env.exists():
            single_atom_setup_env = system.assets_dir / "setup-env.sh"
        if not single_atom_setup_env.exists():
            single_atom_setup_env = None

        for atom_dir in atom_dirs:
            run_dir = single_atoms_dir / atom_dir.name
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(atom_dir / "input.inp", run_dir / "input.inp")
            shutil.copy2(atom_dir / "pos.xyz", run_dir / "pos.xyz")
            if single_atom_setup_env is not None:
                shutil.copy2(single_atom_setup_env, run_dir / "setup-env.sh")
            single_atom_run_dirs.append(run_dir)

    return system_dir, snapshot_run_dirs, single_atom_run_dirs


def _run_cp2k(
    *,
    cp2k_cmd: str,
    fn_input: str,
    fn_output: str,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    command = [cp2k_cmd, "-i", fn_input, "-o", fn_output]
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun") is not None:
        command = ["srun", *command]
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _last_xyz_frame(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not a valid XYZ trajectory file.")

    n_atoms = int(lines[0].strip())
    frame_len = n_atoms + 2
    if len(lines) < frame_len:
        raise ValueError(f"{path} does not contain a complete XYZ frame.")

    last_frame = lines[-frame_len:]
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in last_frame[2:]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line in {path}: {line}")
        symbols.append(parts[0])
        positions.append([float(value) for value in parts[1:4]])
    return symbols, positions


def _parse_single_point_energy(fn_output: Path) -> float:
    match = ENERGY_RE.search(fn_output.read_text(encoding="utf-8", errors="ignore"))
    if match is None:
        raise ValueError(f"Could not extract single-point energy from {fn_output}.")
    return float(match.group(1)) * HARTREE_TO_EV


def _parse_single_point_forces(fn_output: Path) -> list[list[float]]:
    text = fn_output.read_text(encoding="utf-8", errors="ignore")
    forces = [
        [
            float(match.group(1)) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            float(match.group(2)) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            float(match.group(3)) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        ]
        for match in FORCE_RE.finditer(text)
    ]
    if not forces:
        raise ValueError(f"Could not extract atomic forces from {fn_output}.")
    return forces


def _snapshot_comment(fn_snapshot: Path) -> str:
    lines = fn_snapshot.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{fn_snapshot} is not a valid extxyz snapshot.")
    return lines[1].strip()


def write_snapshot_extxyz(run_dir: Path) -> Path:
    """Combine final MD positions and single-point forces into one extxyz frame."""
    comment = _snapshot_comment(run_dir / "pos.xyz")
    symbols, positions = _last_xyz_frame(run_dir / "md-pos-1.xyz")
    forces = _parse_single_point_forces(run_dir / "sp.out")
    energy = _parse_single_point_energy(run_dir / "sp.out")

    if len(forces) != len(symbols):
        raise ValueError(
            f"Force extraction failed: expected {len(symbols)} rows, got {len(forces)}."
        )

    comment_parts = []
    for pattern in (r'Lattice="[^"]*"', r'pbc="[^"]*"'):
        match = re.search(pattern, comment)
        if match is not None:
            comment_parts.append(match.group(0))
    comment_parts.append("Properties=species:S:1:pos:R:3:forces:R:3")
    comment_parts.append(f"energy={energy:.16g}")
    comment_parts.append('source="sp"')

    fn_extxyz = run_dir / "sp.extxyz"
    with fn_extxyz.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(" ".join(comment_parts) + "\n")
        for symbol, position, force in zip(symbols, positions, forces, strict=True):
            values = [*position, *force]
            handle.write(
                symbol + " " + " ".join(f"{value:.16g}" for value in values) + "\n"
            )
    return fn_extxyz


def run_snapshot_job(run_dir: Path, cp2k_cmd: str) -> None:
    """Run one snapshot short-MD plus single-point reference job."""
    env = os.environ.copy()
    env["CP2K_CMD"] = cp2k_cmd
    _run_cp2k(
        cp2k_cmd=cp2k_cmd,
        fn_input="md.inp",
        fn_output="md.out",
        cwd=run_dir,
        env=env,
    )
    _run_cp2k(
        cp2k_cmd=cp2k_cmd,
        fn_input="sp.inp",
        fn_output="sp.out",
        cwd=run_dir,
        env=env,
    )
    write_snapshot_extxyz(run_dir)


def _parse_single_atom_energy(fn_output: Path) -> float:
    match = ENERGY_RE.search(fn_output.read_text(encoding="utf-8", errors="ignore"))
    if match is None:
        raise ValueError(f"Could not extract isolated-atom energy from {fn_output}.")
    return float(match.group(1)) * HARTREE_TO_EV


def run_single_atom_job(run_dir: Path, cp2k_cmd: str) -> None:
    """Run one isolated-atom single-point job."""
    env = os.environ.copy()
    env["CP2K_CMD"] = cp2k_cmd
    _run_cp2k(
        cp2k_cmd=cp2k_cmd,
        fn_input="input.inp",
        fn_output="atom.out",
        cwd=run_dir,
        env=env,
    )


def submit_reference_job(
    *,
    kind: Literal["snapshot", "single_atom"],
    run_dir: Path,
    config: ReferenceConfig,
    job_name: str,
    script_name: str,
) -> int:
    """Write one hidden job config and submit it through Slurm."""
    assert config.slurm is not None
    fn_job = run_dir / ".bff-job.yaml"
    save_yaml(
        {
            "kind": kind,
            "run_dir": str(run_dir.resolve()),
            "cp2k_cmd": config.cp2k_cmd,
        },
        fn_job,
    )
    submit_specs = dict(config.slurm.sbatch or {})
    submit_specs.setdefault("job_name", job_name)
    submit_specs.setdefault("output", (run_dir / "slurm-%j.out").resolve())
    submit = build_slurm_cli_job(
        command=bff_cli_command("reference-job", fn_job.resolve()),
        slurm_config=config.slurm,
        sbatch=submit_specs,
        cwd=run_dir,
    )
    return submit.submit(run_dir / script_name)


def collect_single_atom_energies(single_atom_dirs: list[Path]) -> dict[str, float]:
    """Parse isolated-atom energies from finished single-point outputs."""
    energies: dict[str, float] = {}
    for atom_dir in sorted(single_atom_dirs):
        fn_output = atom_dir / "atom.out"
        if not fn_output.exists():
            raise FileNotFoundError(f"Missing isolated-atom output file: {fn_output}")
        energy = _parse_single_atom_energy(fn_output)
        lines = (atom_dir / "pos.xyz").read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            raise ValueError(f"Invalid isolated-atom XYZ file: {atom_dir / 'pos.xyz'}")
        symbol = lines[2].split()[0]
        energies[symbol] = energy
    return energies


def _read_extxyz_properties(comment: str) -> tuple[list[float] | None, float | None]:
    lattice = None
    match = LATTICE_RE.search(comment)
    if match is not None:
        values = [float(value) for value in match.group(1).split()]
        if len(values) != 9:
            raise ValueError("Invalid Lattice field in extxyz comment.")
        lattice = values

    energy = None
    match = re.search(r"\benergy=([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", comment)
    if match is not None:
        energy = float(match.group(1))
    return lattice, energy


def _read_extxyz_frame(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not a valid extxyz file.")

    n_atoms = int(lines[0].strip())
    lattice, energy = _read_extxyz_properties(lines[1])
    atoms: list[str] = []
    positions: list[list[float]] = []
    forces: list[list[float]] = []

    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Invalid extxyz atom line in {path}: {line}")
        atoms.append(parts[0])
        positions.append([float(value) for value in parts[1:4]])
        forces.append([float(value) for value in parts[4:7]])

    return {
        "atoms": atoms,
        "positions": positions,
        "forces": forces,
        "energy": energy,
        "lattice": lattice,
    }


def _format_lattice(lattice: list[float]) -> str:
    values = [0.0 if abs(value) < 5e-5 else value for value in lattice]
    return " ".join(f"{value:.4f}" for value in values)


def _write_extxyz(frames: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            atoms = frame["atoms"]
            positions = frame["positions"]
            forces = frame["forces"]
            energy = frame["energy"]

            handle.write(f"{len(atoms)}\n")
            header: list[str] = []
            lattice = frame.get("lattice")
            if lattice is not None:
                header.append(f'Lattice="{_format_lattice(lattice)}"')
                header.append('pbc="T T T"')
            header.append("Properties=species:S:1:pos:R:3:forces:R:3")
            header.append(f"energy={energy:.16g}")
            header.append(f'source="{frame["source"]}"')
            handle.write(" ".join(header) + "\n")

            for atom, position, force in zip(atoms, positions, forces, strict=True):
                values = [*position, *force]
                handle.write(
                    atom + " " + " ".join(f"{value:.12g}" for value in values) + "\n"
                )


def _split_train_valid(
    frames: list[dict[str, object]],
    train_fraction: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not frames:
        return [], []
    if len(frames) == 1:
        return frames, []

    n_train = round(len(frames) * train_fraction)
    n_train = min(max(n_train, 1), len(frames) - 1)
    return frames[:n_train], frames[n_train:]


def _collect_frame(run_dir: Path) -> dict[str, object]:
    fn_extxyz = run_dir / "sp.extxyz"
    if not fn_extxyz.exists():
        raise FileNotFoundError(f"Missing collected extxyz file: {fn_extxyz}")

    frame = _read_extxyz_frame(fn_extxyz)
    if frame["energy"] is None:
        raise ValueError(f"{fn_extxyz} is missing energy metadata.")
    frame["source"] = run_dir.name
    return frame


def collect_snapshot_splits(
    snapshot_run_dirs: list[Path],
    system_dir: Path,
    config: ReferenceConfig,
    logger: Logger,
) -> None:
    frames: list[dict[str, object]] = []
    for run_dir in sorted(snapshot_run_dirs):
        try:
            frames.append(_collect_frame(run_dir))
        except (FileNotFoundError, ValueError) as exc:
            logger.warn(f"Skipping {run_dir.name}: {exc}", level=2)

    random.Random(config.seed).shuffle(frames)
    train_frames, valid_frames = _split_train_valid(frames, config.train_fraction)
    _write_extxyz(train_frames, system_dir / "train.extxyz")
    _write_extxyz(valid_frames, system_dir / "valid.extxyz")

    n_train = len(train_frames)
    n_valid = len(valid_frames)
    n_total = len(snapshot_run_dirs)
    n_collected = n_train + n_valid
    logger.done(
        "Snapshot collection",
        detail=(
            f"{n_collected}/{n_total} frames -> "
            f"train {n_train}, valid {n_valid}"
        ),
        level=2,
    )
    logger.warn_if(
        n_collected != n_total,
        "Only "
        f"{n_collected} of {n_total} snapshot runs produced usable "
        "sp.extxyz files.",
        level=2,
    )


def collect_system_outputs(
    system_dir: Path,
    config: ReferenceConfig,
    logger: Logger,
) -> None:
    snapshot_run_dirs = sorted(
        path for path in (system_dir / "snapshots").iterdir() if path.is_dir()
    )
    collect_snapshot_splits(snapshot_run_dirs, system_dir, config, logger)

    if not config.single_atoms:
        return

    single_atom_dirs = sorted(
        path for path in (system_dir / "single-atoms").iterdir() if path.is_dir()
    )
    energies = collect_single_atom_energies(single_atom_dirs)
    save_yaml(energies, system_dir / "single-atoms" / "energies.yaml")
    logger.done(
        "Single-atom energies",
        detail=", ".join(f"{name}={value:.6f} eV" for name, value in energies.items()),
        level=2,
    )


def run_system_local(
    system: ReferenceSystemConfig,
    config: ReferenceConfig,
    logger: Logger,
) -> None:
    system_dir, snapshot_run_dirs, single_atom_dirs = stage_system(system, config)
    logger.kv("Prepared assets", system.assets_dir.resolve(), level=2)
    logger.kv("Output directory", system_dir.resolve(), level=2)
    logger.kv("Snapshots", len(snapshot_run_dirs), level=2)

    for index, run_dir in enumerate(snapshot_run_dirs, start=1):
        logger.status(
            "Snapshot jobs",
            f"{index}/{len(snapshot_run_dirs)}",
            detail=run_dir.name,
            level=2,
            overwrite=True,
        )
        run_snapshot_job(run_dir, config.cp2k_cmd)
    logger.done(
        "Snapshot jobs",
        detail=f"{len(snapshot_run_dirs)}/{len(snapshot_run_dirs)}",
        level=2,
    )

    if not config.single_atoms:
        collect_system_outputs(system_dir, config, logger)
        return

    for index, atom_dir in enumerate(single_atom_dirs, start=1):
        logger.status(
            "Single-atom jobs",
            f"{index}/{len(single_atom_dirs)}",
            detail=atom_dir.name,
            level=2,
            overwrite=True,
        )
        run_single_atom_job(atom_dir, config.cp2k_cmd)

    logger.done(
        "Single-atom jobs",
        detail=f"{len(single_atom_dirs)}/{len(single_atom_dirs)}",
        level=2,
    )
    collect_system_outputs(system_dir, config, logger)


def submit_system_slurm(
    system: ReferenceSystemConfig,
    config: ReferenceConfig,
    logger: Logger,
    job_ids: list[int],
) -> None:
    assert config.slurm is not None
    system_dir, snapshot_run_dirs, single_atom_dirs = stage_system(system, config)
    max_parallel_jobs = config.slurm.max_parallel_jobs

    logger.kv("Prepared assets", system.assets_dir.resolve(), level=2)
    logger.kv("Output directory", system_dir.resolve(), level=2)
    logger.kv("Snapshots", len(snapshot_run_dirs), level=2)

    for index, run_dir in enumerate(snapshot_run_dirs, start=1):
        if max_parallel_jobs != -1:
            wait_for_scheduler_slot(
                job_ids=job_ids,
                scheduler="slurm",
                max_parallel_jobs=max_parallel_jobs,
            )
        logger.status(
            "Submitting snapshot jobs",
            f"{index}/{len(snapshot_run_dirs)}",
            detail=run_dir.name,
            level=2,
            overwrite=True,
        )
        job_id = submit_reference_job(
            kind="snapshot",
            run_dir=run_dir,
            config=config,
            job_name=f"bff-ref-{system.system_id}",
            script_name=".bff-snapshot.sbatch.sh",
        )
        job_ids.append(job_id)

    logger.done(
        "Submitting snapshot jobs",
        detail=f"{len(snapshot_run_dirs)}/{len(snapshot_run_dirs)}",
        level=2,
    )

    if not config.single_atoms:
        return

    for index, atom_dir in enumerate(single_atom_dirs, start=1):
        if max_parallel_jobs != -1:
            wait_for_scheduler_slot(
                job_ids=job_ids,
                scheduler="slurm",
                max_parallel_jobs=max_parallel_jobs,
            )
        logger.status(
            "Submitting single-atom jobs",
            f"{index}/{len(single_atom_dirs)}",
            detail=atom_dir.name,
            level=2,
            overwrite=True,
        )
        job_id = submit_reference_job(
            kind="single_atom",
            run_dir=atom_dir,
            config=config,
            job_name=f"bff-atom-{system.system_id}",
            script_name=".bff-single-atom.sbatch.sh",
        )
        job_ids.append(job_id)

    logger.done(
        "Submitting single-atom jobs",
        detail=f"{len(single_atom_dirs)}/{len(single_atom_dirs)}",
        level=2,
    )


def run_job(fn_config: str | Path) -> None:
    """Run one staged snapshot or isolated-atom reference job."""
    job = ReferenceJobConfig.load(fn_config)
    if job.kind == "snapshot":
        run_snapshot_job(job.run_dir, job.cp2k_cmd)
        return
    run_single_atom_job(job.run_dir, job.cp2k_cmd)


def main(fn_config: str) -> None:
    """Run staged CP2K reference calculations locally or through Slurm."""
    config = ReferenceConfig.load(fn_config)
    if config.job_scheduler == "local":
        config = replace(config, cp2k_cmd=check_cp2k_available(config.cp2k_cmd))

    config.reference_dir.resolve().mkdir(parents=True, exist_ok=True)

    logger = Logger("reference")
    print_reference_summary(config, logger)

    if config.job_scheduler == "local":
        for index, system in enumerate(config.systems, start=1):
            logger.info(f"System {index}/{len(config.systems)}", level=1)
            run_system_local(system, config, logger)
            logger.blank()
        return

    job_ids: list[int] = []
    for index, system in enumerate(config.systems, start=1):
        logger.info(f"System {index}/{len(config.systems)}", level=1)
        submit_system_slurm(system, config, logger, job_ids)
        logger.blank()

    if job_ids:
        logger.status("Waiting for Slurm jobs", f"{len(job_ids)} submitted", level=1)
        control_jobs(job_ids, "slurm")
        logger.done("Waiting for Slurm jobs", detail="all jobs finished", level=1)

    for index, system in enumerate(config.systems, start=1):
        logger.info(f"Collecting system {index}/{len(config.systems)}", level=1)
        system_dir = config.reference_dir.resolve() / system.assets_dir.name
        collect_system_outputs(system_dir, config, logger)
        logger.blank()
