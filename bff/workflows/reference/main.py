"""Workflow entry point for canonical reference-data generation."""

from __future__ import annotations

import os
import random
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ...io.cp2k import (
    collect_single_atom_energies,
    cp2k_supports_gfn_type,
    strip_cp2k_gfn_type,
    write_cp2k_snapshot_extxyz,
)
from ...io.extxyz import read_extxyz_frame, write_extxyz_frames
from ...io.logs import Logger
from ...io.utils import load_yaml, save_yaml
from .._shared.scheduler import (
    bff_cli_command,
    build_slurm_cli_job,
    control_jobs,
    wait_for_scheduler_slot,
)
from .config import (
    ImportedReferenceSystemConfig,
    ReferenceConfig,
    ReferenceSystemConfig,
)

ReferenceJobKind = Literal["snapshot", "single_atom"]

REFERENCE_JOBS = {
    "snapshot": {
        "steps": (("md.inp", "md.out"), ("sp.inp", "sp.out")),
        "label": "Snapshot jobs",
        "submit_label": "Submitting snapshot jobs",
        "job_name_prefix": "bff-ref",
        "script_name": ".bff-snapshot.sbatch.sh",
    },
    "single_atom": {
        "steps": (("input.inp", "atom.out"),),
        "label": "Single-atom jobs",
        "submit_label": "Submitting single-atom jobs",
        "job_name_prefix": "bff-atom",
        "script_name": ".bff-single-atom.sbatch.sh",
    },
}


@dataclass(frozen=True)
class ReferenceJobConfig:
    kind: ReferenceJobKind
    run_dir: Path
    cp2k_cmd: str

    @classmethod
    def load(cls, fn_config: str | Path) -> "ReferenceJobConfig":
        data = load_yaml(fn_config)
        if not isinstance(data, dict):
            raise ValueError("Reference job config must contain a mapping.")

        kind = data.get("kind")
        if kind not in REFERENCE_JOBS:
            raise ValueError(
                "Reference job config 'kind' must be 'snapshot' or 'single_atom'."
            )

        run_dir = Path(data.get("run_dir", "")).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Reference job directory not found: {run_dir}")

        cp2k_cmd = str(data.get("cp2k_cmd", "cp2k.psmp"))
        parts = shlex.split(cp2k_cmd)
        if len(parts) != 1:
            raise ValueError("'cp2k_cmd' must be a single executable name or path.")

        executable = parts[0]
        if "/" in executable:
            executable = str(Path(executable).expanduser())

        return cls(kind=kind, run_dir=run_dir, cp2k_cmd=executable)


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
    logger.section("Reference Data")
    logger.kv("Mode", config.mode)
    logger.kv("Reference directory", config.reference_dir.resolve())
    logger.kv("Systems", len(config.systems))
    if config.mode == "import":
        logger.info(
            "Imported trajectories are copied verbatim as canonical system.top, "
            "system.gro, and trajectory.* assets.",
            level=1,
        )
        logger.blank()
        return

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
    logger.kv("Cleanup snapshots", "yes" if config.cleanup_snapshots else "no")
    logger.kv("Collection wait", f"{config.collection_wait_seconds:g} s")
    if config.job_scheduler == "slurm" and config.slurm is not None:
        logger.kv("Max parallel jobs", config.slurm.max_parallel_jobs)
    if not config.single_atoms:
        logger.warn("Single-atom reference energies are disabled for this run.")
    logger.blank()


def import_reference_system(
    system: ImportedReferenceSystemConfig,
    reference_dir: Path,
    logger: Logger,
) -> Path:
    """Copy one externally generated reference trajectory into canonical assets."""
    system_dir = reference_dir / f"system-{system.system_id}"
    system_dir.mkdir(parents=True, exist_ok=True)

    fn_topol = system_dir / "system.top"
    fn_gro = system_dir / "system.gro"
    fn_trj = system_dir / f"trajectory{system.fn_trj.suffix}"
    shutil.copy2(system.fn_topol, fn_topol)
    shutil.copy2(system.fn_gro, fn_gro)
    shutil.copy2(system.fn_trj, fn_trj)
    save_yaml(
        {
            "mode": "import",
            "topology": fn_topol.name,
            "coordinates": fn_gro.name,
            "trajectory": fn_trj.name,
        },
        system_dir / "imported.yaml",
    )
    logger.kv("Output directory", system_dir.resolve(), level=2)
    logger.kv("Imported trajectory", fn_trj.name, level=2)
    return system_dir


def _write_snapshot_md_input(
    *,
    src: Path,
    dst: Path,
    steps: int | None,
) -> None:
    """Copy one staged short-MD input and optionally override its step count."""
    text = src.read_text(encoding="utf-8")
    if steps is not None:
        text, n_replaced = re.subn(
            r"(?m)^(\s*STEPS\s+)\d+\s*$",
            rf"\g<1>{steps}",
            text,
            count=1,
        )
        if n_replaced != 1:
            raise ValueError(f"Could not override STEPS in staged CP2K input {src}.")

    text, n_replaced = re.subn(
        r"(?im)^(\s*BACKUP_COPIES\s+)\d+\s*$",
        r"\g<1>1",
        text,
        count=1,
    )
    if n_replaced == 0:
        text = re.sub(
            r"(?im)^(\s*)&END\s+PRINT\s*$",
            (
                r"\1  &RESTART\n"
                r"\1    BACKUP_COPIES 1\n"
                r"\1  &END RESTART\n"
                r"\g<0>"
            ),
            text,
            count=1,
        )
    dst.write_text(text, encoding="utf-8")


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
    shutil.copy2(system.fn_system_xyz, system_dir / "system.extxyz")

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
) -> None:
    command = [cp2k_cmd, "-i", fn_input, "-o", fn_output]
    if os.environ.get("SLURM_JOB_ID") and shutil.which("srun") is not None:
        command = ["srun", *command]
    subprocess.run(command, cwd=str(cwd), check=True)


def _remove_cp2k_restart_files(run_dir: Path) -> None:
    """Remove large CP2K restart artifacts from one run directory."""
    for pattern in ("*.wfn*", "*.restart*"):
        for path in run_dir.glob(pattern):
            path.unlink()


def run_reference_job(
    kind: ReferenceJobKind,
    run_dir: Path,
    cp2k_cmd: str,
) -> None:
    """Run one staged snapshot or isolated-atom CP2K job."""
    if kind == "snapshot" and cp2k_supports_gfn_type(cp2k_cmd) is False:
        strip_cp2k_gfn_type(run_dir / "md.inp")

    for fn_input, fn_output in REFERENCE_JOBS[kind]["steps"]:
        _run_cp2k(
            cp2k_cmd=cp2k_cmd,
            fn_input=fn_input,
            fn_output=fn_output,
            cwd=run_dir,
        )
    _remove_cp2k_restart_files(run_dir)
    if kind == "snapshot":
        write_cp2k_snapshot_extxyz(run_dir)


def submit_reference_job(
    *,
    kind: ReferenceJobKind,
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

    frame = read_extxyz_frame(fn_extxyz)
    if frame["energy"] is None:
        raise ValueError(f"{fn_extxyz} is missing energy metadata.")
    if frame["forces"] is None:
        raise ValueError(f"{fn_extxyz} is missing force data.")
    frame["source"] = run_dir.name
    return frame


def wait_for_snapshot_outputs(
    snapshot_run_dirs: list[Path],
    timeout_seconds: float,
    logger: Logger,
) -> None:
    """Give shared filesystems a short window to expose finished extxyz files."""
    missing = [
        run_dir
        for run_dir in snapshot_run_dirs
        if not (run_dir / "sp.extxyz").exists()
    ]
    if not missing or timeout_seconds <= 0:
        return

    deadline = time.monotonic() + timeout_seconds
    n_initial = len(missing)
    logger.status(
        "Waiting for snapshot outputs",
        f"{n_initial} missing",
        detail=f"timeout {timeout_seconds:g} s",
        level=2,
    )

    while missing:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(2.0, remaining))
        missing = [
            run_dir
            for run_dir in snapshot_run_dirs
            if not (run_dir / "sp.extxyz").exists()
        ]
        if missing:
            logger.status(
                "Waiting for snapshot outputs",
                f"{len(missing)} missing",
                detail=f"{max(deadline - time.monotonic(), 0.0):.0f} s left",
                level=2,
                overwrite=True,
            )

    n_ready = n_initial - len(missing)
    if missing:
        logger.warn(
            "Snapshot output wait timed out: "
            f"{n_ready}/{n_initial} delayed sp.extxyz files appeared; "
            f"{len(missing)} still missing.",
            level=2,
        )
        return

    logger.done(
        "Snapshot output wait",
        detail=f"{n_initial} delayed sp.extxyz files appeared",
        level=2,
    )


def cleanup_collected_snapshot_dirs(
    snapshot_run_dirs: list[Path],
    logger: Logger,
) -> None:
    """Remove snapshot run directories after their frames were collected."""
    if not snapshot_run_dirs:
        logger.done(
            "Snapshot cleanup",
            detail="no collected snapshot directories",
            level=2,
        )
        return

    parents = sorted({run_dir.parent for run_dir in snapshot_run_dirs})
    n_removed = 0
    for run_dir in sorted(snapshot_run_dirs):
        if not run_dir.exists():
            continue
        shutil.rmtree(run_dir)
        n_removed += 1

    for parent in parents:
        if not parent.exists():
            continue
        try:
            parent.rmdir()
        except OSError:
            pass

    logger.done(
        "Snapshot cleanup",
        detail=f"removed {n_removed} collected directories",
        level=2,
    )


def collect_snapshot_splits(
    snapshot_run_dirs: list[Path],
    system_dir: Path,
    config: ReferenceConfig,
    logger: Logger,
) -> list[Path]:
    wait_for_snapshot_outputs(
        snapshot_run_dirs,
        config.collection_wait_seconds,
        logger,
    )

    frames: list[dict[str, object]] = []
    collected_run_dirs: list[Path] = []
    for run_dir in sorted(snapshot_run_dirs):
        try:
            frame = _collect_frame(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.warn(f"Skipping {run_dir.name}: {exc}", level=2)
        else:
            frames.append(frame)
            collected_run_dirs.append(run_dir)

    random.Random(config.seed).shuffle(frames)
    train_frames, valid_frames = _split_train_valid(frames, config.train_fraction)
    write_extxyz_frames(train_frames, system_dir / "train.extxyz")
    write_extxyz_frames(valid_frames, system_dir / "valid.extxyz")

    n_train = len(train_frames)
    n_valid = len(valid_frames)
    n_total = len(snapshot_run_dirs)
    n_collected = n_train + n_valid
    logger.done(
        "Snapshot collection",
        detail=f"{n_collected}/{n_total} frames -> train {n_train}, valid {n_valid}",
        level=2,
    )
    if n_collected != n_total:
        logger.warn(
            "Only "
            f"{n_collected} of {n_total} snapshot runs produced usable "
            "sp.extxyz files.",
            level=2,
        )
    return collected_run_dirs


def collect_system_outputs(
    system_dir: Path,
    config: ReferenceConfig,
    logger: Logger,
) -> None:
    snapshot_run_dirs = sorted(
        path for path in (system_dir / "snapshots").iterdir() if path.is_dir()
    )
    collected_snapshot_dirs = collect_snapshot_splits(
        snapshot_run_dirs,
        system_dir,
        config,
        logger,
    )

    if config.single_atoms:
        single_atom_root = system_dir / "single-atoms"
        single_atom_dirs = sorted(
            path for path in single_atom_root.iterdir() if path.is_dir()
        )
        energies = collect_single_atom_energies(single_atom_dirs)
        save_yaml(energies, system_dir / "single-atoms.yaml")
        logger.done(
            "Single-atom energies",
            detail=", ".join(
                f"Z={atomic_number}={value:.6f} eV"
                for atomic_number, value in energies.items()
            ),
            level=2,
        )

    if config.cleanup_snapshots:
        cleanup_collected_snapshot_dirs(collected_snapshot_dirs, logger)
        single_atom_root = system_dir / "single-atoms"
        if single_atom_root.exists():
            shutil.rmtree(single_atom_root)


def _enabled_job_kinds(include_single_atoms: bool) -> tuple[ReferenceJobKind, ...]:
    if include_single_atoms:
        return ("snapshot", "single_atom")
    return ("snapshot",)


def process_system(
    system: ReferenceSystemConfig,
    config: ReferenceConfig,
    logger: Logger,
    *,
    job_ids: list[int] | None = None,
) -> Path:
    system_dir, snapshot_run_dirs, single_atom_dirs = stage_system(system, config)

    logger.kv("Prepared assets", system.assets_dir.resolve(), level=2)
    logger.kv("Output directory", system_dir.resolve(), level=2)
    logger.kv("Snapshots", len(snapshot_run_dirs), level=2)

    run_dirs_by_kind = {
        "snapshot": snapshot_run_dirs,
        "single_atom": single_atom_dirs,
    }
    is_local = config.job_scheduler == "local"
    max_parallel_jobs = -1
    if not is_local:
        assert config.slurm is not None
        if job_ids is None:
            raise ValueError("job_ids are required for Slurm reference jobs.")
        max_parallel_jobs = config.slurm.max_parallel_jobs

    for kind in _enabled_job_kinds(config.single_atoms):
        run_dirs = run_dirs_by_kind[kind]
        if is_local:
            label = REFERENCE_JOBS[kind]["label"]
            for index, run_dir in enumerate(run_dirs, start=1):
                logger.status(
                    label,
                    f"{index}/{len(run_dirs)}",
                    detail=run_dir.name,
                    level=2,
                    overwrite=True,
                )
                run_reference_job(kind, run_dir, config.cp2k_cmd)
            logger.done(label, detail=f"{len(run_dirs)}/{len(run_dirs)}", level=2)
            continue

        job = REFERENCE_JOBS[kind]
        submit_label = job["submit_label"]
        job_name_prefix = job["job_name_prefix"]
        script_name = job["script_name"]
        for index, run_dir in enumerate(run_dirs, start=1):
            if max_parallel_jobs != -1:
                wait_for_scheduler_slot(
                    job_ids=job_ids,
                    scheduler="slurm",
                    max_parallel_jobs=max_parallel_jobs,
                )
            logger.status(
                submit_label,
                f"{index}/{len(run_dirs)}",
                detail=run_dir.name,
                level=2,
                overwrite=True,
            )
            job_ids.append(
                submit_reference_job(
                    kind=kind,
                    run_dir=run_dir,
                    config=config,
                    job_name=f"{job_name_prefix}-{system.system_id}",
                    script_name=script_name,
                )
            )
        logger.done(
            submit_label,
            detail=f"{len(run_dirs)}/{len(run_dirs)}",
            level=2,
        )

    if is_local:
        collect_system_outputs(system_dir, config, logger)
    return system_dir


def run_job(fn_config: str | Path) -> None:
    """Run one staged snapshot or isolated-atom reference job."""
    job = ReferenceJobConfig.load(fn_config)
    run_reference_job(job.kind, job.run_dir, job.cp2k_cmd)


def main(fn_config: str) -> None:
    """Run staged CP2K jobs or import external reference trajectories."""
    config = ReferenceConfig.load(fn_config)
    config.reference_dir.resolve().mkdir(parents=True, exist_ok=True)

    logger = Logger("reference")
    print_reference_summary(config, logger)

    if config.mode == "import":
        for index, system in enumerate(config.systems, start=1):
            if not isinstance(system, ImportedReferenceSystemConfig):
                raise TypeError(
                    f"Expected imported reference system, got {type(system)}"
                )
            logger.info(f"System {index}/{len(config.systems)}", level=1)
            import_reference_system(system, config.reference_dir.resolve(), logger)
            logger.blank()
        return

    if config.cp2k_cmd is None or config.job_scheduler is None:
        raise ValueError("Run mode requires both 'cp2k_cmd' and 'job_scheduler'.")

    if config.job_scheduler == "local":
        config = replace(config, cp2k_cmd=check_cp2k_available(config.cp2k_cmd))

    job_ids: list[int] = []
    staged_system_dirs: list[Path] = []
    for index, system in enumerate(config.systems, start=1):
        if not isinstance(system, ReferenceSystemConfig):
            raise TypeError(f"Expected staged reference system, got {type(system)}")
        logger.info(f"System {index}/{len(config.systems)}", level=1)
        staged_system_dirs.append(
            process_system(
                system,
                config,
                logger,
                job_ids=job_ids if config.job_scheduler == "slurm" else None,
            )
        )
        logger.blank()

    if config.job_scheduler == "local":
        return

    if job_ids:
        logger.status("Waiting for Slurm jobs", f"{len(job_ids)} submitted", level=1)
        control_jobs(job_ids, "slurm")
        logger.done("Waiting for Slurm jobs", detail="all jobs finished", level=1)

    for index, system_dir in enumerate(staged_system_dirs, start=1):
        logger.info(f"Collecting system {index}/{len(staged_system_dirs)}", level=1)
        collect_system_outputs(system_dir, config, logger)
        logger.blank()
