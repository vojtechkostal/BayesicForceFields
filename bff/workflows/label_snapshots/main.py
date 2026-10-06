"""Label trajectory snapshots with CP2K energies and forces.

Every snapshot (and, optionally, every isolated element) becomes one staged
CP2K job. Jobs run locally one after another or as one Slurm job array; the
finished frames are split into ``train.extxyz`` and ``test.extxyz``.
"""

from __future__ import annotations

import random
import shlex
import shutil
import time
import warnings
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from ... import slurm
from ...io.cp2k import collect_single_atom_energies, write_cp2k_single_atom_xyz
from ...io.extxyz import read_extxyz_frame, write_extxyz_frame, write_extxyz_frames
from ...io.logs import Logger
from ...io.utils import file_sha256, save_yaml
from .config import LabelSnapshotsConfig, SnapshotSystemConfig
from .job import resolve_cp2k_command, run_snapshot_job, write_job_config


def print_label_summary(config: LabelSnapshotsConfig, logger: Logger) -> None:
    """Print a concise snapshot-labeling workflow summary."""
    logger.section("Label Snapshots")
    logger.kv("Output directory", config.output_dir.resolve())
    logger.kv("Systems", len(config.systems))
    logger.kv("Scheduler", config.job_scheduler)
    logger.kv("CP2K command", config.cp2k_cmd)
    logger.kv("Single-atom energies", "yes" if config.single_atoms else "no")
    logger.kv("Train fraction", config.train_fraction)
    logger.kv("Shuffle seed", config.seed)
    logger.kv("Cleanup snapshots", "yes" if config.cleanup_snapshots else "no")
    logger.kv("Collection wait", f"{config.collection_wait_seconds:g} s")
    if not config.single_atoms:
        logger.warn("Single-atom energies are disabled for this run.")
    logger.blank()


def validate_single_atom_inputs(
    system: SnapshotSystemConfig,
    elements: set[str],
) -> dict[str, Path]:
    """Require exactly one user CP2K input for every selected element."""
    supplied = system.single_atom_input_paths
    if not supplied:
        raise ValueError(
            f"system {system.system_id!r} requires single_atom_inputs when "
            "single_atoms is true."
        )

    missing = elements - set(supplied)
    extra = set(supplied) - elements
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unused " + ", ".join(sorted(extra)))
        raise ValueError(
            f"system {system.system_id!r} single_atom_inputs do not match its "
            "selected elements: " + "; ".join(details) + "."
        )
    return supplied


def stage_system(
    system: SnapshotSystemConfig,
    config: LabelSnapshotsConfig,
) -> tuple[Path, list[Path], list[Path], tuple[int, ...]]:
    """Extract and stage one trajectory's snapshots."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Reader has no dt information, set to 1.0 ps.*",
            category=UserWarning,
        )
        universe = mda.Universe(system.topology_path, system.trajectory_path)
        if not hasattr(universe.atoms, "elements"):
            universe.guess_TopologyAttrs(to_guess=["elements"])
    n_frames = universe.trajectory.n_frames
    if system.n_snapshots > n_frames:
        raise ValueError(
            f"system {system.system_id!r} requests {system.n_snapshots} snapshots "
            f"from a trajectory containing only {n_frames} frames."
        )

    frame_indices = np.unique(
        np.linspace(0, n_frames - 1, num=system.n_snapshots, dtype=int)
    )
    atoms = universe.select_atoms(system.atom_selection)
    if len(atoms) == 0:
        raise ValueError(
            f"system {system.system_id!r} atom_selection selects no atoms."
        )
    elements = tuple(sorted({str(element).capitalize() for element in atoms.elements}))
    single_atom_inputs = None
    if config.single_atoms:
        single_atom_inputs = validate_single_atom_inputs(system, set(elements))

    system_dir = config.output_dir.resolve() / "systems" / system.system_id
    system_dir.mkdir(parents=True, exist_ok=True)
    for stale in (system_dir / "train.extxyz", system_dir / "test.extxyz"):
        if stale.exists():
            stale.unlink()

    snapshots_dir = system_dir / "snapshots"
    if snapshots_dir.exists():
        shutil.rmtree(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshot_run_dirs: list[Path] = []
    for output_index, frame_index in enumerate(frame_indices):
        run_dir = snapshots_dir / f"snapshot-{output_index:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        timestep = universe.trajectory[int(frame_index)]
        write_extxyz_frame(
            atoms,
            run_dir / "pos.xyz",
            dimensions=timestep.dimensions,
        )
        shutil.copy2(system.md_input_path, run_dir / "md.inp")
        shutil.copy2(system.sp_input_path, run_dir / "sp.inp")
        snapshot_run_dirs.append(run_dir)

    single_atom_run_dirs: list[Path] = []
    if config.single_atoms:
        assert single_atom_inputs is not None
        single_atoms_dir = system_dir / "single-atoms"
        if single_atoms_dir.exists():
            shutil.rmtree(single_atoms_dir)
        single_atoms_dir.mkdir(parents=True, exist_ok=True)

        for element in elements:
            run_dir = single_atoms_dir / element.lower()
            run_dir.mkdir(parents=True, exist_ok=True)
            write_cp2k_single_atom_xyz(element, run_dir / "pos.xyz")
            shutil.copy2(single_atom_inputs[element], run_dir / "input.inp")
            single_atom_run_dirs.append(run_dir)

    return (
        system_dir,
        snapshot_run_dirs,
        single_atom_run_dirs,
        tuple(int(index) for index in frame_indices),
    )


def _split_train_test(
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
        run_dir for run_dir in snapshot_run_dirs if not (run_dir / "sp.extxyz").exists()
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
    config: LabelSnapshotsConfig,
    logger: Logger,
) -> tuple[list[Path], int, int]:
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
    train_frames, test_frames = _split_train_test(frames, config.train_fraction)
    write_extxyz_frames(train_frames, system_dir / "train.extxyz")
    write_extxyz_frames(test_frames, system_dir / "test.extxyz")

    n_train = len(train_frames)
    n_test = len(test_frames)
    n_total = len(snapshot_run_dirs)
    n_collected = n_train + n_test
    logger.done(
        "Snapshot collection",
        detail=f"{n_collected}/{n_total} frames -> train {n_train}, test {n_test}",
        level=2,
    )
    if n_collected != n_total:
        logger.warn(
            "Only "
            f"{n_collected} of {n_total} snapshot runs produced usable "
            "sp.extxyz files.",
            level=2,
        )
    return collected_run_dirs, n_train, n_test


def collect_system_outputs(
    system_dir: Path,
    config: LabelSnapshotsConfig,
    logger: Logger,
) -> dict[str, object]:
    snapshot_run_dirs = sorted(
        path for path in (system_dir / "snapshots").iterdir() if path.is_dir()
    )
    collected_snapshot_dirs, n_train, n_test = collect_snapshot_splits(
        snapshot_run_dirs,
        system_dir,
        config,
        logger,
    )

    energies: dict[int, float] = {}
    single_atom_failures: dict[str, str] = {}
    if config.single_atoms:
        single_atom_root = system_dir / "single-atoms"
        single_atom_dirs = sorted(
            path for path in single_atom_root.iterdir() if path.is_dir()
        )
        for atom_dir in single_atom_dirs:
            try:
                energies.update(collect_single_atom_energies([atom_dir]))
            except (FileNotFoundError, ValueError) as exc:
                single_atom_failures[atom_dir.name] = str(exc)
                logger.warn(f"Skipping isolated atom {atom_dir.name}: {exc}", level=2)
        save_yaml(energies, system_dir / "single-atoms.yaml")
        logger.done(
            "Single-atom energies",
            detail=f"{len(energies)}/{len(single_atom_dirs)} collected",
            level=2,
        )

    if config.cleanup_snapshots:
        cleanup_collected_snapshot_dirs(collected_snapshot_dirs, logger)
        single_atom_root = system_dir / "single-atoms"
        if single_atom_root.exists():
            shutil.rmtree(single_atom_root)

    return {
        "train_count": n_train,
        "test_count": n_test,
        "single_atom_energies": energies,
        "single_atom_failures": single_atom_failures,
    }


def _system_manifest(
    system: SnapshotSystemConfig,
    system_dir: Path,
    output_dir: Path,
    frame_indices: tuple[int, ...],
    results: dict[str, object],
) -> dict[str, object]:
    def source(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": file_sha256(path)}

    return {
        "directory": str(system_dir.relative_to(output_dir)),
        "train": str((system_dir / "train.extxyz").relative_to(output_dir)),
        "test": str((system_dir / "test.extxyz").relative_to(output_dir)),
        "selected_frame_indices": list(frame_indices),
        **results,
        "sources": {
            "topology": source(system.topology_path),
            "trajectory": source(system.trajectory_path),
            "md_input": source(system.md_input_path),
            "sp_input": source(system.sp_input_path),
            "single_atom_inputs": {
                element: source(path)
                for element, path in sorted(
                    (system.single_atom_input_paths or {}).items()
                )
            },
        },
    }


def main(fn_config: str | Path) -> None:
    """Stage, run, and collect CP2K labels for every configured system."""
    config = LabelSnapshotsConfig.load(fn_config)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger("label-snapshots", str(config.log), mode="w")
    print_label_summary(config, logger)
    cp2k_cmd = resolve_cp2k_command(
        config.cp2k_cmd, must_exist=config.job_scheduler == "local"
    )

    staged = []
    jobs: list[tuple[str, Path]] = []
    for system in config.systems:
        system_dir, snapshot_dirs, single_atom_dirs, frame_indices = stage_system(
            system, config
        )
        staged.append((system, system_dir, frame_indices))
        jobs += [("snapshot", run_dir) for run_dir in snapshot_dirs]
        jobs += [("single_atom", run_dir) for run_dir in single_atom_dirs]
        logger.kv(
            system.system_id,
            f"{len(snapshot_dirs)} snapshots, {len(single_atom_dirs)} isolated atoms",
        )
    logger.blank()

    if config.job_scheduler == "local":
        n_failed = 0
        for index, (kind, run_dir) in enumerate(jobs):
            logger.progress_status(
                f"CP2K jobs: {index}/{len(jobs)} | {run_dir.parent.parent.name}/"
                f"{run_dir.name}",
                index,
                len(jobs),
                overwrite=True,
                write_file=False,
            )
            try:
                run_snapshot_job(kind, run_dir, cp2k_cmd)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                n_failed += 1
                logger.warn(f"{run_dir} failed: {exc}")
        logger.done("CP2K jobs", detail=f"{len(jobs) - n_failed}/{len(jobs)} succeeded")
    else:
        fn_jobs = output_dir / "jobs.txt"
        fn_jobs.write_text(
            "".join(
                f"{write_job_config(run_dir, kind, cp2k_cmd)}\n"
                for kind, run_dir in jobs
            ),
            encoding="utf-8",
        )
        fn_script = slurm.write_task_script(
            output_dir / "run.sh",
            config=config.slurm,
            sbatch={
                "job_name": "bff-label",
                "output": output_dir / "slurm" / "%A_%a.out",
                **(config.slurm.sbatch or {}),
            },
            commands=[
                f'JOB_CONFIG=$(sed -n "$((TASK_ID + 1))p" {shlex.quote(str(fn_jobs))})',
                slurm.bff_command("label-snapshot-job", '"$JOB_CONFIG"'),
            ],
        )
        (output_dir / "slurm").mkdir(exist_ok=True)
        logger.kv("Job list", fn_jobs)
        slurm.run_tasks(fn_script, len(jobs), config=config.slurm, logger=logger)
    logger.blank()

    manifest: dict[str, object] = {}
    for system, system_dir, frame_indices in staged:
        logger.info(f"Collecting {system.system_id}", level=1)
        results = collect_system_outputs(system_dir, config, logger)
        manifest[system.system_id] = _system_manifest(
            system, system_dir, output_dir, frame_indices, results
        )
        logger.blank()
    save_yaml(
        {"stage": "label-snapshots", "systems": manifest}, config.results_manifest
    )
    logger.done("Label results", detail=str(config.results_manifest), level=1)
