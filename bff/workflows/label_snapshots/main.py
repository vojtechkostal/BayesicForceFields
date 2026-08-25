"""Workflow entry point for canonical snapshot evaluation."""

from __future__ import annotations

import os
import random
import shlex
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal

import MDAnalysis as mda

from ...io.cp2k import (
    collect_single_atom_energies,
    write_cp2k_single_atom_xyz,
    write_cp2k_snapshot_extxyz,
)
from ...io.extxyz import read_extxyz_frame, write_extxyz_frame, write_extxyz_frames
from ...io.logs import Logger
from ...io.utils import file_sha256, load_yaml, save_yaml
from .._shared.preparation import sample_snapshot_indices
from .._shared.scheduler import (
    bff_cli_command,
    build_slurm_cli_job,
    control_jobs,
    get_job_state_counts,
    wait_for_scheduler_slot,
)
from .config import (
    LabelSnapshotsConfig,
    SnapshotSystemConfig,
)

SnapshotJobKind = Literal["snapshot", "single_atom"]

SNAPSHOT_JOBS = {
    "snapshot": {
        "steps": (("md.inp", "md.out"), ("sp.inp", "sp.out")),
        "label": "Snapshot jobs",
        "submit_label": "Submitting snapshot jobs",
        "job_name_prefix": "bff-snap",
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
class LabelSnapshotJobConfig:
    kind: SnapshotJobKind
    run_dir: Path
    cp2k_cmd: str

    @classmethod
    def load(cls, fn_config: str | Path) -> "LabelSnapshotJobConfig":
        data = load_yaml(fn_config)
        if not isinstance(data, dict):
            raise ValueError("Snapshot job config must contain a mapping.")
        unknown = set(data) - {"kind", "run_dir", "cp2k_cmd"}
        if unknown:
            raise ValueError(
                "Snapshot job config contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )

        kind = data.get("kind")
        if kind not in SNAPSHOT_JOBS:
            raise ValueError(
                "Snapshot job config 'kind' must be 'snapshot' or 'single_atom'."
            )
        for key in ("run_dir", "cp2k_cmd"):
            if key not in data:
                raise ValueError(f"Snapshot job config is missing {key!r}.")

        run_dir = Path(data["run_dir"]).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Snapshot job directory not found: {run_dir}")

        cp2k_cmd = str(data["cp2k_cmd"])
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
    if config.job_scheduler == "slurm" and config.slurm is not None:
        logger.kv("Max parallel jobs", config.slurm.max_parallel_jobs)
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


def count_scheduler_jobs(config: LabelSnapshotsConfig) -> int:
    """Count Slurm jobs that will be submitted for snapshot evaluation."""
    total = 0
    for system in config.systems:
        total += system.n_snapshots
        if config.single_atoms:
            universe = mda.Universe(system.topology_path)
            if not hasattr(universe.atoms, "elements"):
                universe.guess_TopologyAttrs(to_guess=["elements"])
            atoms = universe.select_atoms(system.atom_selection)
            elements = {
                str(element).capitalize() for element in atoms.elements
            }
            validate_single_atom_inputs(system, elements)
            total += len(elements)
    return total


def build_scheduler_monitor(
    *,
    total_jobs: int,
    logger: Logger,
) -> Callable[..., None]:
    """Build a Slurm status logger matching the sampling campaign monitor."""
    count_width = len(str(max(total_jobs, 1)))

    def log_job_monitor(
        counts: dict[str, int],
        *,
        overwrite: bool = True,
    ) -> None:
        finished_percent = (
            100 if total_jobs == 0 else 100 * counts["finished"] / total_jobs
        )
        logger.status(
            "Scheduler jobs",
            (
                f"submitted {counts['submitted']:>{count_width}d}/"
                f"{total_jobs:<{count_width}d} | "
                f"pending {counts['pending']:>{count_width}d} | "
                f"running {counts['running']:>{count_width}d} | "
                f"finished {counts['finished']:>{count_width}d} "
                f"[{finished_percent:3.0f}%]"
            ),
            level=1,
            overwrite=overwrite,
        )

    return log_job_monitor


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

    frame_indices = sample_snapshot_indices(n_frames, system.n_snapshots)
    atoms = universe.select_atoms(system.atom_selection)
    if len(atoms) == 0:
        raise ValueError(
            f"system {system.system_id!r} atom_selection selects no atoms."
        )
    elements = tuple(
        sorted({str(element).capitalize() for element in atoms.elements})
    )
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
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        output = (completed.stdout or "").strip()
        detail = f"\n{output[-4000:]}" if output else ""
        raise RuntimeError(
            f"CP2K command failed for {cwd / fn_input} with exit code "
            f"{completed.returncode}; inspect {cwd / fn_output}.{detail}"
        )


def _remove_cp2k_restart_files(run_dir: Path) -> None:
    """Remove large CP2K restart artifacts from one run directory."""
    for pattern in ("*.wfn*", "*.restart*"):
        for path in run_dir.glob(pattern):
            path.unlink()


def run_snapshot_job(
    kind: SnapshotJobKind,
    run_dir: Path,
    cp2k_cmd: str,
) -> None:
    """Run one staged snapshot or isolated-atom CP2K job."""
    for fn_input, fn_output in SNAPSHOT_JOBS[kind]["steps"]:
        _run_cp2k(
            cp2k_cmd=cp2k_cmd,
            fn_input=fn_input,
            fn_output=fn_output,
            cwd=run_dir,
        )
    _remove_cp2k_restart_files(run_dir)
    if kind == "snapshot":
        write_cp2k_snapshot_extxyz(run_dir)


def submit_snapshot_job(
    *,
    kind: SnapshotJobKind,
    run_dir: Path,
    config: LabelSnapshotsConfig,
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
        command=bff_cli_command("label-snapshot-job", fn_job.resolve()),
        slurm_config=config.slurm,
        sbatch=submit_specs,
        cwd=run_dir,
    )
    return submit.submit(run_dir / script_name)


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


def _enabled_job_kinds(include_single_atoms: bool) -> tuple[SnapshotJobKind, ...]:
    if include_single_atoms:
        return ("snapshot", "single_atom")
    return ("snapshot",)


def process_system(
    system: SnapshotSystemConfig,
    config: LabelSnapshotsConfig,
    logger: Logger,
    *,
    job_ids: list[int] | None = None,
    job_monitor: Callable[..., None] | None = None,
) -> tuple[Path, tuple[int, ...], dict[str, object] | None]:
    system_dir, snapshot_run_dirs, single_atom_dirs, frame_indices = stage_system(
        system, config
    )

    logger.kv("System ID", system.system_id, level=2)
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
            raise ValueError("job_ids are required for Slurm snapshot jobs.")
        max_parallel_jobs = config.slurm.max_parallel_jobs

    for kind in _enabled_job_kinds(config.single_atoms):
        run_dirs = run_dirs_by_kind[kind]
        if is_local:
            label = SNAPSHOT_JOBS[kind]["label"]
            n_succeeded = 0
            for index, run_dir in enumerate(run_dirs, start=1):
                if index < len(run_dirs):
                    logger.status(
                        label,
                        f"{index}/{len(run_dirs)}",
                        detail=run_dir.name,
                        level=2,
                        overwrite=True,
                    )
                try:
                    run_snapshot_job(kind, run_dir, config.cp2k_cmd)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    logger.warn(f"{run_dir.name} failed: {exc}", level=2)
                else:
                    n_succeeded += 1
            logger.done(
                label,
                detail=f"{n_succeeded}/{len(run_dirs)} succeeded",
                level=2,
            )
            continue

        job = SNAPSHOT_JOBS[kind]
        submit_label = job["submit_label"]
        job_name_prefix = job["job_name_prefix"]
        script_name = job["script_name"]
        for index, run_dir in enumerate(run_dirs, start=1):
            if max_parallel_jobs != -1:
                wait_for_scheduler_slot(
                    job_ids=job_ids,
                    scheduler="slurm",
                    max_parallel_jobs=max_parallel_jobs,
                    monitor=job_monitor,
                )
            logger.status(
                submit_label,
                f"{index}/{len(run_dirs)}",
                detail=run_dir.name,
                level=2,
                overwrite=True,
            )
            job_ids.append(
                submit_snapshot_job(
                    kind=kind,
                    run_dir=run_dir,
                    config=config,
                    job_name=f"{job_name_prefix}-{system.system_id}",
                    script_name=script_name,
                )
            )
            if job_monitor is not None:
                job_monitor(get_job_state_counts(job_ids, "slurm"))
        logger.done(
            submit_label,
            detail=f"{len(run_dirs)}/{len(run_dirs)}",
            level=2,
        )

    if is_local:
        results = collect_system_outputs(system_dir, config, logger)
    else:
        results = None
    return system_dir, frame_indices, results


def run_job(fn_config: str | Path) -> None:
    """Run one staged snapshot or isolated-atom CP2K job."""
    job = LabelSnapshotJobConfig.load(fn_config)
    run_snapshot_job(job.kind, job.run_dir, job.cp2k_cmd)


def main(fn_config: str) -> None:
    """Run staged CP2K snapshot jobs."""
    config = LabelSnapshotsConfig.load(fn_config)
    config.output_dir.resolve().mkdir(parents=True, exist_ok=True)

    logger = Logger("label-snapshots", str(config.log), mode="w")
    print_label_summary(config, logger)

    if config.job_scheduler == "local":
        config = replace(config, cp2k_cmd=check_cp2k_available(config.cp2k_cmd))

    job_ids: list[int] = []
    staged_system_dirs: list[Path] = []
    frame_indices_by_system: dict[str, tuple[int, ...]] = {}
    results_by_system: dict[str, dict[str, object]] = {}
    job_monitor = None
    n_scheduler_jobs = 0
    if config.job_scheduler == "slurm":
        n_scheduler_jobs = count_scheduler_jobs(config)
        job_monitor = build_scheduler_monitor(
            total_jobs=n_scheduler_jobs,
            logger=logger,
        )

    for index, system in enumerate(config.systems, start=1):
        logger.info(
            f"System {index}/{len(config.systems)}: {system.system_id}", level=1
        )
        system_dir, frame_indices, results = process_system(
            system,
            config,
            logger,
            job_ids=job_ids if config.job_scheduler == "slurm" else None,
            job_monitor=job_monitor,
        )
        staged_system_dirs.append(system_dir)
        frame_indices_by_system[system.system_id] = frame_indices
        if results is not None:
            results_by_system[system.system_id] = results
        logger.blank()

    if config.job_scheduler == "slurm" and job_ids:
        if job_monitor is not None:
            job_monitor(get_job_state_counts(job_ids, "slurm"))
        control_jobs(job_ids, "slurm", monitor=job_monitor)
        if job_monitor is not None:
            job_monitor(get_job_state_counts(job_ids, "slurm"), overwrite=False)
        logger.done(
            "Scheduler jobs",
            detail=f"{len(job_ids)}/{n_scheduler_jobs} [100%]",
            level=1,
        )

    if config.job_scheduler == "slurm":
        for index, (system, system_dir) in enumerate(
            zip(config.systems, staged_system_dirs), start=1
        ):
            logger.info(
                f"Collecting system {index}/{len(staged_system_dirs)}", level=1
            )
            results_by_system[system.system_id] = collect_system_outputs(
                system_dir, config, logger
            )
            logger.blank()

    save_yaml(
        {
            "stage": "label-snapshots",
            "systems": {
                system.system_id: {
                    "directory": str(
                        directory.relative_to(config.output_dir.resolve())
                    ),
                    "train": str(
                        (directory / "train.extxyz").relative_to(
                            config.output_dir.resolve()
                        )
                    ),
                    "test": str(
                        (directory / "test.extxyz").relative_to(
                            config.output_dir.resolve()
                        )
                    ),
                    "selected_frame_indices": list(
                        frame_indices_by_system[system.system_id]
                    ),
                    "train_count": results_by_system[system.system_id][
                        "train_count"
                    ],
                    "test_count": results_by_system[system.system_id]["test_count"],
                    "single_atom_energies": results_by_system[system.system_id][
                        "single_atom_energies"
                    ],
                    "single_atom_failures": results_by_system[system.system_id][
                        "single_atom_failures"
                    ],
                    "sources": {
                        "topology": {
                            "path": str(system.topology_path),
                            "sha256": file_sha256(system.topology_path),
                        },
                        "trajectory": {
                            "path": str(system.trajectory_path),
                            "sha256": file_sha256(system.trajectory_path),
                        },
                        "md_input": {
                            "path": str(system.md_input_path),
                            "sha256": file_sha256(system.md_input_path),
                        },
                        "sp_input": {
                            "path": str(system.sp_input_path),
                            "sha256": file_sha256(system.sp_input_path),
                        },
                        "single_atom_inputs": {
                            element: {
                                "path": str(path),
                                "sha256": file_sha256(path),
                            }
                            for element, path in sorted(
                                (system.single_atom_input_paths or {}).items()
                            )
                        },
                    },
                }
                for system, directory in zip(config.systems, staged_system_dirs)
            },
        },
        config.results_manifest,
    )
    logger.done("Label results", detail=str(config.results_manifest), level=1)
