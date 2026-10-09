"""Analyze samples in parallel: one worker process per sample.

Within a sample the systems are analyzed one after another; each system's
trajectory is opened once and every routine of that system runs on it. The
reference is analyzed the same way, as one sample.

A sample that fails (for example on a corrupted trajectory) is reported with
its error instead of stopping the other samples.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from typing import Any

from ..io.logs import Logger
from ..io.progress import iter_progress
from .dataset import QoI
from .routines import AnalysisRoutineConfig, run_routine
from .trajectory import open_trajectory

# (sample_id, {system_id: {role: path}})
AnalysisTask = tuple[str, dict[str, dict[str, Any]]]
# {system_id: {routine name: QoI}}
SampleResult = dict[str, dict[str, QoI]]


def analyze_sample(
    task: AnalysisTask,
    *,
    routines_by_system: Mapping[str, Sequence[AnalysisRoutineConfig]],
    frames: slice,
    in_memory: bool,
    memory_limit: int | None,
) -> tuple[str, SampleResult | str]:
    """Run every routine on every system of one sample; return an error text
    instead of the results if anything fails."""
    sample_id, systems = task
    results: SampleResult = {}
    try:
        for system_id, inputs in systems.items():
            routines = routines_by_system[system_id]
            universe = None
            system_frames = frames
            if any(routine.uses_trajectory for routine in routines):
                universe, system_frames = open_trajectory(
                    inputs,
                    start=frames.start,
                    stop=frames.stop,
                    step=frames.step,
                    in_memory=in_memory,
                    memory_limit=memory_limit,
                    context=f"System {system_id!r}, sample {sample_id!r}",
                )
            try:
                results[system_id] = {
                    routine.name: run_routine(
                        routine,
                        universe=universe,
                        frames=system_frames,
                        inputs=inputs,
                        system_id=system_id,
                        sample_id=sample_id,
                    )
                    for routine in routines
                }
            finally:
                if universe is not None:
                    universe.trajectory.close()
    except Exception as exc:  # one broken sample must not stop the others
        return sample_id, f"{type(exc).__name__}: {exc}"
    return sample_id, results


def _available_memory() -> int | None:
    """Memory available to this job in bytes: the free physical memory, capped
    by the Slurm allocation, which is far less than the node's on a shared node.
    """
    limits = []
    try:
        limits.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        pass
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if per_node:
        limits.append(int(per_node) * 2**20)
    elif per_cpu and cpus:
        limits.append(int(per_cpu) * int(cpus) * 2**20)
    return min(limits) if limits else None


def analyze_samples(
    tasks: Sequence[AnalysisTask],
    *,
    routines_by_system: Mapping[str, Sequence[AnalysisRoutineConfig]],
    frames: slice,
    workers: int,
    in_memory: bool,
    logger: Logger,
    label: str,
) -> tuple[dict[str, SampleResult], dict[str, str]]:
    """Analyze samples, ``workers`` at a time (``-1``: every CPU).

    Returns the results of the samples that succeeded and the error of each
    sample that failed.
    """
    if workers == -1:
        workers = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else mp.cpu_count()
        )
    workers = max(1, min(workers, len(tasks)))
    logger.status(
        label, "started", detail=f"{len(tasks)} sample(s), {workers} worker(s)"
    )
    available = _available_memory()
    # In-memory trajectories may use half the available memory, shared by
    # the workers; larger ones are read from disk.
    memory_limit = None if available is None else available // (2 * workers)
    analyze_one = partial(
        analyze_sample,
        routines_by_system=routines_by_system,
        frames=frames,
        in_memory=in_memory,
        memory_limit=memory_limit,
    )
    if workers == 1:
        outcomes = map(analyze_one, tasks)
        return _collect(outcomes, len(tasks), logger, label)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        outcomes = executor.map(analyze_one, tasks, chunksize=1)
        try:
            return _collect(outcomes, len(tasks), logger, label)
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "An analysis worker was killed, most likely for running out of "
                "memory; lower training_samples.workers or set run.in_memory: "
                "false."
            ) from exc


def _collect(
    outcomes: Any, total: int, logger: Logger, label: str
) -> tuple[dict[str, SampleResult], dict[str, str]]:
    results: dict[str, SampleResult] = {}
    failures: dict[str, str] = {}
    for sample_id, outcome in iter_progress(
        outcomes, total=total, logger=logger, label=label
    ):
        if isinstance(outcome, str):
            failures[sample_id] = outcome
        else:
            results[sample_id] = outcome
    return results, failures
