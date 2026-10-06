"""Analyze reference and sampled systems with the same execution path."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any

from ..io.logs import Logger
from ..io.progress import iter_progress
from .dataset import QoI
from .routines import AnalysisRoutineConfig, run_routine
from .trajectory import open_trajectory

AnalysisTask = tuple[str, dict[str, dict[str, Any]]]
AnalysisResults = dict[str, dict[str, dict[str, QoI]]]


def analyze_sample(
    task: AnalysisTask,
    *,
    routines_by_system: Mapping[str, Sequence[AnalysisRoutineConfig]],
    start: int,
    stop: int | None,
    step: int,
    in_memory: bool,
) -> tuple[str, dict[str, dict[str, QoI]]]:
    """Run every configured routine on every system of one sample.

    All trajectory routines of a system share one opened Universe.
    """
    sample_id, systems = task
    results: dict[str, dict[str, QoI]] = {}
    for system_id, inputs in systems.items():
        routines = routines_by_system[system_id]
        universe = None
        frames = slice(start, stop, step)
        if any(routine.uses_trajectory for routine in routines):
            universe, frames = open_trajectory(
                inputs,
                start=start,
                stop=stop,
                step=step,
                in_memory=in_memory,
                context=f"System {system_id!r}, sample {sample_id!r}",
            )
        try:
            results[system_id] = {
                routine.name: run_routine(
                    routine,
                    universe=universe,
                    frames=frames,
                    inputs=inputs,
                    system_id=system_id,
                    sample_id=sample_id,
                )
                for routine in routines
            }
        finally:
            if universe is not None:
                universe.trajectory.close()
    return sample_id, results


def analyze_samples(
    tasks: Sequence[AnalysisTask],
    *,
    routines_by_system: Mapping[str, Sequence[AnalysisRoutineConfig]],
    start: int,
    stop: int | None,
    step: int,
    workers: int,
    progress_stride: int,
    progress_label: str,
    logger: Logger,
    in_memory: bool,
) -> AnalysisResults:
    """Analyze complete samples, one sample per worker process."""
    if workers == 0 or workers < -1:
        raise ValueError("workers must be a positive integer or -1.")
    if not tasks:
        return {}

    if workers == -1:
        workers = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else mp.cpu_count()
        )
    workers = min(workers, len(tasks))
    logger.status(
        progress_label,
        "in progress...",
        detail=f"{len(tasks)} sample(s), {workers} worker(s)",
        level=1,
    )
    analyze_one = partial(
        analyze_sample,
        routines_by_system=routines_by_system,
        start=start,
        stop=stop,
        step=step,
        in_memory=in_memory,
    )

    def collect(analyzed):
        return dict(
            iter_progress(
                analyzed,
                total=len(tasks),
                stride=progress_stride,
                logger=logger,
                label=progress_label,
            )
        )

    if workers == 1:
        return collect(map(analyze_one, tasks))
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        return collect(executor.map(analyze_one, tasks, chunksize=1))
