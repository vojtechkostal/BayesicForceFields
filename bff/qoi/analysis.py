"""Analyze reference and sampled systems with the same execution path."""

from __future__ import annotations

import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import MDAnalysis as mda

from ..io.logs import Logger
from ..io.progress import iter_progress
from ..tools import _normalized_dimensions
from ..topology import prepare_universe
from .data import QoI
from .routines import AnalysisRoutineConfig, run_analysis_routine

AnalysisTask = tuple[str, dict[str, dict[str, Any]]]
AnalysisResults = dict[str, dict[str, dict[str, QoI]]]


@contextmanager
def _trajectory_context(
    inputs: Mapping[str, Any],
    *,
    system_id: str,
    sample_id: str,
    start: int,
    stop: int | None,
    step: int,
    in_memory: bool,
) -> Iterator[tuple[mda.Universe, tuple[int, int | None, int]]]:
    """Load, prepare, and close one system trajectory."""
    paths: dict[str, Path] = {}
    for role in ("topology", "coordinates", "trajectory"):
        value = inputs.get(role)
        if not isinstance(value, Path):
            raise ValueError(
                f"System {system_id!r}, sample {sample_id!r}, input role "
                f"{role!r}: expected one path, got {value!r}."
            )
        paths[role] = value

    universe = prepare_universe(
        str(paths["topology"]),
        str(paths["coordinates"]),
        dt=1,
    )
    try:
        default_dimensions = _normalized_dimensions(universe.dimensions)
        trajectory = paths["trajectory"]
        universe.load_new(str(trajectory))
        universe._bff_default_dimensions = default_dimensions

        effective_stop = stop
        if stop is None:
            frame_count = len(universe.trajectory)
            if start >= frame_count:
                raise ValueError(
                    f"System {system_id!r}, sample {sample_id!r}: frame start "
                    f"{start} is outside trajectory {trajectory}, which reports "
                    f"{frame_count} frames."
                )
            last_frame = start + ((frame_count - 1 - start) // step) * step
            requested_last_frame = last_frame
            while last_frame >= start:
                try:
                    universe.trajectory[last_frame]
                    break
                except (EOFError, OSError):
                    last_frame -= step
            if last_frame < start:
                raise OSError(
                    f"System {system_id!r}, sample {sample_id!r}: no readable "
                    f"frames remain in trajectory {trajectory} from frame {start}."
                )
            effective_stop = last_frame + 1
            if last_frame < requested_last_frame:
                warnings.warn(
                    f"System {system_id!r}, sample {sample_id!r}: ignoring an "
                    f"unreadable trailing record in trajectory {trajectory}; "
                    f"using all readable frames through frame {last_frame}.",
                    RuntimeWarning,
                    stacklevel=3,
                )

        frames = (start, effective_stop, step)
        if in_memory:
            universe.transfer_to_memory(start=start, stop=effective_stop, step=step)
            if default_dimensions is not None:
                for timestep in universe.trajectory:
                    if timestep.dimensions is None:
                        timestep.dimensions = default_dimensions
            frames = (0, None, 1)
        yield universe, frames
    finally:
        close = getattr(universe.trajectory, "close", None)
        if callable(close):
            close()


def analyze_sample(
    task: AnalysisTask,
    *,
    routines_by_system: Mapping[str, Sequence[AnalysisRoutineConfig]],
    start: int,
    stop: int | None,
    step: int,
    in_memory: bool,
) -> tuple[str, dict[str, dict[str, QoI]]]:
    """Analyze every configured system belonging to one sample or reference."""
    sample_id, systems = task
    sample_results: dict[str, dict[str, QoI]] = {}

    for system_id, inputs in systems.items():
        routines = tuple(routines_by_system[system_id])
        trajectory_context = (
            _trajectory_context(
                inputs,
                system_id=system_id,
                sample_id=sample_id,
                start=start,
                stop=stop,
                step=step,
                in_memory=in_memory,
            )
            if any(routine.uses_trajectory for routine in routines)
            else nullcontext((None, (start, stop, step)))
        )
        with trajectory_context as (universe, frames):
            sample_results[system_id] = {
                routine.name: run_analysis_routine(
                    routine,
                    universe=universe,
                    inputs=inputs,
                    system_id=system_id,
                    sample_id=sample_id,
                    start=frames[0],
                    stop=frames[1],
                    step=frames[2],
                )
                for routine in routines
            }

    return sample_id, sample_results


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
    """Analyze complete samples, optionally in parallel."""
    if workers == 0 or workers < -1:
        raise ValueError("workers must be a positive integer or -1.")
    if not tasks:
        return {}

    if workers == -1:
        worker_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else mp.cpu_count()
        )
    else:
        worker_count = workers
    worker_count = min(worker_count, len(tasks))
    logger.status(
        progress_label,
        "in progress...",
        detail=f"{len(tasks)} sample(s), {worker_count} worker(s)",
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

    pool_context = (
        nullcontext(None)
        if worker_count == 1
        else ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        )
    )
    with pool_context as executor:
        analyzed = (
            map(analyze_one, tasks)
            if executor is None
            else executor.map(analyze_one, tasks, chunksize=1)
        )
        return {
            sample_id: sample_results
            for sample_id, sample_results in iter_progress(
                analyzed,
                total=len(tasks),
                stride=progress_stride,
                logger=logger,
                label=progress_label,
            )
        }
