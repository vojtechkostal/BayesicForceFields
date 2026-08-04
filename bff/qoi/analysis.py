"""Execution helpers for explicit QoI routine inputs."""

from __future__ import annotations

import gc
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import MDAnalysis as mda

from ..io.logs import Logger
from ..io.progress import iter_progress
from ..tools import _normalized_dimensions
from ..topology import prepare_universe
from .data import QoI
from .routines import RuntimeRoutine, run_analysis_routine


def _cleanup_universe(universe: mda.Universe | None) -> None:
    if universe is None:
        return
    trajectory = getattr(universe, "trajectory", None)
    close = getattr(trajectory, "close", None)
    if callable(close):
        close()


def _require_path(
    inputs: Mapping[str, Any],
    role: str,
    *,
    system_id: str,
    sample_id: str,
) -> Path:
    value = inputs.get(role)
    if not isinstance(value, Path):
        raise ValueError(
            f"System {system_id!r}, sample {sample_id!r}, input role {role!r}: "
            f"expected one path, got {value!r}."
        )
    return value


def analyze_system_inputs(
    task: tuple[str, str, dict[str, Any]],
    *,
    routines: Sequence[RuntimeRoutine],
    start: int,
    stop: int | None,
    step: int,
    in_memory: bool,
) -> tuple[str, str, dict[str, QoI]]:
    """Run all applicable routines while sharing one Universe."""
    sample_id, system_id, inputs = task
    universe: mda.Universe | None = None
    routine_start, routine_stop, routine_step = start, stop, step
    try:
        if any(routine.loader == "mdanalysis" for routine in routines):
            topology = _require_path(
                inputs, "topology", system_id=system_id, sample_id=sample_id
            )
            coordinates = _require_path(
                inputs, "coordinates", system_id=system_id, sample_id=sample_id
            )
            trajectory = _require_path(
                inputs, "trajectory", system_id=system_id, sample_id=sample_id
            )
            universe = prepare_universe(str(topology), str(coordinates), dt=1)
            default_dimensions = _normalized_dimensions(universe.dimensions)
            universe.load_new(str(trajectory))
            universe._bff_default_dimensions = default_dimensions
            if in_memory:
                universe.transfer_to_memory(start=start, stop=stop, step=step)
                if default_dimensions is not None:
                    for ts in universe.trajectory:
                        if ts.dimensions is None:
                            ts.dimensions = default_dimensions
                routine_start, routine_stop, routine_step = 0, None, 1

        results: dict[str, QoI] = {}
        for routine in routines:
            qoi = run_analysis_routine(
                routine,
                universe=universe,
                inputs=inputs,
                system_id=system_id,
                sample_id=sample_id,
                start=routine_start,
                stop=routine_stop,
                step=routine_step,
            )
            results[routine.name] = qoi
        return sample_id, system_id, results
    finally:
        _cleanup_universe(universe)


def _iter_results(
    tasks: Sequence[tuple[str, str, dict[str, Any]]],
    *,
    analyze_one: Any,
    workers: int,
    maxtasksperchild: int,
) -> Iterable[tuple[str, str, dict[str, QoI]]]:
    if workers <= 1:
        yield from (analyze_one(task) for task in tasks)
        return
    with mp.get_context().Pool(
        workers, maxtasksperchild=maxtasksperchild
    ) as pool:
        yield from pool.imap(analyze_one, tasks, chunksize=1)


def analyze_input_sets(
    tasks: Sequence[tuple[str, str, dict[str, Any]]],
    *,
    routines_by_system: Mapping[str, Sequence[RuntimeRoutine]],
    start: int,
    stop: int | None,
    step: int,
    workers: int,
    progress_stride: int,
    progress_label: str,
    logger: Logger,
    in_memory: bool,
    gc_collect: bool,
    maxtasksperchild: int,
) -> dict[str, dict[str, dict[str, QoI]]]:
    """Analyze explicit (sample, system, inputs) tasks."""
    grouped_tasks: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for task in tasks:
        grouped_tasks.setdefault(task[1], []).append(task)
    results: dict[str, dict[str, dict[str, QoI]]] = {}
    completed = 0
    total = len(tasks)
    worker_count = mp.cpu_count() if workers == -1 else workers
    if worker_count == 0 or worker_count < -1:
        raise ValueError("workers must be a positive integer or -1.")

    for system_id in sorted(grouped_tasks):
        routines = tuple(routines_by_system.get(system_id, ()))
        analyze_one = partial(
            analyze_system_inputs,
            routines=routines,
            start=start,
            stop=stop,
            step=step,
            in_memory=in_memory,
        )
        analyzed = _iter_results(
            grouped_tasks[system_id],
            analyze_one=analyze_one,
            workers=worker_count,
            maxtasksperchild=maxtasksperchild,
        )
        for sample_id, returned_system_id, qois in iter_progress(
            analyzed,
            total=len(grouped_tasks[system_id]),
            stride=progress_stride,
            logger=logger,
            label=f"{progress_label} [{system_id}]",
        ):
            results.setdefault(sample_id, {})[returned_system_id] = qois
            completed += 1
            if gc_collect and (
                completed % progress_stride == 0 or completed == total
            ):
                gc.collect()
    return results
