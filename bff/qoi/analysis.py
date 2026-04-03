import gc
import multiprocessing as mp
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import MDAnalysis as mda

from ..domain.trainset import TrajectorySet
from ..io.logs import Logger
from ..io.progress import iter_progress
from ..tools import _normalized_dimensions
from ..topology import prepare_universe
from .data import QoI
from .routines import RuntimeRoutine, run_analysis_routines

PathLike = str | Path


warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="MDAnalysis.coordinates.XDR"
)


def _cleanup_universe(universe: mda.Universe | None) -> None:
    """Release trajectory resources after one analysis task."""
    if universe is not None:
        trajectory = getattr(universe, "trajectory", None)
        if trajectory is not None:
            close = getattr(trajectory, "close", None)
            if callable(close):
                close()
        raw_trajectory = getattr(universe, "_trajectory", None)
        if raw_trajectory is not None and raw_trajectory is not trajectory:
            close = getattr(raw_trajectory, "close", None)
            if callable(close):
                close()


def _prepare_universe(
    fn_topol: PathLike,
    fn_coord: PathLike,
    fn_trj: PathLike,
    *,
    start: int,
    stop: int | None,
    step: int,
    in_memory: bool,
) -> mda.Universe:
    """Prepare one MDAnalysis universe for trajectory analysis."""
    universe = prepare_universe(str(fn_topol), str(fn_coord), dt=1)
    default_dimensions = _normalized_dimensions(universe.dimensions)
    universe.load_new(str(fn_trj))
    universe._bff_default_dimensions = default_dimensions
    if in_memory:
        universe.transfer_to_memory(start=start, stop=stop, step=step)
        if default_dimensions is not None:
            for ts in universe.trajectory:
                if ts.dimensions is None:
                    ts.dimensions = default_dimensions
    return universe


def analyze_trajectory_set(
    trajectory_set: TrajectorySet,
    *,
    routines_by_system: Sequence[tuple[RuntimeRoutine, ...]],
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    in_memory: bool = False,
) -> list[dict[str, QoI]]:
    """Analyze all trajectories that belong to one sample or reference set."""
    if len(routines_by_system) != len(trajectory_set.fn_trj):
        raise ValueError(
            "Analysis routine count must match the number of trajectories in the set."
        )

    results: list[dict[str, QoI]] = []
    routine_start = 0 if in_memory else start
    routine_stop = None if in_memory else stop
    routine_step = 1 if in_memory else step
    for fn_topol, fn_coord, fn_trj, routines in zip(
        trajectory_set.fn_topol,
        trajectory_set.fn_coord,
        trajectory_set.fn_trj,
        routines_by_system,
    ):
        universe = None
        try:
            universe = _prepare_universe(
                fn_topol,
                fn_coord,
                fn_trj,
                start=start,
                stop=stop,
                step=step,
                in_memory=in_memory,
            )
            result = run_analysis_routines(
                routines,
                universe=universe,
                start=routine_start,
                stop=routine_stop,
                step=routine_step,
            )
            if not result:
                raise ValueError("Analysis routine returned no QoI outputs.")
            results.append(result)
        finally:
            _cleanup_universe(universe)
    return results


def _maybe_collect_garbage(
    completed: int,
    total: int,
    *,
    gc_collect: bool,
    progress_stride: int,
) -> None:
    """Run the garbage collector at coarse progress intervals when requested."""
    if not gc_collect:
        return
    if (completed % progress_stride == 0) or (completed == total):
        gc.collect()


def _iter_analyzed_sets(
    trajectory_sets: Sequence[TrajectorySet],
    *,
    analyze_one: Any,
    workers: int,
    maxtasksperchild: int,
) -> Iterable[list[dict[str, QoI]]]:
    """Yield analyzed trajectory sets from the serial or multiprocessing path."""
    if workers <= 1:
        yield from (analyze_one(trajectory_set) for trajectory_set in trajectory_sets)
        return

    context = mp.get_context()
    with context.Pool(workers, maxtasksperchild=maxtasksperchild) as pool:
        yield from pool.imap(
            analyze_one,
            trajectory_sets,
            chunksize=1,
        )


def analyze_trajectory_sets(
    trajectory_sets: Sequence[TrajectorySet],
    *,
    routines_by_system: Sequence[tuple[RuntimeRoutine, ...]],
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    workers: int = 1,
    progress_stride: int = 10,
    progress_label: str = "Trajectory QoI",
    logger: Logger | None = None,
    in_memory: bool = False,
    gc_collect: bool = False,
    maxtasksperchild: int = 100,
) -> list[list[dict[str, QoI]]]:
    """Analyze multiple trajectory sets with one shared routine setup."""
    logger = logger or Logger(progress_label)
    n_sets = len(trajectory_sets)
    if n_sets == 0:
        return []

    analyze_one = partial(
        analyze_trajectory_set,
        routines_by_system=routines_by_system,
        start=start,
        stop=stop,
        step=step,
        in_memory=in_memory,
    )

    workers = mp.cpu_count() if workers == -1 else workers
    completed_sets = _iter_analyzed_sets(
        trajectory_sets,
        analyze_one=analyze_one,
        workers=workers,
        maxtasksperchild=maxtasksperchild,
    )
    qoi = []
    for completed, result in enumerate(
        iter_progress(
            completed_sets,
            total=n_sets,
            stride=progress_stride,
            logger=logger,
            label=progress_label,
        ),
        start=1,
    ):
        qoi.append(result)
        _maybe_collect_garbage(
            completed,
            n_sets,
            gc_collect=gc_collect,
            progress_stride=progress_stride,
        )
    return qoi
