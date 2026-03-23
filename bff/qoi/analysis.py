import gc
from functools import partial
import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import MDAnalysis as mda
import numpy as np

from ..domain.trainset import TrajectorySet
from .data import QoI
from ..io.logs import Logger
from ..io.progress import iter_progress
from ..topology import prepare_universe
from .routines import RuntimeRoutine, run_analysis_routines


PathLike = str | Path


warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="MDAnalysis.coordinates.DCD"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="MDAnalysis.coordinates.XDR"
)


def _cleanup_universe(universe: mda.Universe | None, *, gc_collect: bool) -> None:
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
    if gc_collect:
        gc.collect()


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
    default_dimensions = (
        None
        if universe.dimensions is None
        else np.asarray(universe.dimensions, dtype=float)
    )
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
    mol_resname: str,
    routines_by_system: Sequence[tuple[RuntimeRoutine, ...]],
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    in_memory: bool = False,
    gc_collect: bool = True,
) -> list[dict[str, QoI]]:
    """Analyze all trajectories that belong to one sample or reference set."""
    if len(routines_by_system) != len(trajectory_set.fn_trj):
        raise ValueError(
            "Analysis routine count must match the number of trajectories in the set."
        )

    results: list[dict[str, QoI]] = []
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
                mol_resname=mol_resname,
                start=start,
                stop=stop,
                step=step,
            )
            if not result:
                raise ValueError("Analysis routine returned no QoI outputs.")
            results.append(result)
        finally:
            _cleanup_universe(universe, gc_collect=gc_collect)

    return results


def analyze_trajectory_sets(
    trajectory_sets: Sequence[TrajectorySet],
    *,
    mol_resname: str,
    routines_by_system: Sequence[tuple[RuntimeRoutine, ...]],
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    workers: int = 1,
    progress_stride: int = 10,
    progress_label: str = "Trajectory QoI",
    logger: Logger | None = None,
    in_memory: bool = False,
    gc_collect: bool = True,
    chunksize: int = 1,
    maxtasksperchild: int = 1,
) -> list[list[dict[str, QoI]]]:
    """Analyze multiple trajectory sets with one shared routine setup."""
    logger = logger or Logger(progress_label)
    n_sets = len(trajectory_sets)
    if n_sets == 0:
        return []

    analyze_one = partial(
        analyze_trajectory_set,
        mol_resname=mol_resname,
        routines_by_system=routines_by_system,
        start=start,
        stop=stop,
        step=step,
        in_memory=in_memory,
        gc_collect=gc_collect,
    )

    workers = mp.cpu_count() if workers == -1 else workers
    if workers > 1:
        context = mp.get_context("spawn")
        with context.Pool(workers, maxtasksperchild=maxtasksperchild) as pool:
            iterator = pool.imap(analyze_one, trajectory_sets, chunksize=chunksize)
            qoi = list(
                iter_progress(
                    iterator,
                    total=n_sets,
                    stride=progress_stride,
                    logger=logger,
                    label=progress_label,
                )
            )
    else:
        iterator = iter_progress(
            trajectory_sets,
            total=n_sets,
            stride=progress_stride,
            logger=logger,
            label=progress_label,
        )
        qoi = [analyze_one(trajectory_set) for trajectory_set in iterator]
    return qoi


def resolve_reference_labels(blocks: Sequence[QoI]) -> tuple[str, ...] | None:
    """Resolve a stable label ordering from reference QoI blocks."""
    labeled_blocks = [block for block in blocks if block.labels is not None]
    if not labeled_blocks:
        return None

    labels: list[str] = []
    seen: set[str] = set()
    for block in labeled_blocks:
        for label in block.labels or ():
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return tuple(labels)


def stack_qoi_blocks(
    blocks: Sequence[QoI],
    *,
    labels: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Stack aligned QoI blocks into one flat numeric array."""
    if not blocks:
        return np.empty(0, dtype=float)

    values_per_label = blocks[0].values_per_label
    if any(block.values_per_label != values_per_label for block in blocks):
        raise ValueError("All QoI blocks must have the same values_per_label.")

    aligned = [block.aligned(labels) for block in blocks]
    return np.concatenate(aligned) if aligned else np.empty(0, dtype=float)


def collect_qoi_dataset(
    ref_blocks: Sequence[QoI],
    train_blocks: Sequence[Sequence[QoI]],
    *,
    qoi_metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Collect aligned reference and training arrays for one QoI."""
    labels = resolve_reference_labels(ref_blocks)
    outputs_ref = stack_qoi_blocks(ref_blocks, labels=labels)
    outputs = [stack_qoi_blocks(blocks, labels=labels) for blocks in train_blocks]

    metadata = dict(qoi_metadata or {})

    settings_kwargs = {}
    metadata_out = dict(metadata)
    first = ref_blocks[0] if ref_blocks else None
    if first is not None:
        settings_kwargs = dict(first.settings_kwargs)
        metadata_out = dict(first.metadata) | metadata_out
        metadata_out["values_per_label"] = first.values_per_label
        if labels is not None:
            metadata_out["labels"] = list(labels)

    return outputs_ref, outputs, {
        "settings_kwargs": settings_kwargs,
        "metadata": metadata_out,
    }
