"""Open system trajectories and helpers shared by built-in and custom routines."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import SelectionError

from ..topology import prepare_universe


def select_atoms(
    universe: mda.Universe,
    selection: str,
    *,
    updating: bool = False,
    field: str = "selection",
) -> mda.AtomGroup:
    """Select a non-empty AtomGroup and name the offending option on failure."""
    try:
        atoms = universe.select_atoms(selection, updating=updating)
    except SelectionError as exc:
        raise ValueError(f"Invalid atom selection for {field}: {selection!r}.") from exc
    if len(atoms) == 0:
        raise ValueError(f"Atom selection for {field} is empty: {selection!r}.")
    return atoms


def get_unitcell(universe: mda.Universe, ts: Any = None) -> np.ndarray:
    """Return the six box dimensions of ``ts`` (default: the current frame).

    Trajectories opened by ``bff build-qoi-datasets`` fall back to the box of
    the coordinate file for frames that store no box.
    """
    ts = universe.trajectory.ts if ts is None else ts
    box = None if ts.dimensions is None else np.asarray(ts.dimensions, dtype=float)
    if (
        box is None
        or box.shape != (6,)
        or not np.all(np.isfinite(box))
        or np.any(box[:3] <= 0)
        or np.any(box[3:] <= 0)
        or np.any(box[3:] >= 180)
    ):
        raise ValueError(
            f"Frame {ts.frame} has no valid box dimensions: "
            f"{None if box is None else box.tolist()}."
        )
    return box


def open_trajectory(
    inputs: Mapping[str, Any],
    *,
    start: int,
    stop: int | None,
    step: int,
    in_memory: bool,
    context: str,
) -> tuple[mda.Universe, slice]:
    """Load topology, coordinates, and trajectory; return the frames to analyze.

    A trailing frame that cannot be read (for example from an interrupted MD
    run) is skipped with a warning. With ``in_memory`` the selected frames are
    copied into memory once, so every routine reads them without decompressing
    the trajectory again.
    """
    for role in ("topology", "coordinates", "trajectory"):
        if not isinstance(inputs.get(role), Path):
            raise ValueError(
                f"{context}, input role {role!r}: expected one path, "
                f"got {inputs.get(role)!r}."
            )
    trajectory = inputs["trajectory"]
    universe = prepare_universe(str(inputs["topology"]), str(inputs["coordinates"]))
    coordinates_box = universe.dimensions
    universe.load_new(str(trajectory))

    if coordinates_box is not None:

        def fill_missing_box(ts):
            if ts.dimensions is None:
                ts.dimensions = coordinates_box
            return ts

        universe.trajectory.add_transformations(fill_missing_box)

    if stop is None:
        n_frames = len(universe.trajectory)
        if start >= n_frames:
            raise ValueError(
                f"{context}: frame start {start} is outside trajectory "
                f"{trajectory}, which reports {n_frames} frames."
            )
        requested_last = start + ((n_frames - 1 - start) // step) * step
        last = requested_last
        while last >= start:
            try:
                universe.trajectory[last]
                break
            except (EOFError, OSError):
                last -= step
        if last < start:
            raise OSError(
                f"{context}: no readable frames remain in trajectory "
                f"{trajectory} from frame {start}."
            )
        if last < requested_last:
            warnings.warn(
                f"{context}: ignoring an unreadable trailing record in trajectory "
                f"{trajectory}; using all readable frames through frame {last}.",
                RuntimeWarning,
                stacklevel=2,
            )
        stop = last + 1

    if in_memory:
        universe.transfer_to_memory(start=start, stop=stop, step=step)
        return universe, slice(None)
    return universe, slice(start, stop, step)
