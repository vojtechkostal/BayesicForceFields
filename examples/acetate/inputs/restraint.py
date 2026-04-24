"""Example custom QoI routine for biased acetate trajectories."""

from typing import Any, Sequence

import numpy as np
from MDAnalysis.lib.distances import capped_distance

from bff.qoi.data import QoI
from bff.tools import get_unitcell


def distance_distribution(
    universe: Any,
    *,
    atom_pair: Sequence[str] = ("C2", "CAL"),
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    r_range: tuple[float, float] = (1.0, 7.0),
    n_bins: int = 200,
) -> QoI:
    """Compute one capped-distance distribution for a requested atom-name pair.

    Parameters
    ----------
    universe
        MDAnalysis universe with the trajectory already loaded.
    atom_pair
        Atom-name pair such as ``["C2", "CAL"]``.
    start, stop, step
        Frame slice used for the analysis.
    r_range
        Histogram range in Angstrom.
    n_bins
        Number of histogram bins.

    Returns
    -------
    QoI
        Flattened distance distribution for the requested pair.
    """
    if len(atom_pair) != 2:
        raise ValueError("'atom_pair' must contain exactly two atom names.")

    atom_1, atom_2 = atom_pair
    atoms_1 = universe.select_atoms(f"name {atom_1}")
    atoms_2 = universe.select_atoms(f"name {atom_2}")
    if len(atoms_1) == 0 or len(atoms_2) == 0:
        raise ValueError(
            f"Could not resolve atom pair ({atom_1!r}, {atom_2!r}) in the universe."
        )

    edges = np.linspace(r_range[0], r_range[1], n_bins + 1, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    start = 0 if start is None else start
    stop = len(universe.trajectory) if stop is None else stop
    step = 1 if step is None else step

    min_cutoff = None if r_range[0] <= 0 else float(r_range[0])
    max_cutoff = float(r_range[1])
    for ts in universe.trajectory[start:stop:step]:
        box = get_unitcell(universe, ts)
        _, distances = capped_distance(
            atoms_1,
            atoms_2,
            max_cutoff=max_cutoff,
            min_cutoff=min_cutoff,
            box=box,
            return_distances=True,
        )
        if len(distances) > 0:
            counts += np.histogram(distances, bins=edges)[0]

    hist = counts.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total

    return QoI(
        name="dist",
        values=hist,
        labels=(f"{atom_1} {atom_2}",),
        values_per_label=int(n_bins),
        settings={
            "atom_pair": [atom_1, atom_2],
            "n_bins": int(n_bins),
            "r_range": tuple(r_range),
        },
    )
