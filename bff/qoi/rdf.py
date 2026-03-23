from typing import Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance
from scipy.ndimage import gaussian_filter

from .data import QoI
from ..tools import get_unitcell


def compute_rdf(
    universe: mda.Universe,
    atoms_ref: mda.AtomGroup,
    atoms_sel: mda.AtomGroup,
    r_range: Tuple[float, float] = (0, 10),
    n_bins: int = 200,
    pbc: bool = True,
    start: int = None,
    stop: int = None,
    step: int = None,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function between two atom groups.

    Distances are accumulated frame-by-frame using ``capped_distance`` up to
    the maximum requested RDF radius, which avoids building full pair-distance
    matrices and keeps the memory footprint small.
    """
    start = 0 if start is None else start
    stop = len(universe.trajectory) if stop is None else stop
    step = 1 if step is None else step

    edges = np.linspace(r_range[0], r_range[1], n_bins + 1, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    shell_volumes = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    r = 0.5 * (edges[1:] + edges[:-1])

    normalization = 0.0
    n_ref = len(atoms_ref)
    n_sel = len(atoms_sel)
    if n_ref == 0 or n_sel == 0:
        return r, counts

    min_cutoff = None if r_range[0] <= 0 else float(r_range[0])
    max_cutoff = float(r_range[1])

    for ts in universe.trajectory[start:stop:step]:
        box = get_unitcell(universe, ts) if pbc else None
        _, distances = capped_distance(
            atoms_ref,
            atoms_sel,
            max_cutoff=max_cutoff,
            min_cutoff=min_cutoff,
            box=box,
            return_distances=True,
        )
        if len(distances) > 0:
            counts += np.histogram(distances, bins=edges)[0]

        if pbc:
            volume = float(np.prod(np.asarray(box[:3], dtype=float)))
            normalization += n_ref * n_sel / volume
        else:
            normalization += n_ref * n_sel

    if normalization > 0:
        g = counts / (shell_volumes * normalization)
    else:
        g = counts

    if smooth:
        g = gaussian_filter(g, sigma=3)

    return r, g


def compute_all_rdfs(
    universe: mda.Universe,
    mol_resname: str,
    solvent_sel: str = "resname SOL HOH WAT and name O*",
    r_range: Tuple[float, float] = (0, 10),
    n_bins: int = 200,
    pbc: bool = True,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    smooth: bool = True,
) -> QoI:
    """Compute solvent RDF QoIs around all solute atom types."""
    mol = universe.select_atoms(f"resname {mol_resname}")
    mask = mol.masses > 0.5
    mol_atomtypes = np.unique(mol[mask].types)

    rdf_results: dict[str, np.ndarray] = {}
    atoms_solvent = universe.select_atoms(solvent_sel)
    for atomtype in mol_atomtypes:
        atoms_reference = universe.select_atoms(f"type {atomtype}")
        r, g = compute_rdf(
            universe,
            atoms_reference,
            atoms_solvent,
            r_range=r_range,
            n_bins=n_bins,
            pbc=pbc,
            start=start,
            stop=stop,
            step=step,
            smooth=smooth,
        )
        rdf_results[atomtype] = np.array([r, g])

    atomtypes = tuple(sorted(rdf_results))
    values = np.concatenate(
        [
            np.asarray(rdf_results[atomtype][1], dtype=float).reshape(-1)
            for atomtype in atomtypes
        ]
    )
    metadata = {
        "solvent_sel": solvent_sel,
        "r_range": tuple(r_range),
        "n_bins": int(n_bins),
        "pbc": bool(pbc),
        "smooth": bool(smooth),
    }
    return QoI(
        name="rdf",
        values=values,
        labels=atomtypes,
        values_per_label=int(n_bins),
        settings_kwargs=metadata,
    )
