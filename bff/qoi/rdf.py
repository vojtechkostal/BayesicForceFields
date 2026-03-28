from typing import Tuple

import MDAnalysis as mda
import numpy as np
from scipy.ndimage import gaussian_filter

from .data import QoI
from ..tools import compute_distances, get_unitcell


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
    """Compute the radial distribution function between two atom groups."""
    distances = compute_distances(
        universe,
        atoms_ref,
        atoms_sel,
        start=start,
        stop=stop,
        step=step,
        pbc=pbc,
    )
    g, edges = np.histogram(distances.reshape(-1), range=r_range, bins=n_bins)
    g = g.astype(np.float64)
    r = 0.5 * (edges[1:] + edges[:-1])

    shell_volumes = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    if pbc:
        volume = float(np.prod(np.asarray(get_unitcell(universe)[:3], dtype=float)))
        norm = distances.size * (1.0 / volume) * shell_volumes
        g /= norm
    else:
        g /= distances.size * shell_volumes

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
        "mol_resname": mol_resname,
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
