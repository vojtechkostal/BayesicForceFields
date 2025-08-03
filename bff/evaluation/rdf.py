"""Handling radial distribution functions (RDFs)"""

import numpy as np
import MDAnalysis as mda
from scipy.ndimage import gaussian_filter

from ..tools import compute_distances


def compute_rdf(
    universe: mda.Universe,
    atoms_ref: mda.AtomGroup,
    atoms_sel: mda.AtomGroup,
    r_range: list = (0, 10),
    n_bins: int = 200,
    pbc: bool = True,
    start: int = None,
    stop: int = None,
    step: int = None,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the radial distribution function (RDF) between two atom groups.

    Parameters
    ----------
    universe : mda.Universe
    atoms_ref : mda.AtomGroup
    atoms_sel : mda.AtomGroup
    r_range : tuple, optional
        Range of distances for computing the RDF in Angstroms.
        Default is (0, 10).
    n_bins : int, optional
        Number of bins for histogramming the RDF. Default is 200.
    pbc : bool, optional
        Whether to take periodic boundary conditions into account.
        Default is True.
    normalize_vol : bool, optional
        Whether to normalize the RDF by volume. Default is True.
    start : int, optional
        Number of frames to discard from the beginning of the trajectory.
        Default is 0.
    stop : int, optional
        Frame index to finish at. Default is None.
    step : int, optional
        Frame stride for sampling frames. Default is 1.
    smooth : bool, optional
        Whether to smoothen the RDF using a spline. Default is True.

    Returns
    -------
    r : numpy.ndarray
        Array of distances.
    g : numpy.ndarray
        Array of (normalized) probability density values.
    """

    unitcell = universe.dimensions if pbc else None

    # Compute distances (under periodic boundary conditions)
    distances = compute_distances(
        universe, atoms_ref, atoms_sel, start, stop, step, pbc=pbc)

    # Compute RDF
    g, edges = np.histogram(distances.flatten(), range=r_range, bins=n_bins)
    g = g.astype(np.float64)
    r = 0.5 * (edges[1:] + edges[:-1])

    # Normalize
    volume_shells = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    V = unitcell[:3].prod()
    norm = distances.size * np.sum(1 / V) * volume_shells
    g /= norm

    # Smoothen the rdf
    if smooth:
        g = gaussian_filter(g, sigma=3)

    return r, g


def compute_all_rdfs(
    universe: mda.Universe,
    mol_resname: str,
    solvent_sel: str = "resname SOL HOH WAT and name O*",
    **kwargs,
) -> dict:
    """
    Compute radial distribution functions (RDFs) of solvent around
    a molecule's atomtypes

    Parameters
    ----------
    universe : MDAnalysis universe
    molecule_resname : str
        resname of the molecule
    solvent_sel : str
        selection string to reprecent solvent atoms.
        Default is "resname SOL HOH WAT and name O*"
    **kwargs : dict
        keyword arguments for the underlying compute_rdf function
        - `r_range` : tuple, optional
        - `n_bins` : int, optional
        - `pbc` : bool, optional
        - `normalize_vol` : bool, optional
        - `start` : int, optional
        - `stop` : int, optional
        - `step` : int, optional
        - `smooth` : bool, optional

    Returns
    -------
    rdfs : dict
        dictionary of solvent rdfs around each atomtype.
        Formated as [r, g], where r, g are distances, rdf respectively
    """

    # Get unique atomtypes of the molecule
    for residue in universe.residues:
        if residue.resname == mol_resname:
            mol_atomtypes = np.unique(residue.atoms.types)

    rdfs = {}
    atoms_solvent = universe.select_atoms(solvent_sel)
    for atomtype in mol_atomtypes:
        atoms_reference = universe.select_atoms(f"type {atomtype}")
        r, g = compute_rdf(universe, atoms_reference, atoms_solvent, **kwargs)
        rdfs[atomtype] = [r, g]

    return rdfs
