import inspect
from typing import Callable

import MDAnalysis as mda
import numpy as np
from MDAnalysis.guesser.tables import masses as MDA_MASSES
from scipy.constants import atomic_mass
from scipy.spatial.transform import Rotation as R

MASSES = np.array(list(MDA_MASSES.values()))
ELEMENTS = list(MDA_MASSES.keys())


def _normalized_dimensions(dimensions: np.ndarray | None) -> np.ndarray | None:
    """Return a usable unit-cell array or ``None`` when box data is missing."""
    if dimensions is None:
        return None

    arr = np.asarray(dimensions, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    if not np.all(np.isfinite(arr[:3])):
        return None
    if np.any(arr[:3] <= 0):
        return None
    return arr


def modify_dcd_frc_header(fn: str) -> None:
    """
    Modify the header of a DCD file to include the 'CORD' flag.

    Parameters
    ----------
    fn : str
        Path to the DCD file to be modified.

    Notes
    -----
    This function opens the file in binary mode and writes the 'CORD' flag
    at the appropriate position in the header.
    """
    flag = 'CORD'.encode()
    with open(fn, 'rb+') as f:
        f.seek(4)
        f.write(flag)


def compute_distances(
    universe: mda.Universe, ag1: mda.AtomGroup, ag2: mda.AtomGroup,
    start: int = None, stop: int = None, step: int = None,
    pbc: bool = True
) -> np.ndarray:
    """Compute distances between two AtomGroups over a trajectory."""

    start = start or 0
    stop = stop or len(universe.trajectory)
    step = step or 1
    displacements = [
        ag1.positions[:, np.newaxis] - ag2.positions
        for ts in universe.trajectory[start:stop:step]
    ]
    if not displacements:
        shape = (0, len(ag1), len(ag2))
        return np.empty(shape, dtype=float)

    displacements = np.asarray(displacements, dtype=float)
    if pbc:
        box = np.asarray(get_unitcell(universe)[:3], dtype=float)
        displacements -= np.round(displacements / box) * box
    return np.linalg.norm(displacements, axis=-1)


def get_unitcell(
    universe: mda.Universe,
    ts: mda.coordinates.base.Timestep | None = None,
) -> np.ndarray:
    """Return the current or fallback unit cell for a universe.

    Parameters
    ----------
    universe
        MDAnalysis universe that provides the default box information.
    ts
        Optional timestep whose box should be preferred.

    Returns
    -------
    numpy.ndarray
        Unit-cell vector of length 6.

    Raises
    ------
    ValueError
        If neither the timestep nor the universe provides box dimensions.
    """
    ts_dimensions = None if ts is None else _normalized_dimensions(ts.dimensions)
    if ts_dimensions is not None:
        return ts_dimensions

    dimensions = _normalized_dimensions(universe.dimensions)
    if dimensions is not None:
        return dimensions

    default = _normalized_dimensions(getattr(universe, "_bff_default_dimensions", None))
    if default is None:
        raise ValueError(
            "Trajectory frames do not define box dimensions and no fallback "
            "unit cell is available."
        )
    if ts is not None and ts.dimensions is None:
        ts.dimensions = default
    return default


def random_placement(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Randomly place a molecule within a box."""
    displacement = np.random.rand(3) * box
    rotation = R.random().as_matrix()
    coords = coords @ rotation.T
    coords += displacement
    return coords


def guess_box(n_mol: int) -> np.ndarray:
    """Approximates a cubic box size based on density of neat water."""
    mass = float(n_mol) * 18.015 * atomic_mass  # kg
    density = 1000  # kg/m^3
    length = np.cbrt(mass / density) * 1e10  # Angstroms
    return np.array([length] * 3 + [90, 90, 90])


def sigmoid(x: np.ndarray, x0: float = 3, scale: float = 5) -> np.ndarray:
    """Smoothly transitions from 0 to 1 around x=3."""
    x = np.asarray(x)
    arg = (x - x0) * scale
    return 1 / (1 + np.exp(- arg))


def rdf_sigmoid_mean(n_bins: int, r_range: tuple, ref_rdf: np.ndarray) -> np.ndarray:
    """Create a sigmoid function for concatenated RDFs."""
    r0, r1 = r_range
    dr_half = (r1 - r0) / (2 * n_bins)
    r = np.linspace(r0, r1, n_bins, endpoint=False) + dr_half
    n_rdf = ref_rdf.size // n_bins
    return np.tile(sigmoid(r), n_rdf)


def extract_defaults(fn: Callable) -> dict[str, object]:
    """
    Extract default values from the function signature.

    Parameters
    ----------
    fn : callable
        The function from which to extract default values.

    Returns
    -------
    dict
        A dictionary with parameter names as keys and their default values.
    """
    sig = inspect.signature(fn)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
