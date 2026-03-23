import numpy as np
import MDAnalysis as mda
import inspect
from typing import Callable
from scipy.constants import atomic_mass
from scipy.spatial.transform import Rotation as R

from MDAnalysis.guesser.tables import masses as MDA_MASSES
from MDAnalysis.lib.distances import distance_array

MASSES = np.array(list(MDA_MASSES.values()))
ELEMENTS = list(MDA_MASSES.keys())


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
    distances = []
    for ts in universe.trajectory[start:stop:step]:
        box = get_unitcell(universe, ts) if pbc else None
        distances.append(distance_array(ag1.positions, ag2.positions, box=box))
    return np.asarray(distances)


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
    if ts is not None and ts.dimensions is not None:
        return np.asarray(ts.dimensions, dtype=float)

    dimensions = universe.dimensions
    if dimensions is not None:
        return np.asarray(dimensions, dtype=float)

    default = getattr(universe, "_bff_default_dimensions", None)
    if default is None:
        raise ValueError(
            "Trajectory frames do not define box dimensions and no fallback "
            "unit cell is available."
        )
    if ts is not None and ts.dimensions is None:
        ts.dimensions = np.asarray(default, dtype=float)
    return np.asarray(default, dtype=float)


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
