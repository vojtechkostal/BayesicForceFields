import inspect
from typing import Callable

import numpy as np
from MDAnalysis.guesser.tables import masses as MDA_MASSES
from scipy.constants import atomic_mass
from scipy.spatial.transform import Rotation as R

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
