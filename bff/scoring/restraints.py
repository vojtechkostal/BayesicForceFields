import numpy as np
import MDAnalysis as mda

from scipy.constants import R
from scipy.ndimage import gaussian_filter

from ..tools import compute_distances


def compute_beta(T: float) -> float:
    """Returns thermodynamic beta = 1 / (R * T), where
    R is the universal gas constant and T the absolute temperature.
    """
    kB = R * 1e-3  # kJ/mol/K
    return 1 / (kB * T)


def compute_bias_weights(
        x: np.ndarray, x0: float, k: float, T: float) -> np.ndarray:
    """Returns weights of the bised collective variable.

    Parameters
    ----------
    x : np.ndarray
        Collective variable values.
    x0 : float
        Equilibrium value of the collective variable.
    k : float
        Force constant of the bias potential in kJ/mol/nm^2.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray: array of the weights

    Notes
    -----
    Restraint potential is considered in a form
    V = 0.5 * k * (x0 - x)^2
    """
    beta = compute_beta(T)
    return np.exp(0.5 * beta * k * (x0 - x) ** 2)


def compute_probability_density(
    universe: mda.Universe,
    atoms_1: mda.AtomGroup,
    atoms_2: mda.AtomGroup,
    x0: float,
    k: float,
    start: int = 0,
    stop: int = None,
    step: int = 1,
    nbins: int = 200,
    T: float = 300,
    unbias: bool = True,
    smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes (unbiased) probability density of distances.

    Parameters
    ----------
    universe : mda.Universe
    atoms : str
        Atom names defining the collective variable.
    x0 : float
        Equilibrium value of the collective variable.
    k : float
        Force constant of the bias potential in kJ/mol/nm^2.
    start : int, optional
        Start frame for the analysis. Default is 0.
    stop : int, optional
        End frame for the analysis. Default is None.
    step : int, optional
        Frame stride for the analysis. Default is 1.
    nbins : int, optional
        Number of bins for histogramming the distances. Default is 200.
    T : float, optional
        Temperature in Kelvin. Default is 300.
    unbias : bool, optional
        Whether to unbias the probability density. Default is True.

    Returns
    -------
    distances : np.ndarray
    probability : np.ndarray
    """

    d = compute_distances(
        universe, atoms_1, atoms_2, start, stop, step, pbc=True).flatten()
    x0 *= 10  # Convert to Angstroms
    k /= 100  # Convert to kJ/mol/A^2

    d_range = (x0 - 1.5, x0 + 1.5)
    probability, bin_edges = np.histogram(
        d, range=d_range, bins=nbins, density=True
    )
    distances = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Unbias the probability if desired
    if unbias:
        weights = compute_bias_weights(distances, x0, k, T)
        probability *= weights

    if smooth:
        probability = gaussian_filter(probability, sigma=1)

    return distances, probability


def compute_all_restraints(
        universe: mda.Universe, restraints: list[dict], **kwargs) -> dict:
    """Computes probability density of all restrained coordinates.

    Parameters
    ----------
    universe : mda.Universe
    restraints : list[dict]
        List of restraints specified as dictionary in a form of
        {'atoms': 'atom1 atom2', 'x0': float, 'k': float}
    **kwargs : dict
        Additional keyword arguments for the probability density calculation.
        - `start` (int)
        - `stop` (int)
        - `step` (int)
        - `nbins` (int)
        - `T` (float)
        - `unbias` (bool)

    Returns
    -------
    results : dict
        Dictionary of distances and corresponding probability densities.
    """

    results = {}
    if restraints:
        for restraint in restraints:
            atoms = restraint["atoms"]
            atomname_1, atomname_2 = atoms.split()
            atoms_1 = universe.select_atoms(f"name {atomname_1}")
            atoms_2 = universe.select_atoms(f"name {atomname_2}")

            x0 = restraint["x0"]
            k = restraint["k"]
            distances, probability = compute_probability_density(
                universe, atoms_1, atoms_2, x0, k, **kwargs
            )
            results[atoms] = [distances, probability]

    return results
