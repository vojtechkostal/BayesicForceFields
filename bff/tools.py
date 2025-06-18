import numpy as np
from scipy.constants import atomic_mass
from scipy.spatial.transform import Rotation as R
from scipy.stats import chi2

    
def modify_dcd_frc_header(fn):
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
        universe, ag1, ag2, start=None, stop=None, step=None, pbc=True):
    """Compute distances between two AtomGroups over a trajectory."""

    start = start or 0
    stop = stop or len(universe.trajectory)
    step = step or 1
    displacements = [
        ag1.positions[:, np.newaxis] - ag2.positions
        for ts in universe.trajectory[start:stop:step]
    ]
    displacements = np.array(displacements)
    if pbc:
        box = universe.dimensions[:3]
        displacements -= np.round(displacements / box) * box
    distances = np.linalg.norm(displacements, axis=-1)
    return distances


def random_placement(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Randomly place a molecule within a box."""
    displacement = np.random.rand(3) * box
    rotation = R.random().as_matrix()
    coords = coords @ rotation.T
    coords += displacement
    return coords


def guess_box(n_mol: int):
    """Approximates a cubic box size based on density of neat water."""
    mass = float(n_mol) * 18.015 * atomic_mass  # kg
    density = 1000  # kg/m^3
    length = np.cbrt(mass / density) * 1e10  # Angstroms
    return np.array([length] * 3 + [90, 90, 90])


def sigmoid(x, x0=3, scale=5):
    """Smoothly transitions from 0 to 1 around x=3."""
    x = np.asarray(x)
    arg = (x - x0) * scale
    return 1 / (1 + np.exp(- arg))


def sample_within_confidence(samples, committee_size, confidence=0.68):
    if committee_size == 1:
        return np.median(samples, axis=0).reshape(1, -1)

    median = np.median(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    dim = samples.shape[1]
    # Mahalanobis distance threshold corresponding to the desired confidence level
    threshold = chi2.ppf(confidence, df=dim)

    accepted = []
    max_attempts = committee_size * 1000  # Can reduce this with better efficiency

    for _ in range(max_attempts):
        sample = np.random.multivariate_normal(median, cov)
        mahalanobis_sq = (sample - median) @ inv_cov @ (sample - median)
        if mahalanobis_sq <= threshold:
            accepted.append(sample)
        if len(accepted) == committee_size:
            break

    if len(accepted) < committee_size:
        raise RuntimeError(f"Could only collect {len(accepted)} samples within {confidence*100:.1f}% confidence region.")

    return np.array(accepted)