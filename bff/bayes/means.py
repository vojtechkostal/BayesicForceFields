import numpy as np


def sigmoid(x: np.ndarray, x0: float = 3, scale: float = 5) -> np.ndarray:
    """Smoothly transition from 0 to 1 around ``x0``."""
    x = np.asarray(x)
    arg = (x - x0) * scale
    return 1 / (1 + np.exp(-arg))


def rdf_sigmoid_mean(
    n_bins: int,
    r_range: tuple,
    ref_rdf: np.ndarray,
) -> np.ndarray:
    """Create a sigmoid mean for concatenated RDFs."""
    r0, r1 = r_range
    dr_half = (r1 - r0) / (2 * n_bins)
    r = np.linspace(r0, r1, n_bins, endpoint=False) + dr_half
    n_rdf = ref_rdf.size // n_bins
    return np.tile(sigmoid(r), n_rdf)
