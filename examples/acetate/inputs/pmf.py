"""Custom QoI routine: the calcium-acetate PMF along the Ca-C2 distance."""

import numpy as np

from bff.qoi import QoI


def read_pmf(*, inputs, options) -> QoI:
    """Read a two-column PMF file (distance, free energy) onto a fixed grid.

    The profile is interpolated onto ``points`` distances spanning ``range``
    (nm), so the reference and every sample share one grid even if their
    files use different bins. A PMF is defined only up to a constant, so the
    profile is shifted to zero mean over that window: the offset that best
    aligns two profiles.
    """
    lower, upper = options.get("range", (0.27, 0.65))
    points = int(options.get("points", 39))
    distance, free_energy = np.loadtxt(
        inputs["pmf"], comments="#", usecols=(0, 1), unpack=True
    )
    if lower < distance.min() or upper > distance.max():
        raise ValueError(
            f"range [{lower}, {upper}] nm is outside the profile "
            f"[{distance.min()}, {distance.max()}] nm in {inputs['pmf']}."
        )

    grid = np.linspace(lower, upper, points)
    values = np.interp(grid, distance, free_energy)
    return QoI(
        name="pmf",
        values=values - values.mean(),
        labels=("Ca-C2",),
        values_per_label=points,
        settings={"distance_nm": grid.round(6).tolist()},
    )
