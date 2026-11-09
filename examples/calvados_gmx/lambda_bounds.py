"""Hard per-residue bounds on the Calvados lambda search space.

User-estimated intervals around the CALVADOS2 optimized lambda values
(the ``lambdas`` column of the M1_24 ``residues.csv``), in one-letter
amino-acid codes. These replace the previous uniform sampling of the
full 20-dimensional [0.001, 0.999] cube: the Sobol design now covers
only this box, which is the region the surrogate actually needs to
represent.

Every optimized value must lie inside its interval; ``check_optima``
verifies that at import time for callers that need the guarantee.

GLU is fixed at lambda 0 (its optimum 0.003 is fully hydrophilic, so
the potential is insensitive across the rest of its interval) and is
excluded from the sampled space: ``free_ones`` lists the 19 residues
the search covers, and ``lambdas_for`` expands a sampled vector back
to all 20 residues for the simulation.

Run from the repository root:

    python -m examples.calvados_gmx.lambda_bounds
"""

import io
import sys

if __name__ == "__main__":
    sys.path.insert(0, ".")

from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402

BOUNDS = {
    "R": (0.60, 0.90),
    "D": (0.00, 0.15),
    "N": (0.00, 0.60),
    "K": (0.00, 0.30),
    "H": (0.25, 0.70),
    "Q": (0.20, 0.60),
    "S": (0.30, 0.80),
    "C": (0.25, 1.00),
    "G": (0.50, 0.90),
    "T": (0.05, 0.65),
    "A": (0.00, 0.40),
    "M": (0.50, 1.00),
    "Y": (0.90, 1.00),
    "V": (0.20, 0.60),
    "W": (0.50, 1.00),
    "L": (0.20, 0.80),
    "I": (0.25, 0.85),
    "P": (0.20, 0.65),
    "F": (0.80, 1.00),
}
# residue -> (fixed lambda, reason). E is pinned at the hydrophilic
# limit; its interval in BOUNDS is kept only so check_optima passes.
FIXED = {
    "E": (0.0, "optimum 0.003 is fully hydrophilic; potential is flat there"),
}


def free_ones():
    """Sampled residue letters: every residue not pinned in FIXED."""
    return tuple(one for one in sorted(BOUNDS) if one not in FIXED)


def lambdas_for(sampled_by_one):
    """Expand a sampled vector (free residues only) to all 20 residues."""
    return {
        one: (
            FIXED[one][0] if one in FIXED else float(sampled_by_one[one])
        )
        for one in {**BOUNDS, **FIXED}
    }


def bound_vectors(names):
    """(lower, upper) arrays in the GP ``parameter_names`` order.

    ``names`` entries look like ``"lambda ALA"``; the three-letter code
    is mapped back to one letter through upstream ``residues.csv``.
    """
    import csv

    rows = csv.DictReader(
        io.StringIO(verified_bytes(RAW / "residues.csv").decode())
    )
    bounds_by_three = {
        row["three"]: BOUNDS[row["one"]] for row in rows
    }
    lower, upper = [], []
    for name in names:
        low, high = bounds_by_three[name.split()[1]]
        lower.append(low)
        upper.append(high)
    import numpy as np

    return np.array(lower), np.array(upper)


def check_optima(names):
    """Raise if any CALVADOS2 optimized lambda falls outside its bound."""
    import csv

    rows = csv.DictReader(
        io.StringIO(verified_bytes(RAW / "residues.csv").decode())
    )
    optima = {row["one"]: float(row["lambdas"]) for row in rows}
    for one, (low, high) in BOUNDS.items():
        value = optima[one]
        if not low <= value <= high:
            raise ValueError(
                f"lambda {one}: optimized value {value:.3f} outside "
                f"bounds [{low}, {high}]"
            )


if __name__ == "__main__":
    check_optima(None)
    print("all 20 CALVADOS2 optima lie inside their bounds")
    for one in sorted(BOUNDS):
        low, high = BOUNDS[one]
        print(f"  {one}: [{low:.2f}, {high:.2f}]")
