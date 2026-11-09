"""Plot the bounded lambda search space within the full [0, 1] cube.

One panel per residue: the whole [0, 1] marginal range of that lambda
(the ``generated/priors`` space the posteriors live in) with the hard
``lambda_bounds.BOUNDS`` interval shaded inside it, the CALVADOS2
optimized value marked for reference, and the independent-beta prior
fit to the 70 hydrophobicity scales (``prepare_priors``), truncated to
zero outside the bounds and renormalized inside them. Written next to
the priors as ``generated/priors/search_space_bounds.png``.

Run from the repository root:

    python -m examples.calvados_gmx.plot_bounds
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, ".")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

from examples.calvados.fetch_data import RAW, verified_bytes  # noqa: E402
from examples.calvados_gmx.lambda_bounds import BOUNDS, check_optima  # noqa: E402

OUTPUT = (
    Path(__file__).resolve().parent.parent / "calvados" / "generated" / "priors"
)
TEMPER_EXPONENT = 0.5


def beta_shapes():
    """(alpha, beta) per one-letter code, from the priors' beta fit."""
    import csv
    import io

    table = pd.read_csv(OUTPUT / "beta_parameters.csv")
    rows = csv.DictReader(io.StringIO(verified_bytes(RAW / "residues.csv").decode()))
    three_to_one = {row["three"]: row["one"] for row in rows}
    shapes = {}
    for _, row in table.iterrows():
        shapes[three_to_one[row["parameter"].split()[1]]] = (
            float(row["alpha"]), float(row["beta"])
        )
    return shapes


def optima():
    import csv
    import io

    rows = csv.DictReader(
        io.StringIO(verified_bytes(RAW / "residues.csv").decode())
    )
    return {row["one"]: float(row["lambdas"]) for row in rows}


def main():
    import csv
    import io

    check_optima(None)
    optimum = optima()
    shapes = beta_shapes()
    figure, axes = plt.subplots(4, 5, figsize=(15, 8))
    grid = np.linspace(0.001, 0.999, 500)
    curves = {}
    for index, (axis, one) in enumerate(zip(axes.flat, sorted(BOUNDS))):
        low, high = BOUNDS[one]
        axis.axvspan(0, 1, color="0.92", zorder=0)
        axis.axvspan(low, high, color="C0", alpha=0.25, zorder=1)
        alpha, beta = shapes[one]
        # truncated at the bounds: zero outside, renormalized inside
        pdf = beta_dist.pdf(grid, alpha, beta)
        inside = (grid >= low) & (grid <= high)
        pdf = np.where(inside, pdf / np.trapezoid(pdf[inside], grid[inside]), 0.0)
        curves[one] = pdf
        axis.plot(grid, pdf, color="C6", linewidth=1.5, zorder=3)
        # pi ** TEMPER_EXPONENT is Beta(e*(alpha-1)+1, e*(beta-1)+1);
        # truncate and renormalize the same way
        t_alpha = TEMPER_EXPONENT * (alpha - 1) + 1
        t_beta = TEMPER_EXPONENT * (beta - 1) + 1
        tempered = beta_dist.pdf(grid, t_alpha, t_beta)
        tempered = np.where(
            inside, tempered / np.trapezoid(tempered[inside], grid[inside]), 0.0
        )
        curves[one + "_tempered"] = tempered
        axis.plot(grid, tempered, color="C2", linewidth=1.2, linestyle="--", zorder=3)
        axis.axvline(low, color="C0", linewidth=1)
        axis.axvline(high, color="C0", linewidth=1)
        axis.axvline(optimum[one], color="C3", linewidth=1.2)
        axis.set_xlim(0, 1)
        curve_max = float(pdf.max())
        axis.set_ylim(0, max(1.0, 1.05 * curve_max))
        axis.set_yticks([])
        axis.set_title(one, fontsize=11)
        width = high - low
        axis.text(
            0.5, 0.04,
            f"[{low:.2f}, {high:.2f}]  ({width:.2f})",
            ha="center", fontsize=8, transform=axis.transAxes,
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="0.92"),
        plt.Rectangle((0, 0), 1, 1, color="C0", alpha=0.25),
        plt.Line2D([], [], color="C6", linewidth=1.5),
        plt.Line2D([], [], color="C2", linewidth=1.2, linestyle="--"),
        plt.Line2D([], [], color="C3", linewidth=1.2),
    ]
    figure.legend(
        handles,
        (
            "full space [0, 1]",
            "hard search bounds",
            "beta prior (hydrophobicity scales), truncated",
            f"beta prior^{TEMPER_EXPONENT:g}, truncated",
            "CALVADOS2 optimum",
        ),
        loc="upper center",
        ncol=5,
        fontsize=9,
        frameon=False,
    )
    figure.supxlabel("Stickiness λ")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / "search_space_bounds.png", dpi=180)
    print(f"bounds figure: {OUTPUT / 'search_space_bounds.png'}")


if __name__ == "__main__":
    main()
