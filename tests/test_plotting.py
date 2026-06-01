import numpy as np

from bff.bayes.results import PosteriorResults
from bff.domain.specs import Specs
from bff.plotting import plot_corner


def test_plot_corner_includes_reconstructed_implicit_charges(
    monkeypatch,
) -> None:
    specs = Specs(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "charge B": [-1.0, 1.0],
            },
            "charge_constraints": [
                {
                    "selection": "name A B",
                    "target": 0.0,
                    "scope": "residue",
                    "implicit": "charge B",
                    "coefficients": {"charge A": 1.0, "charge B": 1.0},
                }
            ],
        }
    )
    results = PosteriorResults(
        np.linspace(-0.4, 0.4, 40).reshape(10, 4, 1),
        sample_labels=["charge A"],
        specs=specs,
    )
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    import bff.plotting as plotting

    subplots = plotting.plt.subplots
    plotted = []

    def record_subplots(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        plotted.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plotting.plt, "subplots", record_subplots)
    monkeypatch.setattr(plotting.plt, "show", lambda: None)

    plot_corner(results)

    figure, axes = plotted[0]
    assert axes.shape == (2, 2)
    assert axes[1, 0].get_ylabel() == "charge B"
    plotting.plt.close(figure)
