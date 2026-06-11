import numpy as np
from matplotlib.legend import Legend

from bff.bayes.priors import Priors
from bff.bayes.results import PosteriorResults
from bff.domain.specs import Specs
from bff.plotting import plot_corner, plot_qoi_marginals


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


def test_plot_qoi_marginals_stacks_qoi_profiles_without_annotations(
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
    values = np.linspace(-0.6, 0.6, 80)
    results = PosteriorResults(
        values.reshape(20, 4, 1),
        priors=Priors.from_bounds(
            [[-1.0, 1.0]],
            names=["charge A"],
        ),
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

    plot_qoi_marginals(
        results,
        specs,
        {
            "rdf": -values**2,
            "density": -(values - 0.3) ** 2,
        },
    )

    figure, axes = plotted[0]
    ax = np.atleast_1d(axes)[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A", "B"]
    assert len(ax.texts) == 0
    assert len(ax.collections) >= 4
    outline = next(line for line in ax.lines if len(line.get_ydata()) == 400)
    assert outline.get_ydata().min() == -1.0
    assert outline.get_ydata().max() == 1.0
    legends = figure.findobj(Legend)
    assert [
        [text.get_text() for text in legend.get_texts()]
        for legend in legends
    ] == [
        ["prior", "posterior", "bounds"],
        ["rdf", "density"],
    ]
    plotting.plt.close(figure)
