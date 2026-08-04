import numpy as np
import pytest
from matplotlib.legend import Legend

from bff.bayes.priors import Priors
from bff.bayes.results import PosteriorResults
from bff.domain.specs import Specs
from bff.plotting import plot_corner, plot_marginals, plot_qoi_marginals


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


def test_plot_marginals_annotates_posterior_mean(monkeypatch) -> None:
    specs = Specs(
        {"bounds": {"sigma A": [0.0, 2.0]}, "charge_constraints": []}
    )
    values = np.concatenate([np.linspace(0.1, 0.3, 36), np.full(4, 1.8)])
    results = PosteriorResults(
        values.reshape(10, 4, 1),
        priors=Priors.from_bounds([[0.0, 2.0]], names=["sigma A"]),
        sample_labels=["sigma A"],
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
    plot_marginals(results, specs)

    figure, axes = plotted[0]
    ax = np.atleast_1d(axes)[0]
    assert [text.get_text() for text in ax.texts] == [f"{values.mean():.3f}"]
    plotting.plt.close(figure)


@pytest.mark.parametrize(
    ("count", "figsize", "bounds"),
    [
        (4, (3.0, 2.4), (-0.001, 0.001)),
        (8, (5.0, 2.8), (-1_000_000.0, 1_000_000.0)),
    ],
)
def test_marginal_annotation_lanes_do_not_overlap(
    count: int,
    figsize: tuple[float, float],
    bounds: tuple[float, float],
) -> None:
    import bff.plotting as plotting

    figure, ax = plotting.plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, count - 0.5)
    ax.set_xticks(range(count), [f"long parameter label {i}" for i in range(count)])
    ax.plot([], [], label="posterior with a long label")
    ax.legend(loc="upper center")
    means = np.linspace(bounds[0] * 0.1, bounds[1] * 0.1, count)
    artists = plotting._layout_marginal_mean_annotations(
        figure,
        [ax],
        [
            (ax, index, mean, bounds[0], bounds[1])
            for index, mean in enumerate(means)
        ],
    )

    renderer = figure.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer) for artist in artists]
    assert not any(
        first.overlaps(second)
        for index, first in enumerate(boxes)
        for second in boxes[index + 1 :]
    )
    plotting.plt.close(figure)
