import numpy as np
import pytest
from matplotlib.legend import Legend

from bff.bayes.priors import Priors
from bff.bayes.results import PosteriorResults
from bff.domain.specs import Specs
from bff.plotting import (
    _marginal_figure,
    _marginal_panel_sections,
    plot_corner,
    plot_marginals,
    plot_qoi_marginals,
)


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


def test_plot_corner_limits_samples_used_by_kde(monkeypatch) -> None:
    import bff.plotting as plotting

    samples = np.column_stack([
        np.linspace(-1.0, 1.0, 100),
        np.linspace(-0.5, 1.5, 100) ** 2,
    ])
    kde_sample_counts = []
    real_kde = plotting.gaussian_kde

    def record_kde(values, *args, **kwargs):
        kde_sample_counts.append(np.asarray(values).shape[-1])
        return real_kde(values, *args, **kwargs)

    monkeypatch.setattr(plotting, "gaussian_kde", record_kde)
    monkeypatch.setattr(plotting.plt, "show", lambda: None)

    plot_corner(samples, max_samples=20)

    assert kde_sample_counts == [20, 20, 20]
    plotting.plt.close("all")


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

    monkeypatch.setattr(plotting.plt, "show", lambda: None)

    plot_qoi_marginals(
        results,
        specs,
        {
            "rdf": -values**2,
            "density": -(values - 0.3) ** 2,
        },
    )

    figure = plotting.plt.gcf()
    ax = figure.axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A", "B"]
    assert len(ax.texts) == 0
    assert len(ax.collections) >= 4
    outline = next(line for line in ax.lines if len(line.get_ydata()) == 400)
    assert outline.get_ydata().min() == -1.0
    assert outline.get_ydata().max() == 1.0
    prior = next(
        collection
        for collection in ax.collections
        if collection.get_zorder() == 1
    )
    posterior = [
        collection for collection in ax.collections if collection.get_zorder() == 2
    ]
    assert posterior
    assert all(item.get_zorder() > prior.get_zorder() for item in posterior)
    legends = figure.findobj(Legend)
    assert [
        [text.get_text() for text in legend.get_texts()]
        for legend in legends
    ] == [
        ["prior", "posterior", "bounds"],
        ["rdf", "density"],
    ]
    plotting.plt.close(figure)


def test_plot_marginals_annotates_posterior_mode(monkeypatch) -> None:
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

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    plot_marginals(results, specs)

    figure = plotting.plt.gcf()
    ax = figure.axes[0]
    annotated_value = float(ax.texts[0].get_text())
    assert annotated_value < 0.4
    assert annotated_value != pytest.approx(values.mean(), abs=0.01)
    annotation = ax.texts[0]
    assert annotation.get_position()[0] == 0
    assert ax.get_ylim()[0] < annotation.get_position()[1] < 0.0
    assert annotation.get_color() == "tab:red"
    assert annotation.get_fontweight() == "bold"
    prior, posterior = ax.collections[:2]
    assert posterior.get_zorder() > prior.get_zorder()
    plotting.plt.close(figure)


def test_plot_marginals_limits_samples_used_by_kde(monkeypatch) -> None:
    specs = Specs(
        {"bounds": {"sigma A": [0.0, 2.0]}, "charge_constraints": []}
    )
    values = np.linspace(0.1, 1.9, 100)
    results = PosteriorResults(
        values.reshape(25, 4, 1),
        sample_labels=["sigma A"],
        specs=specs,
    )
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    import bff.plotting as plotting

    sample_counts = []
    real_kde = plotting.gaussian_kde

    def record_kde(data, *args, **kwargs):
        sample_counts.append(np.asarray(data).shape[-1])
        return real_kde(data, *args, **kwargs)

    monkeypatch.setattr(plotting, "gaussian_kde", record_kde)
    monkeypatch.setattr(plotting.plt, "show", lambda: None)

    plot_marginals(results, specs, max_samples=17)

    assert sample_counts == [17]
    plotting.plt.close("all")


def test_plot_marginal_charge_mode_uses_at_most_three_decimals(
    monkeypatch,
) -> None:
    specs = Specs(
        {"bounds": {"charge A": [-0.2, 0.2]}, "charge_constraints": []}
    )
    values = np.linspace(-0.1241, -0.1231, 40)
    results = PosteriorResults(
        values.reshape(10, 4, 1),
        sample_labels=["charge A"],
        specs=specs,
    )
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    import bff.plotting as plotting

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    plot_marginals(results, specs)

    figure = plotting.plt.gcf()
    ax = figure.axes[0]
    annotation = ax.texts[0].get_text()
    assert float(annotation) == pytest.approx(-0.1236, abs=0.001)
    assert len(annotation.partition(".")[2]) <= 3
    plotting.plt.close(figure)


def test_marginal_panels_balance_rows_and_separate_defines() -> None:
    sections = _marginal_panel_sections(
        [
            "charge A",
            "charge B",
            "define VSA",
            "define VSD",
            "sigma O",
            "sigma H",
        ]
    )

    assert sections == [
        ("charge", [[[0, 1]]]),
        ("define", [[[2]], [[3]]]),
        ("sigma", [[[4, 5]]]),
    ]
    assert _marginal_panel_sections([f"charge A{i}" for i in range(6)]) == [
        ("charge", [[[0, 1, 2]], [[3, 4, 5]]]),
    ]
    assert _marginal_panel_sections([f"define D{i}" for i in range(4)]) == [
        ("define", [[[0], [1]], [[2], [3]]]),
    ]


def test_marginal_type_sections_are_side_by_side() -> None:
    sections = _marginal_panel_sections(
        ["charge A", "charge B", "define VSA", "define VSD"]
    )

    figure, panel_axes = _marginal_figure(sections)
    figure.canvas.draw()
    charge = panel_axes[0][0].get_position()
    first_define = panel_axes[1][0].get_position()
    second_define = panel_axes[2][0].get_position()

    assert charge.x1 < first_define.x0
    assert first_define.x0 == pytest.approx(second_define.x0)
    assert first_define.y0 > second_define.y0
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_plot_marginals_uses_define_metadata(monkeypatch) -> None:
    specs = Specs(
        {
            "bounds": {
                "define VSA": [60.0, 150.0],
                "define VSD": [0.0, 0.075],
            },
            "charge_constraints": [],
        }
    )
    values = np.column_stack([
        np.linspace(70.0, 140.0, 40),
        np.linspace(0.005, 0.07, 40) ** 1.1,
    ])
    results = PosteriorResults(
        values.reshape(10, 4, 2),
        sample_labels=["define VSA", "define VSD"],
        specs=specs,
    )
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    import bff.plotting as plotting

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    plot_marginals(
        results,
        specs,
        plot_metadata={
            "define VSA": {
                "xlabel": "O-VS",
                "ylabel": "angle [degree]",
            },
            "define VSD": {
                "xlabel": "C-O-VS",
                "ylabel": "distance [nm]",
            },
        },
    )

    figure = plotting.plt.gcf()
    assert len(figure.axes) == 2
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "angle [degree]",
        "distance [nm]",
    ]
    assert [axis.get_xlabel() for axis in figure.axes] == ["O-VS", "C-O-VS"]
    assert all(not axis.get_xticklabels() for axis in figure.axes)
    assert all(len(axis.texts) == 1 for axis in figure.axes)
    plotting.plt.close(figure)


def test_marginal_legend_does_not_overlap_multiple_panels(monkeypatch) -> None:
    specs = Specs(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "define VSA": [60.0, 150.0],
                "define VSD": [0.0, 0.075],
            },
            "charge_constraints": [],
        }
    )
    values = np.column_stack([
        np.linspace(-0.8, 0.8, 80),
        np.linspace(70.0, 140.0, 80),
        np.linspace(0.005, 0.07, 80),
    ])
    results = PosteriorResults(
        values.reshape(20, 4, 3),
        sample_labels=["charge A", "define VSA", "define VSD"],
        specs=specs,
    )
    results.prepare_samples(discard=0, thin=1, strip_outliers=False)

    import bff.plotting as plotting

    monkeypatch.setattr(plotting.plt, "show", lambda: None)
    plot_marginals(results, specs)

    figure = plotting.plt.gcf()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend = figure.findobj(Legend)[0]
    legend_box = legend.get_window_extent(renderer)
    axes_top = max(axis.get_window_extent(renderer).y1 for axis in figure.axes)
    assert axes_top < legend_box.y0

    charge_axis, angle_axis, distance_axis = figure.axes
    charge_right = charge_axis.get_window_extent(renderer).x1
    define_label_boxes = [
        axis.yaxis.label.get_window_extent(renderer)
        for axis in (angle_axis, distance_axis)
    ]
    assert charge_right < min(box.x0 for box in define_label_boxes)
    label_centers = [0.5 * (box.x0 + box.x1) for box in define_label_boxes]
    assert label_centers[0] == pytest.approx(label_centers[1], abs=1.0)
    plotting.plt.close(figure)
