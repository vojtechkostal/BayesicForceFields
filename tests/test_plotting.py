import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.legend import Legend

import bff.plotting as plotting
from bff.bayes.priors import Priors
from bff.bayes.results import Results
from bff.domain.specs import Specs
from bff.plotting import (
    _marginal_figure,
    _marginal_panel_sections,
    plot_corner,
    plot_marginals,
    plot_qoi_marginals,
)

CHARGE_PAIR = {
    "bounds": {"charge A": [-1.0, 1.0], "charge B": [-1.0, 1.0]},
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


def make_results(
    specs: dict,
    columns: np.ndarray,
    *,
    prior: bool = False,
    qoi_log_likelihood: dict | None = None,
    nuisances: tuple[str, ...] = (),
) -> Results:
    """Results whose chain columns are given (one row per saved state, four
    walkers); the log posterior peaks at the center of the samples."""
    specs = Specs(specs)
    columns = np.asarray(columns, dtype=float)
    chain = columns.reshape(-1, 4, columns.shape[1])
    spread = np.maximum(chain.std(axis=(0, 1)), 1e-12)
    log_prob = -np.sum(((chain - chain.mean(axis=(0, 1))) / spread) ** 2, axis=-1)
    return Results(
        chain,
        log_prob,
        specs,
        prior=(
            Priors.from_bounds(
                specs.explicit_bounds,
                names=specs.explicit_names,
                nuisance_names=[f"log noise {q}" for q in nuisances],
            )
            if prior
            else None
        ),
        nuisances=nuisances,
        qoi_index=None if qoi_log_likelihood is None else np.arange(len(columns)),
        qoi_log_likelihood=qoi_log_likelihood,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_plot_corner_includes_implicit_charges_and_returns_the_figure() -> None:
    results = make_results(CHARGE_PAIR, np.linspace(-0.4, 0.4, 40).reshape(-1, 1))

    figure = plot_corner(results)

    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title().startswith("charge A")
    assert figure.axes[2].get_ylabel() == "charge B"
    assert results.plot_corner().axes[2].get_ylabel() == "charge B"


def test_plot_corner_selects_columns_and_includes_noise() -> None:
    rng = np.random.default_rng(0)
    results = make_results(
        CHARGE_PAIR,
        np.column_stack([np.linspace(-0.4, 0.4, 40), rng.normal(-3, 0.1, 40)]),
        nuisances=("rdf",),
    )

    assert len(plot_corner(results).axes) == 9  # charge A, charge B, noise rdf
    only = plot_corner(results, names=["noise rdf", "charge A"])
    assert [ax.get_title().split("\n")[0] for ax in only.axes if ax.get_title()] == [
        "noise rdf",
        "charge A",
    ]
    with pytest.raises(ValueError, match="Unknown parameter"):
        plot_corner(results, names=["charge Z"])


def test_plot_corner_limits_samples_used_by_kde(monkeypatch) -> None:
    results = make_results(
        {
            "bounds": {"sigma A": [-1.0, 1.0], "sigma B": [0.0, 3.0]},
            "charge_constraints": [],
        },
        np.column_stack(
            [np.linspace(-1.0, 1.0, 100), np.linspace(-0.5, 1.5, 100) ** 2]
        ),
    )
    counts = []
    real_kde = plotting.gaussian_kde

    def record_kde(values, *args, **kwargs):
        counts.append(np.asarray(values).shape[-1])
        return real_kde(values, *args, **kwargs)

    monkeypatch.setattr(plotting, "gaussian_kde", record_kde)

    plot_corner(results, max_samples=20)

    assert counts == [20, 20, 20]


def test_plot_qoi_marginals_stacks_qoi_profiles_without_annotations() -> None:
    values = np.linspace(-0.6, 0.6, 80)
    results = make_results(
        CHARGE_PAIR,
        values.reshape(-1, 1),
        prior=True,
        qoi_log_likelihood={"rdf": -(values**2), "density": -((values - 0.3) ** 2)},
    )

    figure = plot_qoi_marginals(results)

    assert isinstance(figure, Figure)
    ax = figure.axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["A", "B"]
    assert len(ax.texts) == 0
    assert len(ax.collections) >= 4
    outline = next(line for line in ax.lines if len(line.get_ydata()) == 400)
    assert outline.get_ydata().min() == -1.0
    assert outline.get_ydata().max() == 1.0
    prior = next(c for c in ax.collections if c.get_zorder() == 1)
    posterior = [c for c in ax.collections if c.get_zorder() == 2]
    assert posterior
    assert all(item.get_zorder() > prior.get_zorder() for item in posterior)
    legends = figure.findobj(Legend)
    assert [[t.get_text() for t in legend.get_texts()] for legend in legends] == [
        ["prior", "posterior", "bounds"],
        ["rdf", "density"],
    ]


def test_plot_qoi_marginals_needs_the_stored_likelihoods() -> None:
    results = make_results(CHARGE_PAIR, np.linspace(-0.6, 0.6, 80).reshape(-1, 1))

    with pytest.raises(ValueError, match="no per-QoI log likelihoods"):
        plot_qoi_marginals(results)


def test_plot_marginals_annotates_mode_and_marks_the_map() -> None:
    values = np.concatenate([np.linspace(0.1, 0.3, 36), np.full(4, 1.8)])
    results = make_results(
        {"bounds": {"sigma A": [0.0, 2.0]}, "charge_constraints": []},
        values.reshape(-1, 1),
        prior=True,
    )

    figure = plot_marginals(results)

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
    (marker,) = [line for line in ax.lines if line.get_marker() == "D"]
    assert marker.get_ydata()[0] == pytest.approx(results.map["sigma A"])
    assert [t.get_text() for t in figure.findobj(Legend)[0].get_texts()] == [
        "prior",
        "posterior",
        "bounds",
        "MAP",
    ]


def test_plot_marginals_limits_samples_used_by_kde(monkeypatch) -> None:
    results = make_results(
        {"bounds": {"sigma A": [0.0, 2.0]}, "charge_constraints": []},
        np.linspace(0.1, 1.9, 100).reshape(-1, 1),
    )
    counts = []
    real_kde = plotting.gaussian_kde

    def record_kde(data, *args, **kwargs):
        counts.append(np.asarray(data).shape[-1])
        return real_kde(data, *args, **kwargs)

    monkeypatch.setattr(plotting, "gaussian_kde", record_kde)

    plot_marginals(results, max_samples=17)

    assert counts[0] == 17
    with pytest.raises(ValueError, match="max_samples"):
        plot_marginals(results, max_samples=0)


def test_plot_marginal_charge_mode_uses_at_most_three_decimals() -> None:
    results = make_results(
        {"bounds": {"charge A": [-0.2, 0.2]}, "charge_constraints": []},
        np.linspace(-0.1241, -0.1231, 40).reshape(-1, 1),
    )

    figure = plot_marginals(results)

    annotation = figure.axes[0].texts[0].get_text()
    assert float(annotation) == pytest.approx(-0.1236, abs=0.001)
    assert len(annotation.partition(".")[2]) <= 3


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


def test_plot_marginals_uses_define_metadata() -> None:
    results = make_results(
        {
            "bounds": {"define VSA": [60.0, 150.0], "define VSD": [0.0, 0.075]},
            "charge_constraints": [],
        },
        np.column_stack(
            [np.linspace(70.0, 140.0, 40), np.linspace(0.005, 0.07, 40) ** 1.1]
        ),
    )

    figure = plot_marginals(
        results,
        plot_metadata={
            "define VSA": {"xlabel": "O-VS", "ylabel": "angle [degree]"},
            "define VSD": {"xlabel": "C-O-VS", "ylabel": "distance [nm]"},
        },
    )

    assert len(figure.axes) == 2
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "angle [degree]",
        "distance [nm]",
    ]
    assert [axis.get_xlabel() for axis in figure.axes] == ["O-VS", "C-O-VS"]
    assert all(not axis.get_xticklabels() for axis in figure.axes)
    assert all(len(axis.texts) == 1 for axis in figure.axes)
    with pytest.raises(ValueError, match="unknown parameter"):
        plot_marginals(results, plot_metadata={"define XYZ": {"xlabel": "x"}})


def test_marginal_legend_does_not_overlap_multiple_panels() -> None:
    results = make_results(
        {
            "bounds": {
                "charge A": [-1.0, 1.0],
                "define VSA": [60.0, 150.0],
                "define VSD": [0.0, 0.075],
            },
            "charge_constraints": [],
        },
        np.column_stack(
            [
                np.linspace(-0.8, 0.8, 80),
                np.linspace(70.0, 140.0, 80),
                np.linspace(0.005, 0.07, 80),
            ]
        ),
    )

    figure = plot_marginals(results)

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
