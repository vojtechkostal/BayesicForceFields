from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.special import softmax
from scipy.stats import gaussian_kde

from .bayes.results import PosteriorResults
from .domain.specs import Specs

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, torch.Tensor]


def _wrap_label(text: str, max_per_line: int = 3) -> str:
    words = text.split()
    return "\n".join(
        " ".join(words[i:i + max_per_line])
        for i in range(0, len(words), max_per_line)
    )


def _coerce_specs(specs: Specs | PathLike) -> Specs:
    return specs if isinstance(specs, Specs) else Specs(specs)


def _coerce_samples(
    samples: PosteriorResults | ArrayLike,
) -> np.ndarray:
    if isinstance(samples, PosteriorResults):
        return np.asarray(samples.prepared_samples, dtype=float)
    if isinstance(samples, torch.Tensor):
        return samples.detach().cpu().numpy()
    return np.asarray(samples, dtype=float)


def _parameter_labels(
    names: Sequence[str],
    labels: Optional[Sequence[str] | Mapping[str, str]] = None,
) -> list[str]:
    if labels is None:
        return list(names)
    if isinstance(labels, Mapping):
        return [labels.get(name, name) for name in names]
    if len(labels) != len(names):
        raise ValueError(
            "parameter_labels must match the number of plotted parameters.")
    return list(labels)


def _expand_short_labels(
    labels: Sequence[str],
    full_labels: Sequence[str],
) -> list[str]:
    """Expand shortened labels like ``C1`` back to ``charge C1`` when possible."""
    if len(labels) != len(full_labels):
        return list(labels)

    lookup: dict[str, str] = {}
    for full_label in full_labels:
        if full_label.startswith("$"):
            lookup.setdefault(full_label, full_label)
            continue
        lookup.setdefault(full_label, full_label)
        lookup.setdefault(full_label.split(maxsplit=1)[-1], full_label)

    expanded: list[str] = []
    changed = False
    for label in labels:
        replacement = lookup.get(label, label)
        expanded.append(replacement)
        changed |= replacement != label

    return expanded if changed else list(labels)


def _axis_labels(kind: str) -> tuple[str, str]:
    if kind == "charge":
        return "Atom", "Charge [e]"
    if kind == "sigma":
        return "Atom type", "$\\sigma$ [nm]"
    return kind.capitalize(), kind.capitalize()


def _marginal_panel_sections(
    param_names: Sequence[str],
    max_parameters_per_row: int = 5,
) -> list[tuple[str, list[list[list[int]]]]]:
    """Build type sections, wrapping groups and gridding independent defines."""
    if max_parameters_per_row < 1:
        raise ValueError("max_parameters_per_row must be positive.")
    param_groups: dict[str, list[int]] = {}
    for idx, name in enumerate(param_names):
        param_groups.setdefault(name.split()[0], []).append(idx)

    sections: list[tuple[str, list[list[list[int]]]]] = []
    for kind, indices in param_groups.items():
        if kind == "define":
            n_columns = (
                1
                if len(indices) <= 2
                else min(max_parameters_per_row, int(np.ceil(np.sqrt(len(indices)))))
            )
            n_rows = int(np.ceil(len(indices) / n_columns))
            rows = []
            for row_indices in np.array_split(indices, n_rows):
                rows.append([[int(idx)] for idx in row_indices])
            sections.append((kind, rows))
            continue

        n_rows = int(np.ceil(len(indices) / max_parameters_per_row))
        rows = [
            [[int(idx) for idx in row_indices]]
            for row_indices in np.array_split(indices, n_rows)
        ]
        sections.append((kind, rows))
    return sections


def _marginal_figure(
    panel_sections: Sequence[
        tuple[str, Sequence[Sequence[Sequence[int]]]]
    ],
) -> tuple[Any, list[tuple[Any, str, Sequence[int]]]]:
    panel_gap = 0.9
    section_gap = 0.4
    section_widths = [
        max(
            sum(
                3.4 if kind == "define" else max(3.6, 1.45 * len(indices))
                for indices in row
            )
            + panel_gap * max(0, len(row) - 1)
            for row in rows
        )
        for kind, rows in panel_sections
    ]
    max_section_rows = max(len(rows) for _, rows in panel_sections)
    fig = plt.figure(
        figsize=(
            sum(section_widths) + section_gap * (len(panel_sections) - 1),
            2 * max_section_rows,
        )
    )
    outer = fig.add_gridspec(
        1,
        len(panel_sections),
        width_ratios=section_widths,
        wspace=0.1,
    )
    axes: list[tuple[Any, str, Sequence[int]]] = []
    for section_idx, ((kind, rows), section_width) in enumerate(
        zip(panel_sections, section_widths)
    ):
        section_grid = outer[0, section_idx].subgridspec(
            len(rows), 1, hspace=0.4
        )
        for row_idx, row in enumerate(rows):
            widths = [
                3.4 if kind == "define" else max(3.6, 1.45 * len(indices))
                for indices in row
            ]
            spare_width = max(0.0, section_width - sum(widths))
            row_grid = section_grid[row_idx, 0].subgridspec(
                1,
                len(row) + 2,
                width_ratios=[
                    max(spare_width / 2, 1e-6),
                    *widths,
                    max(spare_width / 2, 1e-6),
                ],
                wspace=0.4,
            )
            for panel_idx, indices in enumerate(row):
                axes.append(
                    (fig.add_subplot(row_grid[0, panel_idx + 1]), kind, indices)
                )
    return fig, axes


def _align_marginal_ylabels(
    fig: Any,
    panel_axes: Sequence[tuple[Any, str, Sequence[int]]],
) -> None:
    """Align y-label centers for panels occupying the same visual column."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    columns: dict[float, list[Any]] = {}
    for ax, _, _ in panel_axes:
        columns.setdefault(round(ax.get_position().x0, 3), []).append(ax)

    inverse = fig.transFigure.inverted()
    for axes in columns.values():
        if len(axes) < 2:
            continue
        label_centers = [
            inverse.transform(ax.yaxis.label.get_window_extent(renderer).get_points())[
                :, 0
            ].mean()
            for ax in axes
        ]
        label_x = min(label_centers)
        for ax in axes:
            position = ax.get_position()
            ax.yaxis.set_label_coords(
                label_x,
                0.5 * (position.y0 + position.y1),
                transform=fig.transFigure,
            )


def _place_top_legends(
    fig: Any,
    legend_groups: Sequence[tuple[Sequence[Any], int]],
    *,
    fontsize: float = 15,
) -> list[Any]:
    """Stack figure legends above the panels and reserve their rendered space."""
    width, original_height = fig.get_size_inches()
    edge_padding = 0.08
    legend_gap = 0.08
    panel_gap = 0.12
    legends = []
    legend_heights = []

    for handles, requested_columns in legend_groups:
        columns = min(max(1, requested_columns), len(handles))
        while True:
            legend = fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=columns,
                frameon=False,
                fontsize=fontsize,
            )
            fig.canvas.draw()
            bbox = legend.get_window_extent(fig.canvas.get_renderer())
            if bbox.width <= 0.96 * fig.bbox.width or columns == 1:
                break
            legend.remove()
            columns -= 1
        legends.append(legend)
        legend_heights.append(bbox.height / fig.dpi)

    reserved_height = (
        edge_padding
        + sum(legend_heights)
        + legend_gap * max(0, len(legends) - 1)
        + panel_gap
    )
    new_height = original_height + reserved_height
    original_bottom = fig.subplotpars.bottom * original_height
    fig.set_size_inches(width, new_height, forward=True)

    y = 1.0 - edge_padding / new_height
    for legend, legend_height in zip(legends, legend_heights):
        legend.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
        y -= (legend_height + legend_gap) / new_height
    fig.subplots_adjust(
        bottom=original_bottom / new_height,
        top=1.0 - reserved_height / new_height,
    )
    return legends


def _plot_metadata(
    plot_metadata: Optional[Mapping[str, Mapping[str, str]]],
) -> dict[str, dict[str, str]]:
    return {
        name: dict(metadata)
        for name, metadata in (plot_metadata or {}).items()
    }


def _marginal_tick_labels(
    param_names: Sequence[str],
    parameter_labels: Optional[Sequence[str] | Mapping[str, str]],
    plot_metadata: Mapping[str, Mapping[str, str]],
) -> list[str]:
    unknown_metadata = set(plot_metadata) - set(param_names)
    if unknown_metadata:
        raise ValueError(
            "plot_metadata contains unknown parameter(s): "
            + ", ".join(sorted(unknown_metadata))
        )
    if parameter_labels is not None:
        return _parameter_labels(param_names, parameter_labels)
    return [
        plot_metadata.get(name, {}).get(
            "xlabel",
            name if name.startswith("$") else name.split(maxsplit=1)[-1],
        )
        for name in param_names
    ]


def _marginal_axis_labels(
    kind: str,
    indices: Sequence[int],
    param_names: Sequence[str],
    plot_metadata: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    xlabel, ylabel = _axis_labels(kind)
    if kind == "define" and len(indices) == 1:
        name = param_names[indices[0]]
        metadata = plot_metadata.get(name, {})
        xlabel = metadata.get("xlabel", name.split(maxsplit=1)[-1])
        ylabel = metadata.get("ylabel", "Value")
    return xlabel, ylabel


def _format_range_value(
    value: float,
    lower: float,
    upper: float,
    kind: str,
) -> str:
    """Format a marginal summary compactly for its parameter kind."""
    if kind != "charge":
        return f"{value:.3g}"

    span = abs(float(upper) - float(lower))
    if not np.isfinite(span) or span == 0:
        decimals = 3
    else:
        decimals = max(0, min(3, 3 - int(np.floor(np.log10(span)))))
    if round(value, decimals) == 0:
        value = 0.0
    return f"{value:.{decimals}f}"


def _marginal_mode(
    kde: gaussian_kde,
    values: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Estimate a bounded one-dimensional KDE mode on an adaptive grid."""
    grid_lower = max(float(lower), float(np.min(values)))
    grid_upper = min(float(upper), float(np.max(values)))
    if grid_lower >= grid_upper:
        return grid_lower
    grid = np.linspace(grid_lower, grid_upper, 512)
    return float(grid[np.argmax(kde(grid))])


def plot_marginals(
    results: PosteriorResults,
    specs: Specs | PathLike,
    *,
    parameter_labels: Optional[Sequence[str] | Mapping[str, str]] = None,
    plot_metadata: Optional[Mapping[str, Mapping[str, str]]] = None,
    max_samples: Optional[int] = None,
    color_prior: str = "gray",
    color_posterior: str = "tab:red",
    fn_out: Optional[PathLike] = None,
) -> None:
    specs = _coerce_specs(specs)
    posterior = (
        results.prepared_samples
        if results.include_implicit_charge
        else specs.with_implicit_charges(results.prepared_samples)
    )
    if max_samples is not None and max_samples != -1:
        if max_samples < 1:
            raise ValueError("max_samples must be positive, -1, or None.")
        if len(posterior) > max_samples:
            indices = np.linspace(0, len(posterior) - 1, max_samples, dtype=int)
            posterior = posterior[indices]
    param_names = specs.bounds.names.tolist()
    plot_metadata = _plot_metadata(plot_metadata)
    tick_labels = _marginal_tick_labels(
        param_names, parameter_labels, plot_metadata
    )
    panel_sections = _marginal_panel_sections(param_names)
    fig, panel_axes = _marginal_figure(panel_sections)

    explicit_names = specs.explicit_bounds.names.tolist()
    prior_index = {name: i for i, name in enumerate(explicit_names)}
    show_prior = results.priors is not None
    legend_used = {"prior": False, "posterior": False, "bounds": False}
    for ax, kind, indices in panel_axes:
        bounds_block = np.asarray(
            [specs.bounds.by_name[param_names[i]] for i in indices],
            dtype=float,
        )
        y_min = bounds_block[:, 0].min()
        y_max = bounds_block[:, 1].max()
        y_pad = max(0.05, 0.18 * (y_max - y_min))
        posterior_peaks: list[float] = []
        prior_peaks: list[float] = []
        curves: dict[
            int, tuple[np.ndarray, np.ndarray, Optional[np.ndarray], float]
        ] = {}

        for idx in indices:
            name = param_names[idx]
            lower, upper = specs.bounds.by_name[name]
            y = np.linspace(lower - y_pad, upper + y_pad, 400)

            prior_density = None
            if show_prior and name in prior_index:
                prior = results.priors.distributions[prior_index[name]]
                prior_density = (
                    prior.log_prob(torch.as_tensor(y, dtype=torch.float32))
                    .exp()
                    .detach()
                    .cpu()
                    .numpy()
                )
                prior_peaks.append(float(np.max(prior_density)))

            posterior_values = posterior[:, idx]
            posterior_kde = gaussian_kde(posterior_values)
            posterior_density = posterior_kde(y)
            posterior_peaks.append(float(np.max(posterior_density)))
            posterior_mode = _marginal_mode(
                posterior_kde,
                posterior_values,
                lower,
                upper,
            )
            curves[idx] = (y, posterior_density, prior_density, posterior_mode)

        max_posterior_peak = max(posterior_peaks, default=1.0)
        max_prior_peak = max(prior_peaks, default=max_posterior_peak)
        posterior_width = 1.2
        prior_width = 0.7
        posterior_scale = posterior_width / max(max_posterior_peak, 1e-12)
        prior_scale = prior_width / max(max_prior_peak, 1e-12)

        for xpos, idx in enumerate(indices):
            name = param_names[idx]
            lower, upper = specs.bounds.by_name[name]
            y, posterior_density, prior_density, posterior_mode = curves[idx]

            if prior_density is not None:
                ax.fill_betweenx(
                    y,
                    xpos - prior_scale * prior_density,
                    xpos,
                    color=color_prior,
                    lw=0,
                    zorder=1,
                    label="prior" if not legend_used["prior"] else None,
                )
                legend_used["prior"] = True

            ax.fill_betweenx(
                y,
                xpos,
                xpos + posterior_scale * posterior_density,
                color=color_posterior,
                lw=0,
                zorder=2,
                label="posterior" if not legend_used["posterior"] else None,
            )
            legend_used["posterior"] = True

            center = 0.5 * (lower + upper)
            yerr = np.array([[center - lower], [upper - center]])
            ax.errorbar(
                [xpos],
                [center],
                yerr=yerr,
                lw=2.5,
                ls="",
                capsize=4,
                capthick=2.5,
                markeredgewidth=2.5,
                color="k",
                zorder=3,
                label="bounds" if not legend_used["bounds"] else None,
            )
            legend_used["bounds"] = True

            ax.text(
                xpos,
                lower - 0.25 * y_pad,
                _format_range_value(posterior_mode, lower, upper, kind),
                color="tab:red",
                fontweight="bold",
                fontsize=12,
                ha="center",
                va="top",
                zorder=4,
            )

        ax.set_xlim(-prior_width - 0.25, len(indices) - 1 + posterior_width + 0.25)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if kind == "define" and len(indices) == 1:
            ax.set_xticks([])
        else:
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels(
                [_wrap_label(tick_labels[i]) for i in indices],
                rotation=30,
                ha="center",
                fontsize=15,
            )
        xlabel, ylabel = _marginal_axis_labels(
            kind, indices, param_names, plot_metadata
        )
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(direction="in", width=1.2, labelsize=15)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    if panel_axes:
        handles_by_label = {}
        for ax, _, _ in panel_axes:
            handles, labels = ax.get_legend_handles_labels()
            handles_by_label.update(zip(labels, handles))
        labels = [
            label
            for label in ("prior", "posterior", "bounds")
            if label in handles_by_label
        ]
        handles = [handles_by_label[label] for label in labels]
        if handles:
            _place_top_legends(fig, [(handles, len(labels))])
    _align_marginal_ylabels(fig, panel_axes)
    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _local_qoi_responsibilities(
    parameter: np.ndarray,
    grid: np.ndarray,
    standardized_log_likelihood: np.ndarray,
    temperature: float,
) -> np.ndarray:
    bandwidth = gaussian_kde(parameter).factor * np.std(parameter)
    bandwidth = max(bandwidth, np.ptp(parameter) / 100.0, 1e-8)
    distance = (parameter[:, None] - grid[None, :]) / bandwidth
    kernel = np.exp(-0.5 * distance**2)
    local_scores = kernel.T @ standardized_log_likelihood
    local_scores /= np.maximum(kernel.sum(axis=0)[:, None], 1e-12)

    posterior_density = gaussian_kde(parameter)(grid)
    baseline = np.average(local_scores, axis=0, weights=posterior_density)
    return softmax((local_scores - baseline) / temperature, axis=1)


def plot_qoi_marginals(
    results: PosteriorResults,
    specs: Specs | PathLike,
    log_likelihood_by_qoi: Mapping[str, ArrayLike],
    *,
    parameter_labels: Optional[Sequence[str] | Mapping[str, str]] = None,
    plot_metadata: Optional[Mapping[str, Mapping[str, str]]] = None,
    temperature: float = 0.7,
    colors: Optional[Mapping[str, Any]] = None,
    color_prior: str = "gray",
    sample_indices: Optional[ArrayLike] = None,
    fn_out: Optional[PathLike] = None,
) -> None:
    """Plot contrastive QoI attribution within posterior marginals."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite.")
    if not log_likelihood_by_qoi:
        raise ValueError("log_likelihood_by_qoi must not be empty.")

    specs = _coerce_specs(specs)
    posterior = (
        results.prepared_samples
        if results.include_implicit_charge
        else specs.with_implicit_charges(results.prepared_samples)
    )
    if sample_indices is not None:
        sample_indices = np.asarray(_coerce_samples(sample_indices), dtype=int)
        if sample_indices.ndim != 1:
            raise ValueError("sample_indices must be one-dimensional.")
        if np.any(sample_indices < 0) or np.any(sample_indices >= len(posterior)):
            raise ValueError("sample_indices contains an out-of-range index.")
        posterior = posterior[sample_indices]
    param_names = specs.bounds.names.tolist()
    plot_metadata = _plot_metadata(plot_metadata)
    tick_labels = _marginal_tick_labels(
        param_names, parameter_labels, plot_metadata
    )

    qoi_names = list(log_likelihood_by_qoi)
    log_likelihood = np.column_stack([
        _coerce_samples(log_likelihood_by_qoi[qoi]).reshape(-1)
        for qoi in qoi_names
    ])
    if log_likelihood.shape[0] != len(posterior):
        raise ValueError(
            "QoI log likelihoods must match the prepared posterior sample count."
        )
    if not np.all(np.isfinite(log_likelihood)):
        raise ValueError("QoI log likelihoods must contain only finite values.")

    centers = np.median(log_likelihood, axis=0)
    scales = np.subtract(
        *np.quantile(log_likelihood, [0.75, 0.25], axis=0)
    )
    fallback = np.std(log_likelihood, axis=0)
    scales = np.where(scales > 1e-12, scales, fallback)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (log_likelihood - centers) / scales

    default_colors = plt.get_cmap("tab10").colors
    qoi_colors = {
        qoi: (
            colors[qoi]
            if colors is not None and qoi in colors
            else default_colors[i % len(default_colors)]
        )
        for i, qoi in enumerate(qoi_names)
    }

    panel_sections = _marginal_panel_sections(param_names)

    explicit_names = specs.explicit_bounds.names.tolist()
    prior_index = {name: i for i, name in enumerate(explicit_names)}
    show_prior = results.priors is not None

    fig, panel_axes = _marginal_figure(panel_sections)
    profile_width = 1.2
    prior_width = 0.7

    for ax, kind, indices in panel_axes:
        bounds_block = np.asarray(
            [specs.bounds.by_name[param_names[i]] for i in indices],
            dtype=float,
        )
        y_min = bounds_block[:, 0].min()
        y_max = bounds_block[:, 1].max()
        y_pad = max(0.05, 0.18 * (y_max - y_min))

        for xpos, idx in enumerate(indices):
            name = param_names[idx]
            lower, upper = specs.bounds.by_name[name]
            grid = np.linspace(lower, upper, 400)
            values = posterior[:, idx]
            density = gaussian_kde(values)(grid)
            density /= max(float(density.max()), 1e-12)
            responsibilities = _local_qoi_responsibilities(
                values,
                grid,
                standardized,
                temperature,
            )

            cumulative = np.zeros_like(grid)
            for qoi_idx, qoi in enumerate(qoi_names):
                next_cumulative = cumulative + responsibilities[:, qoi_idx]
                ax.fill_betweenx(
                    grid,
                    xpos + profile_width * cumulative * density,
                    xpos + profile_width * next_cumulative * density,
                    color=qoi_colors[qoi],
                    lw=0,
                    zorder=2,
                )
                cumulative = next_cumulative

            if show_prior and name in prior_index:
                prior = results.priors.distributions[prior_index[name]]
                prior_density = (
                    prior.log_prob(torch.as_tensor(grid, dtype=torch.float32))
                    .exp()
                    .detach()
                    .cpu()
                    .numpy()
                )
                prior_density /= max(float(prior_density.max()), 1e-12)
                ax.fill_betweenx(
                    grid,
                    xpos - prior_width * prior_density,
                    xpos,
                    color=color_prior,
                    lw=0,
                    zorder=1,
                )

            ax.plot(
                xpos + profile_width * density,
                grid,
                color="k",
                lw=2.0,
                zorder=3,
            )
            center = 0.5 * (lower + upper)
            ax.errorbar(
                xpos,
                center,
                yerr=[[center - lower], [upper - center]],
                lw=2.5,
                ls="",
                capsize=4,
                capthick=2.5,
                markeredgewidth=2.5,
                color="k",
                zorder=4,
            )

        ax.set_xlim(
            -prior_width - 0.25,
            len(indices) - 1 + profile_width + 0.25,
        )
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        if kind == "define" and len(indices) == 1:
            ax.set_xticks([])
        else:
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels(
                [_wrap_label(tick_labels[i]) for i in indices],
                rotation=30,
                ha="center",
                fontsize=15,
            )
        xlabel, ylabel = _marginal_axis_labels(
            kind, indices, param_names, plot_metadata
        )
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(direction="in", width=1.2, labelsize=15)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    summary_handles = [
        Patch(facecolor=color_prior, label="prior"),
        plt.Line2D([0], [0], color="k", lw=2.0, label="posterior"),
        panel_axes[0][0].errorbar(
            [np.nan],
            [np.nan],
            yerr=[[0.5], [0.5]],
            color="k",
            lw=2.5,
            ls="",
            capsize=4,
            capthick=2.5,
            label="bounds",
        ),
    ]
    if not show_prior:
        summary_handles = summary_handles[1:]
    qoi_handles = [
        Patch(facecolor=qoi_colors[qoi], label=qoi)
        for qoi in qoi_names
    ]
    _place_top_legends(
        fig,
        [
            (summary_handles, len(summary_handles)),
            (qoi_handles, len(qoi_handles)),
        ],
    )
    _align_marginal_ylabels(fig, panel_axes)

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_corner(
    samples: PosteriorResults | ArrayLike,
    labels: Optional[Sequence[str]] = None,
    *,
    quantiles: Sequence[float] = (0.16, 0.5, 0.84),
    figsize: float = 1.5,
    cmap: Any = "Reds",
    levels: int = 5,
    scatter_alpha: float = 0.15,
    max_samples: Optional[int] = None,
    fn_out: Optional[PathLike] = None,
) -> None:
    sample_source = samples
    samples = _coerce_samples(samples)
    result_labels = None
    if (
        isinstance(sample_source, PosteriorResults)
        and sample_source.specs is not None
        and not sample_source.include_implicit_charge
    ):
        samples = sample_source.specs.with_implicit_charges(samples)
        result_labels = sample_source._labels_with_implicit_charges()
    elif isinstance(sample_source, PosteriorResults):
        result_labels = list(sample_source.labels)

    if samples.ndim != 2:
        raise ValueError("plot_corner expects samples with shape (n_samples, n_dim).")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        if len(samples) > max_samples:
            indices = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
            samples = samples[indices]

    n_dim = samples.shape[1]
    if labels is None:
        if result_labels is not None:
            labels = result_labels
        else:
            labels = [f"theta_{i}" for i in range(n_dim)]
    elif len(labels) != n_dim:
        raise ValueError("labels must match the posterior sample dimension.")
    elif result_labels is not None:
        labels = _expand_short_labels(labels, result_labels)

    labels = [_wrap_label(label) for label in labels]
    base_cmap = plt.get_cmap(cmap)
    colors = base_cmap(np.linspace(0, 1, max(levels, 2)))
    colors[0] = np.array([1.0, 1.0, 1.0, 0.0])
    contour_cmap = ListedColormap(colors)
    fig, axes = plt.subplots(
        n_dim,
        n_dim,
        figsize=(figsize * n_dim, figsize * n_dim),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    axes = np.asarray(axes, dtype=object).reshape(n_dim, n_dim)

    limits = [(samples[:, i].min(), samples[:, i].max()) for i in range(n_dim)]

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue

            if i == j:
                x = np.linspace(*limits[i], 400)
                kde = gaussian_kde(samples[:, i])
                density = kde(x)
                ax.plot(x, density, color="k", lw=2.5)
                ax.fill_between(x, 0, density, color="0.75", alpha=0.7)
                if quantiles:
                    q_values = np.quantile(samples[:, i], quantiles)
                    for q in q_values:
                        ax.axvline(q, color="k", ls="--", lw=1.3)
                    if len(q_values) == 3:
                        median = q_values[1]
                        lower = median - q_values[0]
                        upper = q_values[2] - median
                        ax.set_title(
                            (
                                f"{labels[i]}\n"
                                f"{median:.3f}\n"
                                f"(+{upper:.3f} / -{lower:.3f})"
                            ),
                            fontsize=15,
                        )
                ax.set_xlim(*limits[i])
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                x = samples[:, j]
                y = samples[:, i]
                ax.scatter(
                    x[::10],
                    y[::10],
                    s=5,
                    lw=0,
                    alpha=scatter_alpha,
                    color="k",
                    rasterized=True,
                )
                try:
                    kde = gaussian_kde(np.vstack([x, y]))
                    xi, yi = np.mgrid[
                        limits[j][0]:limits[j][1]:100j,
                        limits[i][0]:limits[i][1]:100j,
                    ]
                    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
                    contour_levels = np.linspace(zi.min(), zi.max(), levels)
                    ax.contourf(
                        xi, yi, zi, levels=contour_levels, cmap=contour_cmap
                    )
                    ax.contour(
                        xi, yi, zi, levels=contour_levels, colors="k", linewidths=0.8
                    )
                except np.linalg.LinAlgError:
                    pass
                ax.set_xlim(*limits[j])
                ax.set_ylim(*limits[i])

            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(direction="in", top=True, right=True, labelsize=15)

            if i == n_dim - 1:
                ax.set_xlabel(labels[j], fontsize=15)
            else:
                ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=15)
            elif i != j:
                ax.set_yticklabels([])

    fig.align_labels()

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
