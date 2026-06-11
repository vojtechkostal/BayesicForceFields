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


def _wrap_label(text: str, max_per_line: int = 4) -> str:
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


def plot_marginals(
    results: PosteriorResults,
    specs: Specs | PathLike,
    *,
    parameter_labels: Optional[Sequence[str] | Mapping[str, str]] = None,
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
    param_names = specs.bounds.names.tolist()
    tick_labels = _parameter_labels(param_names, parameter_labels)
    if parameter_labels is None:
        tick_labels = [
            label if label.startswith("$") else label.split(maxsplit=1)[-1]
            for label in tick_labels
        ]

    param_groups: dict[str, list[int]] = {}
    for idx, name in enumerate(param_names):
        kind = name.split()[0]
        param_groups.setdefault(kind, []).append(idx)

    fig, axes = plt.subplots(
        1,
        len(param_groups),
        figsize=(4 * len(param_groups), 3.2),
        gridspec_kw={"wspace": 0.3},
    )
    axes = np.atleast_1d(axes)

    explicit_names = specs.explicit_bounds.names.tolist()
    prior_index = {name: i for i, name in enumerate(explicit_names)}
    show_prior = results.priors is not None
    legend_used = {"prior": False, "posterior": False, "bounds": False}

    for ax, (kind, indices) in zip(axes, param_groups.items()):
        bounds_block = np.asarray(
            [specs.bounds.by_name[param_names[i]] for i in indices],
            dtype=float,
        )
        y_min = bounds_block[:, 0].min()
        y_max = bounds_block[:, 1].max()
        y_pad = max(0.05, 0.18 * (y_max - y_min))
        label_y = y_min - 0.80 * y_pad
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

            posterior_density = gaussian_kde(posterior[:, idx])(y)
            posterior_peaks.append(float(np.max(posterior_density)))
            mode = float(y[np.argmax(posterior_density)])
            curves[idx] = (y, posterior_density, prior_density, mode)

        max_posterior_peak = max(posterior_peaks, default=1.0)
        max_prior_peak = max(prior_peaks, default=max_posterior_peak)
        posterior_width = 1.2
        prior_width = 0.7
        posterior_scale = posterior_width / max(max_posterior_peak, 1e-12)
        prior_scale = prior_width / max(max_prior_peak, 1e-12)

        for xpos, idx in enumerate(indices):
            name = param_names[idx]
            lower, upper = specs.bounds.by_name[name]
            y, posterior_density, prior_density, mode = curves[idx]

            if prior_density is not None:
                ax.fill_betweenx(
                    y,
                    xpos - prior_scale * prior_density,
                    xpos,
                    color=color_prior,
                    lw=0,
                    label="prior" if not legend_used["prior"] else None,
                )
                legend_used["prior"] = True

            ax.fill_betweenx(
                y,
                xpos,
                xpos + posterior_scale * posterior_density,
                color=color_posterior,
                lw=0,
                label="posterior" if not legend_used["posterior"] else None,
            )
            legend_used["posterior"] = True

            center = 0.5 * (lower + upper)
            yerr = np.array([[center - lower], [upper - center]])
            ax.errorbar(
                [xpos],
                [center],
                yerr=yerr,
                lw=2,
                ls="",
                capsize=4,
                capthick=2,
                markeredgewidth=2,
                color="k",
                label="bounds" if not legend_used["bounds"] else None,
            )
            legend_used["bounds"] = True

            ax.text(
                xpos,
                label_y,
                f"{mode:.3f}",
                color=color_posterior,
                fontweight="bold",
                ha="center",
                va="top",
            )

        ax.set_xlim(-prior_width - 0.25, len(indices) - 1 + posterior_width + 0.25)
        ax.set_ylim(label_y - 0.6 * y_pad, y_max + y_pad)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(
            [_wrap_label(tick_labels[i]) for i in indices],
            rotation=30,
            ha="center",
        )
        xlabel, ylabel = _axis_labels(kind)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in")

    if len(axes):
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=3,
            frameon=False,
        )

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
    temperature: float = 0.7,
    colors: Optional[Mapping[str, Any]] = None,
    color_prior: str = "gray",
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
    param_names = specs.bounds.names.tolist()
    tick_labels = _parameter_labels(param_names, parameter_labels)
    if parameter_labels is None:
        tick_labels = [
            label if label.startswith("$") else label.split(maxsplit=1)[-1]
            for label in tick_labels
        ]

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

    param_groups: dict[str, list[int]] = {}
    for idx, name in enumerate(param_names):
        param_groups.setdefault(name.split()[0], []).append(idx)

    explicit_names = specs.explicit_bounds.names.tolist()
    prior_index = {name: i for i, name in enumerate(explicit_names)}
    show_prior = results.priors is not None

    fig, axes = plt.subplots(
        1,
        len(param_groups),
        figsize=(4 * len(param_groups), 3.2),
        gridspec_kw={"wspace": 0.3},
    )
    axes = np.atleast_1d(axes)
    profile_width = 1.2
    prior_width = 0.7

    for ax, (kind, indices) in zip(axes, param_groups.items()):
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
                )

            ax.plot(
                xpos + profile_width * density,
                grid,
                color="k",
                lw=1.5,
            )
            center = 0.5 * (lower + upper)
            ax.errorbar(
                xpos,
                center,
                yerr=[[center - lower], [upper - center]],
                lw=2,
                ls="",
                capsize=4,
                capthick=2,
                markeredgewidth=2,
                color="k",
            )

        ax.set_xlim(
            -prior_width - 0.25,
            len(indices) - 1 + profile_width + 0.25,
        )
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(
            [_wrap_label(tick_labels[i]) for i in indices],
            rotation=30,
            ha="center",
        )
        xlabel, ylabel = _axis_labels(kind)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in")

    summary_handles = [
        Patch(facecolor=color_prior, label="prior"),
        plt.Line2D([0], [0], color="k", lw=1.5, label="posterior"),
        axes[0].errorbar(
            [np.nan],
            [np.nan],
            yerr=[[0.5], [0.5]],
            color="k",
            lw=2,
            ls="",
            capsize=4,
            capthick=2,
            label="bounds",
        ),
    ]
    if not show_prior:
        summary_handles = summary_handles[1:]
    fig.legend(
        handles=summary_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(summary_handles),
        frameon=False,
    )

    qoi_handles = [
        Patch(facecolor=qoi_colors[qoi], label=qoi)
        for qoi in qoi_names
    ]
    fig.legend(
        handles=qoi_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=len(qoi_handles),
        frameon=False,
    )
    fig.subplots_adjust(top=0.76)

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
    figsize: float = 2.6,
    cmap: Any = "Reds",
    levels: int = 5,
    scatter_alpha: float = 0.15,
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
                            fontsize=12,
                        )
                ax.set_xlim(*limits[i])
                ax.set_yticks([])
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                x = samples[::10, j]
                y = samples[::10, i]
                ax.scatter(
                    x, y, s=5, lw=0, alpha=scatter_alpha, color="k", rasterized=True
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
            ax.tick_params(direction="in", top=True, right=True, labelsize=11)

            if i == n_dim - 1:
                ax.set_xlabel(labels[j], fontsize=12)
            else:
                ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=12)
            elif i != j:
                ax.set_yticklabels([])

    fig.align_labels()

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
