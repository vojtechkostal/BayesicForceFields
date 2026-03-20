from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from .structures import InferenceResults, Specs


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
    samples: InferenceResults | ArrayLike,
) -> np.ndarray:
    if isinstance(samples, InferenceResults):
        return np.asarray(samples.samples, dtype=float)
    if isinstance(samples, torch.Tensor):
        return samples.detach().cpu().numpy()
    return np.asarray(samples, dtype=float)


def _parameter_labels(
    names: Sequence[str],
    labels: Optional[Sequence[str] | Mapping[str, str]] = None,
) -> list[str]:
    if labels is None:
        return [name.split(maxsplit=1)[-1] for name in names]
    if isinstance(labels, Mapping):
        return [labels.get(name, name) for name in names]
    if len(labels) != len(names):
        raise ValueError(
            "parameter_labels must match the number of plotted parameters.")
    return list(labels)


def _axis_labels(kind: str) -> tuple[str, str]:
    if kind == "charge":
        return "Atom", "Charge [e]"
    if kind == "sigma":
        return "Atom type", "$\\sigma$ [nm]"
    return kind.capitalize(), kind.capitalize()


def _posterior_with_implicit_charge(
    results: InferenceResults,
    specs: Specs,
) -> np.ndarray:
    samples = np.asarray(results.samples, dtype=float)
    n_explicit = specs.bounds.without(specs.implicit_param).n_params
    explicit_charge = samples[:, :n_explicit] @ specs.constraint_matrix
    implicit_charge = (
        specs.constraint_charge - explicit_charge
    ) / len(specs.implicit_atoms)
    return np.insert(samples, specs.implicit_param_index, implicit_charge, axis=1)


def plot_marginals(
    results: InferenceResults,
    specs: Specs | PathLike,
    *,
    parameter_labels: Optional[Sequence[str] | Mapping[str, str]] = None,
    color_prior: str = "gray",
    color_posterior: str = "tab:red",
    fn_out: Optional[PathLike] = None,
) -> None:
    specs = _coerce_specs(specs)
    posterior = _posterior_with_implicit_charge(results, specs)
    param_names = specs.bounds.names.tolist()
    tick_labels = _parameter_labels(param_names, parameter_labels)

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

    explicit_names = specs.bounds.without(specs.implicit_param).names.tolist()
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
            ha="right",
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


def plot_corner(
    samples: InferenceResults | ArrayLike,
    labels: Optional[Sequence[str]] = None,
    *,
    quantiles: Sequence[float] = (0.16, 0.5, 0.84),
    figsize: float = 2.6,
    cmap: Any = "Reds",
    levels: int = 5,
    scatter_alpha: float = 0.15,
    fn_out: Optional[PathLike] = None,
) -> None:
    samples = _coerce_samples(samples)
    if samples.ndim != 2:
        raise ValueError("plot_corner expects samples with shape (n_samples, n_dim).")

    n_dim = samples.shape[1]
    if labels is None:
        labels = [f"theta_{i}" for i in range(n_dim)]
    elif len(labels) != n_dim:
        raise ValueError("labels must match the posterior sample dimension.")

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
                x = samples[::100, j]
                y = samples[::100, i]
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

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
