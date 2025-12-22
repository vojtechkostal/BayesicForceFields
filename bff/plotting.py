import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from scipy.stats import gaussian_kde

from .structures import InferenceResults


def plot_marginals(
    results: InferenceResults,
    color_prior: str = 'gray', color_posterior: str = 'tab:red',
    fn_out: str = None
) -> None:

    param_kinds = set([p.split()[0] for p in results.bounds_explicit.params])
    n_param_kinds = len(param_kinds)

    gridspecs = {'wspace': 0.25}
    fig, axs = plt.subplots(
        ncols=n_param_kinds,
        nrows=1,
        figsize=(4 * n_param_kinds, 3),
        gridspec_kw=gridspecs
    )

    axs = np.atleast_1d(axs)

    labels_used = {"prior": False, "posterior": False, "bound": False}
    y_offset = {'charge': 0.2, 'sigma': 0.05}
    scale = {'charge': 0.1, 'sigma': 0.015}
    ylabels = {'charge': 'Charge [e]', 'sigma': '$\\sigma$ [nm]'}

    for i, (p_kind, ax) in enumerate(zip(param_kinds, axs)):
        x_offset = 0
        x_lim_low = -0.5
        x_lim_high = sum(1 for name in results.bounds_explicit.params if p_kind in name)
        bounds_raw = [
            bound
            for name, bound in results.bounds_explicit._bounds.items()
            if p_kind in name
        ]
        y_scale = np.abs(np.max(bounds_raw) - np.min(bounds_raw))
        for param in results.labels_explicit_:
            if p_kind not in param:
                continue
            if len(param.split()) == 1:
                continue
            atomtype_idx = results.labels_explicit_.index(param)
            posterior = results.chain_explicit_[:, atomtype_idx]
            bound = results.bounds_explicit._bounds[param]

            x_min = bound[0] - y_offset[p_kind]
            x_max = bound[1] + y_offset[p_kind]
            x = torch.linspace(x_min, x_max, 1000)

            # Plot prior
            if param == results.implicit_param:
                pass
            else:
                all_keys = results.priors.keys()
                prior_key = ''.join(key for key in all_keys if f'{param} ' in key)
                prior = results.priors[prior_key]
                y = prior.log_prob(x).exp()
                ax.fill_between(
                    - y * scale[p_kind] + x_offset,
                    x,
                    color=color_prior,
                    lw=0,
                    label='prior' if not labels_used["prior"] else None,
                )
                labels_used["prior"] = True
                x_lim_low = min(- (max(y) * scale[p_kind] * 1.1), -0.6)

            # Posterior
            kde = gaussian_kde(posterior)
            posterior_kde = kde.evaluate(x)
            ax.fill_between(
                posterior_kde * scale[p_kind] + x_offset,
                x,
                color=color_posterior,
                lw=0,
                label='posterior' if not labels_used["posterior"] else None,
            )

            labels_used["posterior"] = True
            x_lim_high = max((max(posterior_kde) * scale[p_kind] + x_offset) * 1.1, 0.5)

            # Bound as errorbar
            bound_center = np.mean(bound)
            yerr = np.array([[bound_center - bound[0]], [bound[1] - bound_center]])
            ax.errorbar(
                [x_offset],
                [bound_center],
                yerr=yerr,
                lw=2,
                ls='',
                capsize=5,
                markeredgewidth=2,
                color='k',
                label='bound' if not labels_used["bound"] else None,
            )
            labels_used["bound"] = True

            # Add value label
            ax.text(
                x_offset,
                bound[0] - 0.1 * y_scale,
                f'{x[posterior_kde.argmax()]:.3f}',
                ha='center',
                va='top',
                fontsize=10,
                color='tab:red',
                fontweight='bold',
            )

            x_offset += 1

            ax.set_ylabel(ylabels[p_kind])
            if i == 0:
                ax.legend(
                    ncol=3,
                    loc='upper center',
                    frameon=False,
                    bbox_to_anchor=(0.5, 1.15)
                )

        ax.set_xlim(x_lim_low, x_lim_high)

        y_low = np.min(bounds_raw) - 0.25 * y_scale
        y_high = np.max(bounds_raw) + 0.25 * y_scale
        ax.set_ylim(y_low, y_high)
        ax.tick_params(axis='both', direction='in')
        ax.set_xticks(np.arange(0, len(results.bounds_explicit.params), 1))

        xtick_labels = [p.split(maxsplit=1)[-1] for p in results.bounds_explicit.params]
        ax.set_xticklabels(xtick_labels, rotation=30)
        ax.set_xlabel('Atomtype')

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches='tight')
        plt.close(fig)  # Prevents display in Jupyter Notebook
    else:
        plt.show()


def plot_corner(
    samples,
    labels=None,
    levels=5,
    quantiles=[0.16, 0.5, 0.84],
    figsize=6,
    cmap=plt.cm.Reds,
    scatter_alpha=0.2,
    fn_out=None,
    transparent=True
) -> None:
    """
    Corner plot with 1D histograms and 2D KDE contourf + scatter background.
    The lowest density level is transparent; contours are outlined in black.
    """
    samples = np.asarray(samples)
    n_params = samples.shape[1]
    gridspecs = {'wspace': 0.05, 'hspace': 0.05}
    figsize = (figsize * n_params / 3, figsize * n_params / 3)
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize, gridspec_kw=gridspecs)

    # Create transparent colormap: first level transparent
    colors = cmap(np.linspace(0, 1, levels))
    colors[0, -1] = 0.0  # set alpha of lowest level to 0
    transparent_cmap = ListedColormap(colors)

    xlims = (samples.min(axis=0), samples.max(axis=0))
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i == j:
                x_data = samples[:, i]
                x_min, x_max = xlims[0][i], xlims[1][i]
                x = np.linspace(x_min, x_max, 1000)
                kde = gaussian_kde(x_data)

                if '\\sigma' in labels[i]:
                    color = 'k'
                else:
                    color = transparent_cmap.colors[-1]
                ax.plot(x, kde(x), color=color, lw=3)
                ax.fill_between(x, 0, kde(x), color=color, alpha=0.5)

                if quantiles:
                    # Plot vertical lines
                    q_values = np.quantile(samples[:, i], quantiles)
                    for q in q_values:
                        ax.axvline(q, color='black', linestyle='--', linewidth=1)

                    # Add text labels above histogram
                    lower = q_values[1] - q_values[0]
                    upper = q_values[2] - q_values[1]
                    quantiles_label = (
                        f"{labels[i]}\n"
                        f"${q_values[1]:.3f}^{{+{upper:.3f}}}_{{-{lower:.3f}}}$"
                    )

                    ax.text(
                        0.5, 1, quantiles_label,
                        transform=ax.transAxes,
                        ha='center',
                        va='bottom'
                    )

                ax.tick_params(axis='y', length=0)
                ax.tick_params(axis='x', length=5, direction='in')
                ax.set_xlim(x_min, x_max)
            elif i > j:
                x, y = samples[::10, j], samples[::10, i]
                ax.scatter(
                    x, y, s=3, lw=0, alpha=scatter_alpha, color="k", rasterized=True
                )

                try:
                    kde = gaussian_kde(np.vstack([x, y]))
                    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                    zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

                    # Define contour levels
                    levels_lin = np.linspace(zi.min(), zi.max(), levels)

                    # Filled contour
                    ax.contourf(xi, yi, zi, levels=levels_lin, cmap=transparent_cmap)

                    # Black contour lines
                    ax.contour(
                        xi, yi, zi, levels=levels_lin, colors='black', linewidths=0.5
                    )
                except np.linalg.LinAlgError:
                    pass

                ax.tick_params(
                    direction='in',
                    top=True, right=True,
                    which='both', length=5
                )
                ax.set_xlim(xlims[0][j], xlims[1][j])
            else:
                ax.axis('off')

            # Labels and ticks
            # Ensure at least 3 major ticks
            from matplotlib.ticker import MaxNLocator
            locator_kws = {'nbins': 'auto', 'steps': np.arange(1, 11), 'min_n_ticks': 3}
            ax.xaxis.set_major_locator(MaxNLocator(**locator_kws))
            ax.yaxis.set_major_locator(MaxNLocator(**locator_kws))

            if i == n_params - 1 and j < n_params:
                ax.set_xlabel(labels[j] if labels else f"$\\theta_{{{j}}}$")
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i] if labels else f"$\\theta_{{{i}}}$")
            else:
                ax.set_yticklabels([])

            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('center')

    fig.align_ylabels()
    fig.align_xlabels()

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches='tight', dpi=200, transparent=transparent)
        plt.close(fig)  # Prevents display in Jupyter Notebook
    else:
        plt.show()  # Display only if not saving
