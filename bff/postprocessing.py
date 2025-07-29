import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from scipy.stats import gaussian_kde

from .structures import Specs, OptimizationResults
from .bayes.utils import valid_bounds


def draw_samples(
    results: OptimizationResults,
    n_samples: int = 10,
    distribution: str = 'normal',
    confidence: float = 0.9,
    fn_out: str = None
) -> np.ndarray:
    """
    Draw samples from the posterior using
    either uniform or Laplace distribution.
    """
    lower, upper = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    samples = results.chain_implicit_[:, :results.n_params_implicit]
    confint = np.quantile(samples, [lower, upper], axis=0)

    if distribution == 'normal':
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
    elif distribution != 'uniform':
        raise ValueError(
            (
                f'Unknown distribution "{distribution}". '
                'Options are "uniform" or "normal".'
            )
        )

    samples_out = np.empty((n_samples, len(results.bounds_implicit.params)))
    specs = Specs(results.data)
    uniform_range = np.diff(confint, axis=0).ravel()

    i, attempts, max_attempts = 0, 0, n_samples * 1000  # Safety limit

    while i < n_samples and attempts < max_attempts:
        if distribution == 'uniform':
            random_values = np.random.rand(len(uniform_range)) * uniform_range
            sample = random_values + confint[0]
            sample.reshape(1, -1)
        else:
            if cov.size == 1:
                sample = np.random.normal(mean, cov, size=1)[:, np.newaxis]
            else:
                sample = np.random.multivariate_normal(mean, cov, size=1)

        is_within_confint = np.all(
            np.logical_and(sample >= confint[0], sample <= confint[1])
        )
        if is_within_confint and valid_bounds(sample, specs):
            samples_out[i] = sample
            i += 1
        attempts += 1

    if i < n_samples:
        raise RuntimeError(
            "Failed to generate enough valid samples within max attempts."
        )

    if fn_out:
        np.save(fn_out, samples_out)

    return samples_out


def plot_distributions(results: OptimizationResults, fn_out: str = None) -> None:

    labels_used = {"prior": False, "posterior": False, "bound": False}

    x_offset = 0
    x_lim_low, x_lim_high = -0.5, len(results.atomtypes)

    fig, ax = plt.subplots(figsize=(4, 3))
    for posterior, (param, bound) in zip(
        results.chain_explicit_[:, :results.n_params_explicit].T,
        results.bounds_explicit.bounds.items()
    ):
        x_min = bound[0] - 0.2
        x_max = bound[1] + 0.2
        x = torch.linspace(x_min, x_max, 1000)

        # Prior
        if param.split()[1] != results.implicit_atomtype:
            prior = results.priors[param + ' Normal']
            y = prior.log_prob(x).exp()
            ax.fill_between(
                -y * 0.1 + x_offset,
                x,
                color='gray',
                lw=0,
                label='prior' if not labels_used["prior"] else None,
            )
            labels_used["prior"] = True
            x_lim_low = min(- max(y) * 0.1 * 1.1, -0.5)

        # Posterior
        kde = gaussian_kde(posterior)
        posterior_kde = kde.evaluate(x)
        ax.fill_between(
            posterior_kde * 0.1 + x_offset,
            x,
            color='tab:red',
            lw=0,
            label='posterior' if not labels_used["posterior"] else None,
        )

        labels_used["posterior"] = True
        x_lim_high = max((max(posterior_kde) * 0.1 + x_offset) * 1.05, 0.5)

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
            bound[0] - 0.1,
            f'{x[posterior_kde.argmax()]:.3f}',
            ha='center',
            va='top',
            fontsize=10,
            color='tab:red',
            fontweight='bold',
        )

        x_offset += 1

    ax.set_xlim(x_lim_low, x_lim_high)
    ax.tick_params(axis='both', direction='in')
    ax.set_xticks(np.arange(0, len(results.atomtypes), 1))
    ax.set_xticklabels(results.atomtypes, rotation=30)
    ax.set_xlabel('Atomtype')
    ax.set_ylabel('Charge [e]')
    ax.legend(ncol=3, loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.15))

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches='tight')
        plt.close(fig)  # Prevents display in Jupyter Notebook
    else:
        plt.show()


def plot_corner(
    samples,
    labels=None,
    bins=30,
    levels=5,
    figsize=6,
    cmap=plt.cm.Reds,
    scatter_alpha=0.1,
    fn_out=None
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

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                ax.hist(
                    samples[:, i],
                    bins=bins,
                    histtype='step',
                    linewidth=3,
                    color=transparent_cmap.colors[-1]
                )

                q_values = np.quantile(samples[:, i], [0.016, 0.5, 0.84])
                # Plot vertical lines
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
                    va='bottom',
                    fontsize=10
                )
            elif i > j:
                x, y = samples[::10, j], samples[::10, i]
                ax.scatter(x, y, s=3, alpha=scatter_alpha, color="gray")

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

            else:
                ax.axis('off')

            # Labels and ticks
            # Ensure at least 3 major ticks
            from matplotlib.ticker import MaxNLocator
            locator_kws = {'nbins': 'auto', 'steps': np.arange(1, 11), 'min_n_ticks': 3}
            ax.xaxis.set_major_locator(MaxNLocator(**locator_kws))
            ax.yaxis.set_major_locator(MaxNLocator(**locator_kws))

            ax.tick_params(
                direction='in',
                top=True, right=True,
                which='both', length=4
            )
            if i == n_params - 1 and j < n_params:
                ax.set_xlabel(labels[j] if labels else f"$\\theta_{{{j}}}$")
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i] if labels else f"$\\theta_{{{i}}}$")
            else:
                ax.set_yticklabels([])
            ax.tick_params(direction='in', which='both')

            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('center')

    fig.align_ylabels()

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches='tight')
        plt.close(fig)  # Prevents display in Jupyter Notebook
    else:
        plt.show()  # Display only if not saving
