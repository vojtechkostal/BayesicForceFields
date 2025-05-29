import corner
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import numpy as np

from .structures import Specs, OptimizationResults
from .bayes.utils import valid_bounds

__all__ = ['draw_samples', 'plot_confidence_intervals']


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


def plot_confidence_intervals(
        results: OptimizationResults,
        confidence: float = 0.95,
        fn_out: str = None
) -> None:

    param_types = [p.split()[0] for p in results.bounds_explicit.params]
    unique_param_types, counts = np.unique(param_types, return_counts=True)
    n_cols = len(unique_param_types)

    gridspec_kw = dict(width_ratios=counts, wspace=0.5)
    fig, axs = plt.subplots(
        1, n_cols, figsize=(np.sum(counts)*0.75, 3), gridspec_kw=gridspec_kw)
    axs = np.atleast_1d(axs)
    for ax, name in zip(axs, unique_param_types):
        bounds = np.array([
            v for p, v in results.bounds_explicit.bounds.items()
            if name in p
        ])
        bound_ranges = np.diff(bounds, axis=1).flatten()
        bound_bottoms = bounds[:, 0]

        x = np.arange(1, len(bounds) + 1, 1)

        bar_kws = {
            'color': 'gray',
            'alpha': 0.5,
            'label': 'bounds'
        }
        ax.bar(x, bound_ranges, bottom=bound_bottoms, **bar_kws)
        lower, median, upper = results.quantiles(confidence)[:, :len(bounds)]
        errobar_kws = {
            'color': 'tab:red',
            'elinewidth': 2,
            'markeredgewidth': 2,
            'ls': '',
            'marker': 'o',
            'capsize': 5,
            'label': f'{(confidence*100):.0f}% confidence interval'
        }
        ax.errorbar(
            x,
            median,
            yerr=(median - lower, upper - median),
            **errobar_kws
        )

        # Set the x-ticks and labels
        ax.set_xlabel('Atomtype')
        ax.tick_params(direction='in')
        ax.set_xticks(x)
        tick_labels = [
            p.split()[1] for p in results.bounds_explicit.params if name in p
        ]
        ax.set_xticklabels(tick_labels, rotation=45)

        # Plot the zero line for charges
        if name == 'charge':
            ax.axhline(0, c='k', ls='--')

        ax.set_ylabel(name.capitalize())
        y_range = bounds.max() - bounds.min()
        ax.set_ylim(bounds.min() - 0.1 * y_range, bounds.max() + 0.1 * y_range)

    axs[0].legend()

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_corner(
    results: OptimizationResults,
    quantiles: list[float]=[0.025, 0.5, 0.975],
    cmap=None,
    fn_out: str = None
):

    
    contour_levels = [0.3, 0.5, 0.7, 0.9]
    fill_colors_rgb = [i for i in cmap]
    alpha_fill = 1.0
    transparent_white = (1, 1, 1, 0) # RGBA
    colors_for_map = [transparent_white] + [to_rgba(c, alpha=alpha_fill) for c in fill_colors_rgb]

    custom_cmap = ListedColormap(colors_for_map)
    results.get_chain()
    fig = corner.corner(
        results.chain_implicit_,
        labels=results.labels_implicit_,
        show_titles=True,
        title_fmt=".2f",
        hist_bin_factor=1.0,  # Increase the number of histogram bins
        hist_kwargs={"color": colors_for_map[-1], "linewidth": 3},  # Red histograms
        contourf_kwargs={"cmap": None, 'colors': colors_for_map},  # Red gradient colormap
        contour_kwargs={"colors": "k", "linewidths": 2},  # Black contour lines
        smooth=0.5,
        levels=contour_levels,  # Contour levels
        fill_contours=True,  # Fill the contour levels
        plot_density=False,  # Ensure density plots are used
        plot_contours=True,  # Ensure contour plots are drawn
        quantiles=quantiles,  # Add quantiles to the plot
    )

    if fn_out is not None:
        plt.savefig(fn_out, bbox_inches='tight')
        plt.close(fig)  # Prevents display in Jupyter Notebook
    else:
        plt.show()  # Display only if not saving
