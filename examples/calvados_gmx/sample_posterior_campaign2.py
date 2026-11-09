"""Posterior of Hst5's nine sampled lambdas against the experimental Rg.

Uses the ``campaign2`` stickiness-mean surrogate (a linear regression on
chemically grouped stickiness plus a local GP on its residuals) and BFF's
public MCMC machinery:

- Gaussian likelihood of the experimental Rg, with the model-error sigma
  sampled as a nuisance parameter (``log sigma``) and the experimental
  error as the accepted deviation ``tolerance``. This drives
  ``bff.bayes.likelihoods.gaussian_log_likelihood`` through a
  ``LearningProblem`` whose surrogate wraps the linear mean and the GP
  residual.
- Truncated beta priors from ``examples/calvados/generated/priors``,
  restricted to ``lambda_bounds.BOUNDS`` and tempered by
  ``PRIOR_EXPONENT = 0.5`` (the log prior is halved).
- The experimental value is read from ``examples/calvados/c2_rg.csv``
  (``exp_rg``, ``exp_rg_err``) for the Hst5 sequence.

With one scalar observation and a free sigma both the lambdas and sigma
are only weakly identified. The lambda marginals are the prior reweighted
by the band of calibrations that reproduce 1.34 nm, and the sigma
posterior is broad with a ridge near zero: any lambda that matches the
experimental Rg within the experimental error lets the model take zero
discrepancy, so sigma does not settle on the surrogate's own ~0.02 nm
interpolation error.

Outputs in ``campaign2/posterior/``: ``corner.png``, ``chain.npz``,
``posterior_summary.json``, ``mcmc.ckpt``.

Run from the repository root:

    python -m examples.calvados_gmx.sample_posterior_campaign2
"""

import csv
import json
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import scipy.stats
import torch

sys.path.insert(0, ".")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import corner  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bff.bayes.gaussian_process import LGPCommittee  # noqa: E402
from bff.bayes.learning import LearningProblem  # noqa: E402
from bff.bayes.likelihoods import gaussian_log_likelihood  # noqa: E402
from bff.bayes.posterior import log_posterior  # noqa: E402
from bff.bayes.utils import resolve_device  # noqa: E402
from bff.mcmc.proposal import AdaptiveGaussianProposal  # noqa: E402
from bff.mcmc.sampler import Sampler  # noqa: E402
from examples.calvados_gmx.fit_campaign2_interim import GROUPS  # noqa: E402
from examples.calvados_gmx.lambda_bounds import BOUNDS  # noqa: E402
from examples.calvados_gmx.run_campaign2 import HST5, THREE  # noqa: E402

HERE = Path(__file__).resolve().parent
FIT = HERE / "campaign2" / "fit_interim"
OUTPUT = HERE / "campaign2" / "posterior"
PRIORS = HERE.parent / "calvados" / "generated" / "priors"
C2_RG = HERE.parent / "calvados" / "c2_rg.csv"

PRIOR_EXPONENT = 0.5
SIGMA_PRIOR_SCALE = 0.05
SEED = 2026
N_WALKERS = 120
TOTAL_STEPS = 30000
WARMUP = 5000
THIN = 10
PROGRESS_STRIDE = 2500
PROPOSAL_SCALE = 0.05
SIGMA_PROPOSAL_SCALE = 0.03
PLOT_SAMPLES = 50000
CORNER_BINS = 40
SIGMA_PLOT_MAX = 0.30


def experimental_rg(fasta=HST5):
    """``(mean, error)`` of the listed experimental Rg for ``fasta``."""
    with open(C2_RG, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["fasta"] == fasta]
    if len(rows) != 1:
        raise ValueError(
            f"c2_rg.csv: expected one row with fasta {fasta}, found {len(rows)}"
        )
    return float(rows[0]["exp_rg"]), float(rows[0]["exp_rg_err"])


def beta_priors(parameter_names):
    """``(alpha, beta)`` per parameter from the prepared CALVADOS priors."""
    alpha, beta = [], []
    with open(PRIORS / "beta_parameters.csv", newline="") as handle:
        rows = {row["parameter"]: row for row in csv.DictReader(handle)}
    for name in parameter_names:
        if name not in rows:
            raise ValueError(f"beta_parameters.csv: no row for {name!r}")
        alpha.append(float(rows[name]["alpha"]))
        beta.append(float(rows[name]["beta"]))
    return np.array(alpha), np.array(beta)


def lambda_bounds(names):
    """Hard per-residue box in ``parameter_names`` order."""
    resolvers = {three: one for one, three in THREE.items()}
    pairs = [BOUNDS[resolvers[name.split()[1]]] for name in names]
    return np.array([pair[0] for pair in pairs]), np.array(
        [pair[1] for pair in pairs]
    )


def stickiness_weights():
    """``(n_residues, n_groups)`` sequence-weighted group stickiness."""
    residues = [one for one in sorted(set(HST5)) if one != "E"]
    return np.column_stack(
        [
            np.array(
                [
                    HST5.count(one) / len(HST5) if one in members else 0.0
                    for one in residues
                ]
            )
            for members in GROUPS.values()
        ]
    )


class StickinessMeanSurrogate:
    """GP residual plus the linear stickiness mean, as a learning model.

    Exposes the attributes ``LearningProblem`` and
    ``gaussian_log_likelihood`` read: ``predict``, ``members``, ``y_ref``,
    ``n_curves``, ``nuisance`` (``None`` learns sigma), ``tolerance`` and
    ``n_eff`` (one scalar observation).
    """

    def __init__(self, residual, coefficients, y_ref, tolerance):
        names = ["intercept", *GROUPS]
        self.coef = torch.as_tensor(
            [float(coefficients[name]) for name in names], dtype=torch.float64
        )
        self.feature_weights = torch.as_tensor(
            stickiness_weights(), dtype=torch.float64
        )
        self.residual = residual
        self.members = residual.members
        self.y_ref = np.asarray([y_ref], dtype=float)
        self.n_curves = 1
        self.nuisance = None
        self.tolerance = float(tolerance)
        self.n_eff = 1.0

    @property
    def n_inputs(self):
        return self.residual.n_inputs

    @property
    def n_outputs(self):
        return 1

    def to(self, device, dtype=None):
        self.coef = self.coef.to(device, dtype)
        self.feature_weights = self.feature_weights.to(device, dtype)
        self.residual.to(device, dtype)
        return self

    def predict(self, X):
        features = X.to(self.feature_weights) @ self.feature_weights
        mean = self.coef[0] + features @ self.coef[1:]
        return mean.unsqueeze(1) + self.residual.predict(X)


class TruncatedTemperedBeta:
    """Beta priors on the lambdas, truncated to the box, plus half-normal sigma.

    The log density is multiplied by ``exponent``; lambda columns outside
    ``[lower, upper]`` get ``-inf``. Sigma is sampled as ``log sigma`` with
    the Jacobian of the change of variables included.
    """

    def __init__(self, names, alpha, beta, lower, upper, exponent, sigma_scale):
        self.parameter_names = (*names, "log sigma")
        self.kind = "beta"
        self.exponent = float(exponent)
        self.alpha = torch.as_tensor(alpha, dtype=torch.float64)
        self.beta = torch.as_tensor(beta, dtype=torch.float64)
        self.log_beta = (
            torch.lgamma(self.alpha)
            + torch.lgamma(self.beta)
            - torch.lgamma(self.alpha + self.beta)
        )
        self.lower = torch.as_tensor(lower, dtype=torch.float64)
        self.upper = torch.as_tensor(upper, dtype=torch.float64)
        self.sigma_scale = float(sigma_scale)
        self.log_sigma_norm = math.log(2.0) - 0.5 * math.log(
            2.0 * math.pi * self.sigma_scale**2
        )

    def log_prob(self, theta):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        alpha = self.alpha.to(theta)
        beta = self.beta.to(theta)
        lower = self.lower.to(theta)
        upper = self.upper.to(theta)
        lambdas = theta[:, :-1]
        log_sigma = theta[:, -1]

        x = lambdas.clamp(1e-6, 1.0 - 1e-6)
        log_beta_density = (
            (alpha - 1.0) * torch.log(x)
            + (beta - 1.0) * torch.log1p(-x)
            - self.log_beta.to(theta)
        )
        sigma = log_sigma.exp()
        log_sigma_density = (
            self.log_sigma_norm
            - 0.5 * (sigma / self.sigma_scale) ** 2
            + log_sigma
        )
        value = log_beta_density.sum(dim=1) + log_sigma_density
        inside = ((lambdas >= lower) & (lambdas <= upper)).all(dim=1)
        return torch.where(
            inside,
            self.exponent * value,
            torch.full_like(value, -torch.inf),
        )


def initial_walkers(prior, surrogate, y_exp, n_walkers):
    """Half near the training points that predict ``y_exp``, half from the prior."""
    seeds = surrogate.members[0].X_train.to(torch.float64)
    with torch.no_grad():
        predicted = surrogate.predict(seeds)[:, 0]
    order = torch.argsort((predicted - y_exp).abs())[:20]
    candidates = seeds[order].numpy()
    lower = prior.lower.numpy()
    upper = prior.upper.numpy()
    width = upper - lower
    rng = np.random.default_rng(SEED)

    n_seed = 3 * n_walkers // 5
    starts = np.empty((n_walkers, len(prior.parameter_names)))
    for index in range(n_seed):
        base = candidates[index % len(candidates)]
        jitter = rng.uniform(-1.0, 1.0, size=len(lower))
        starts[index, :-1] = np.clip(
            base + 0.02 * width * jitter, lower, upper
        )
    cdf_low = scipy.stats.beta.cdf(lower, prior.alpha.numpy(), prior.beta.numpy())
    cdf_high = scipy.stats.beta.cdf(
        upper, prior.alpha.numpy(), prior.beta.numpy()
    )
    uniform = rng.uniform(
        cdf_low, cdf_high, size=(n_walkers - n_seed, len(lower))
    )
    starts[n_seed:, :-1] = scipy.stats.beta.ppf(
        uniform, prior.alpha.numpy(), prior.beta.numpy()
    )
    starts[:, -1] = np.log(np.abs(rng.normal(0.02, 0.005, n_walkers)))
    return torch.as_tensor(starts, dtype=torch.float32)


def draw_prior_marginal(ax, grid, density, n_samples, axis_range):
    """Draw a known prior density on a corner diagonal (counts) panel."""
    area = np.sum(density) * (grid[1] - grid[0])
    bin_width = (axis_range[1] - axis_range[0]) / CORNER_BINS
    curve = density / area * n_samples * bin_width
    ax.plot(grid, curve, color="tab:orange", lw=2.2)
    ax.set_ylim(0, max(ax.get_ylim()[1], 1.05 * curve.max()))


def plot_corner(
    flat, names, alpha, beta, lower, upper, sigma_scale, exponent, title
):
    """Corner plot of the chain with the prior marginals overlaid."""
    labels = [name.replace("lambda ", "") for name in names] + ["sigma (nm)"]
    ranges = np.vstack(
        [np.column_stack([lower, upper]), [0.0, SIGMA_PLOT_MAX]]
    )
    figure = plt.figure(figsize=(24, 24))
    corner.corner(
        flat,
        labels=labels,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
        bins=CORNER_BINS,
        range=ranges,
        fig=figure,
    )
    positions = {}
    for ax in figure.axes:
        spec = ax.get_subplotspec()
        if spec is not None:
            positions[(spec.rowspan.start, spec.colspan.start)] = ax

    n_samples = len(flat)
    for column in range(len(alpha)):
        ax = positions.get((column, column))
        if ax is None:
            continue
        width = upper[column] - lower[column]
        grid = np.linspace(
            lower[column] + 1e-4 * width, upper[column] - 1e-4 * width, 512
        )
        density = scipy.stats.beta.pdf(
            grid, alpha[column], beta[column]
        ) ** exponent
        draw_prior_marginal(
            ax, grid, density, n_samples, ranges[column]
        )

    sigma_axis = positions.get((len(alpha), len(alpha)))
    if sigma_axis is not None:
        grid = np.linspace(1e-4, 6.0 * sigma_scale, 1024)
        density = (
            2.0 * scipy.stats.norm.pdf(grid, scale=sigma_scale)
        ) ** exponent
        draw_prior_marginal(sigma_axis, grid, density, n_samples, ranges[-1])

    handles = [
        Patch(facecolor="0.75", edgecolor="0.4", label="posterior"),
        Line2D(
            [],
            [],
            color="tab:orange",
            lw=2.0,
            label=f"prior (tempered {exponent}, truncated)",
        ),
    ]
    figure.legend(
        handles=handles, loc="upper right", fontsize=20, frameon=False
    )
    figure.suptitle(title, fontsize=18)
    return figure


def summarize(chain, parameter_names):
    """Per-parameter mean, standard deviation, median, and 5-95% interval."""
    flat = chain.reshape(-1, chain.shape[-1])
    lower, upper = np.percentile(flat, [5, 95], axis=0)
    summary = {}
    for column, name in enumerate(parameter_names):
        values = flat[:, column]
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p5": float(lower[column]),
            "p95": float(upper[column]),
        }
    return summary, flat


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    y_exp, sigma_exp = experimental_rg()
    residual = LGPCommittee.load(FIT / "model_linear_mean.lgp")
    metrics = json.loads((FIT / "metrics.json").read_text())
    surrogate = StickinessMeanSurrogate(
        residual, metrics["linear_mean"]["coefficients"], y_exp, sigma_exp
    )

    names = list(residual.parameter_names)
    alpha, beta = beta_priors(names)
    lower, upper = lambda_bounds(names)
    prior = TruncatedTemperedBeta(
        names,
        alpha,
        beta,
        lower,
        upper,
        PRIOR_EXPONENT,
        SIGMA_PRIOR_SCALE,
    )

    device = resolve_device("auto")
    problem = LearningProblem(models={"rg": surrogate}).to_torch(
        device, torch.float32
    )
    log_probability = partial(
        log_posterior,
        priors=prior,
        log_likelihood_fn=partial(gaussian_log_likelihood, problem=problem),
    )
    variances = np.concatenate(
        [np.full(len(names), PROPOSAL_SCALE**2), [SIGMA_PROPOSAL_SCALE**2]]
    )
    proposal = AdaptiveGaussianProposal(
        torch.as_tensor(np.diag(variances), dtype=torch.float32),
        device=device,
    )
    sampler = Sampler(
        log_prob=log_probability,
        proposal=proposal,
        rng=torch.Generator().manual_seed(SEED),
        device=device,
        dtype=torch.float32,
    )
    p0 = initial_walkers(prior, surrogate, y_exp, N_WALKERS)
    print(
        f"Rg = {y_exp:.2f} +- {sigma_exp:.2f} nm; "
        f"{len(names)} lambdas + log sigma; {N_WALKERS} walkers on {device}"
    )
    checkpoint = None
    with torch.inference_mode():
        for checkpoint in sampler.run(
            p0,
            total_steps=TOTAL_STEPS,
            warmup=WARMUP,
            thin=THIN,
            progress_stride=PROGRESS_STRIDE,
            fn_checkpoint=OUTPUT / "mcmc.ckpt",
            checkpoint_compatibility={
                "model": "campaign2/fit_interim/model_linear_mean.lgp",
                "prior_exponent": PRIOR_EXPONENT,
                "y_exp": y_exp,
                "sigma_exp": sigma_exp,
            },
        ):
            rate = checkpoint.acceptance_rate
            rate_text = "n/a" if rate is None else f"{rate:.3f}"
            print(
                f"  step {checkpoint.step}/{TOTAL_STEPS}  "
                f"acceptance {rate_text}  converged {checkpoint.converged}"
            )

    chain = sampler.chain.detach().cpu().numpy()
    log_prob = sampler.chain_logp.detach().cpu().numpy()
    np.savez_compressed(
        OUTPUT / "chain.npz", chain=chain, log_prob=log_prob
    )
    summary, flat = summarize(chain, prior.parameter_names)
    sigma = np.exp(flat[:, -1])
    sigma_lower, sigma_upper = np.percentile(sigma, [5, 95])
    summary["sigma (nm)"] = {
        "mean": float(sigma.mean()),
        "std": float(sigma.std()),
        "median": float(np.median(sigma)),
        "p5": float(sigma_lower),
        "p95": float(sigma_upper),
    }
    diagnostics = sampler.diagnostics
    report = {
        "model": "campaign2/fit_interim/model_linear_mean.lgp",
        "fit_metrics": {
            "n_samples": metrics["n_samples"],
            "linear_mean_test_rmse_nm": metrics["linear_mean"]["test_metrics"][
                "rmse_nm"
            ],
        },
        "likelihood": "Gaussian in Rg; sigma model-error nuisance + tolerance",
        "experimental": {
            "rg_nm": y_exp,
            "error_nm": sigma_exp,
            "source": "examples/calvados/c2_rg.csv (exp_rg, exp_rg_err)",
        },
        "prior": {
            "kind": "truncated beta x half-normal sigma, tempered",
            "exponent": PRIOR_EXPONENT,
            "sigma_scale_nm": SIGMA_PRIOR_SCALE,
            "bounds": "lambda_bounds.BOUNDS",
        },
        "seed": SEED,
        "n_walkers": N_WALKERS,
        "total_steps": TOTAL_STEPS,
        "warmup": WARMUP,
        "thin": THIN,
        "converged": sampler.converged,
        "acceptance_rate": (
            None if checkpoint is None else checkpoint.acceptance_rate
        ),
        "max_rhat": None if diagnostics is None else diagnostics.max_rhat,
        "min_ess": None if diagnostics is None else diagnostics.min_ess,
        "parameters": summary,
    }
    (OUTPUT / "posterior_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    plot_flat = flat
    if len(plot_flat) > PLOT_SAMPLES:
        rng = np.random.default_rng(SEED)
        rows = rng.choice(len(plot_flat), size=PLOT_SAMPLES, replace=False)
        plot_flat = plot_flat[rows]
    plot_flat = plot_flat.copy()
    plot_flat[:, -1] = np.exp(plot_flat[:, -1])
    figure = plot_corner(
        plot_flat,
        names,
        alpha,
        beta,
        lower,
        upper,
        SIGMA_PRIOR_SCALE,
        PRIOR_EXPONENT,
        f"Hst5 posterior: Rg = {y_exp:.2f} +- {sigma_exp:.2f} nm, "
        f"prior tempered to {PRIOR_EXPONENT}",
    )
    figure.savefig(OUTPUT / "corner.png", dpi=120)
    plt.close(figure)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
