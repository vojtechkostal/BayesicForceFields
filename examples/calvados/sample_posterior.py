"""Sample stickiness posteriors with BFF from the direct-discrepancy GPs.

Requires the artifacts of ``train_discrepancy`` and ``prepare_priors``:

```bash
python -m examples.calvados.sample_posterior
```

The likelihood treats the GP predictive means as chi-squared discrepancies
against a zero reference (no discrepancy):

    log L(theta) = -0.5 * (w_rg * chi2_rg(theta) + w_pre * chi2_pre(theta))

with the paper's cost-function structure in sum form: the stored records
hold means over proteins (51 Rg, 5 PRE), and independent observations
multiply into a sum, so ``w_rg = 51`` and ``w_pre = 5 * 0.1``. As in the
paper's annealed cost, the log prior is weighted by ``theta = 0.05``: the
sampled target is ``log L + 0.05 * log prior`` (a tempered posterior).
Negative GP means are clamped to zero because chi-squares cannot be
negative. Each of the three prepared priors is sampled in turn; results go
to ``generated/posterior/<prior>/``. Walkers start from prior draws, with
the KDE components all occupied round-robin, so the chain covers the
prior's modes from the first step.

This is BFF's public MCMC machinery driven by example-specific likelihood
and prior objects; no ``bff learn`` stage is configured, because its
observable-versus-reference Gaussian likelihood would square the predicted
discrepancies a second time. The posterior is constrained only within the
region the optimization histories cover: outside it the GPs fall back to
their means and the chain follows the prior.
"""

import json
import math
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch

from bff.bayes.gaussian_process import LGPCommittee
from bff.bayes.posterior import log_posterior
from bff.bayes.utils import resolve_device
from bff.mcmc.proposal import AdaptiveGaussianProposal
from bff.mcmc.sampler import Sampler
from examples.calvados.prepare_priors import load_priors

SEED = 2026
# Sum-form likelihood: the stored chi-squareds are means over proteins
# (51 Rg, 5 PRE), but each protein is an independent experiment, so the
# likelihood of all observations is the sum. Multiplying each mean by its
# protein count (PRE keeps the paper's 0.1 weight) gives that sum.
WEIGHT_RG = 51.0
WEIGHT_PRE = 0.5
# The paper's annealed cost weighted the log prior by theta = 0.05
# (upstream optimize.py); the posterior uses the same tempering.
PRIOR_EXPONENT = 0.05
N_WALKERS = 140
TOTAL_STEPS = 30000
WARMUP = 2000
THIN = 10
PROGRESS_STRIDE = 2000
PROPOSAL_SCALE = 0.05
WALKER_JITTER = 0.02
MODELS = Path(__file__).resolve().parent / "generated" / "discrepancy_direct"
OUTPUT = Path(__file__).resolve().parent / "generated" / "posterior"
PRIOR_NAMES = ("tophat", "kde", "independent_beta")


def discrepancy_log_likelihood(models, weights):
    """Weighted chi-squared log likelihood from surrogate predictions.

    ``models`` maps QoI name to anything with a batch ``predict(X)`` returning
    one column per input row, such as an ``LGPCommittee``. ``weights`` maps
    the same names to the cost-function weights of the paper. Predictions
    are clamped at zero: chi-squares are nonnegative, and without the clamp
    a negative GP mean far outside the training data would be rewarded.
    """

    def log_likelihood(theta: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(len(theta), dtype=theta.dtype, device=theta.device)
        for name, weight in weights.items():
            predicted = models[name].predict(theta)[:, 0]
            value = value - weight * predicted.clamp_min(0.0)
        return 0.5 * value

    return log_likelihood


class StructuralBox:
    """Zero density outside padded per-dimension bounds of the GP training set.

    Deliberately crude and acknowledged as such: the bounds ignore
    correlations between dimensions, and the true trust region - the union
    of the optimization-path filaments - is a much smaller subset of this
    box. The padding keeps walkers off the exact min/max walls, where the
    Metropolis ensemble otherwise pins against ``-inf``.
    """

    PAD = 0.1

    def __init__(self, training):
        self.lower = (training.min(dim=0).values - self.PAD).clamp(0.0, 1.0)
        self.upper = (training.max(dim=0).values + self.PAD).clamp(0.0, 1.0)

    def is_valid(self, theta):
        eps = 1e-6
        return ((theta >= self.lower - eps) & (theta <= self.upper + eps)).all(dim=1)


class TemperedPrior:
    """A prior raised to ``exponent``: ``log_prob = exponent * base``.

    ``bff.bayes.posterior.log_posterior`` accepts anything with a batch
    ``log_prob``, so this implements the paper's ``theta = 0.05`` weighting
    of the log-prior term without changing BFF. With ``box``, a
    :class:`StructuralBox` additionally sends the density to zero outside
    the GP training bounds in any dimension - an acknowledged
    approximation that keeps sampling inside the surrogate's box (the real
    trust region is the filament union inside it, not the full box).
    """

    def __init__(self, base, exponent, box=None):
        self.base = base
        self.exponent = float(exponent)
        self.box = box
        self.parameter_names = base.parameter_names
        self.kind = base.kind

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        value = self.base.log_prob(theta).clone()
        if self.box is not None:
            inside = self.box.is_valid(theta)
            value = torch.where(
                inside,
                self.exponent * value,
                torch.full_like(value, -torch.inf),
            )
        else:
            value = self.exponent * value
        return value


def initial_walkers(prior, n_walkers, generator, seeds=None, box=None):
    """Starting points from the prior, covering all its modes.

    For the KDE, the 70 components are occupied round-robin so every mode
    has walkers from the first step. With a ``box``, prior draws are
    projected into it (their mass rarely intersects such a thin region);
    with ``seeds`` (GP training vectors) the remaining walkers start inside
    the training cloud with small jitter. Outside the cloud the surrogate
    reverts to a flat mean, so from prior draws alone a random walk can
    never discover the region where the data live. Initializations only
    place walkers; the proposal can move anywhere the target allows.
    """
    n_prior = n_walkers if seeds is None else n_walkers // 2
    picked = []
    if prior.kind == "uniform":
        picked.append(torch.rand((n_prior, len(prior.parameter_names)),
                                 generator=generator))
    elif prior.kind == "kde":
        centers = torch.tensor(prior.centers, dtype=torch.float32)
        assignment = torch.arange(n_prior) % len(centers)
        jitter = torch.randn((n_prior, len(prior.parameter_names)),
                             generator=generator)
        picked.append(
            (centers[assignment] + jitter * prior.bandwidth).clamp(0.01, 0.99)
        )
    else:
        rng = np.random.default_rng(SEED)
        batches = []
        while sum(len(b) for b in batches) < n_prior:
            columns = np.stack(
                [
                    rng.beta(float(a), float(b), 2 * n_prior)
                    for a, b in zip(prior.alpha, prior.beta)
                ],
                axis=1,
            )
            trial = torch.tensor(columns, dtype=torch.float32)
            batches.append(trial[torch.isfinite(prior.log_prob(trial))])
        picked.append(torch.cat(batches)[:n_prior])
    if box is not None:
        projected = picked[0].clamp(box.lower, box.upper)
        picked[0] = projected
    if seeds is not None:
        pool = seeds[torch.randint(len(seeds), (n_walkers - n_prior,),
                                   generator=generator)]
        jitter = (torch.rand(pool.shape, generator=generator) - 0.5) * (
            2 * WALKER_JITTER
        )
        picked.append((pool + jitter).clamp(0.01, 0.99))
    return torch.cat(picked)


def summarize(chain, log_prob, parameter_names):
    """Per-parameter posterior mean, standard deviation, and sample MAP."""
    flat = chain.reshape(-1, chain.shape[-1])
    flat_logp = log_prob.reshape(-1)
    map_row = flat[int(np.argmax(flat_logp))]
    lower, upper = np.percentile(flat, [5, 95], axis=0)
    summary = {}
    for column, name in enumerate(parameter_names):
        values = flat[:, column]
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "map": float(map_row[column]),
            "interval_5_95": [float(lower[column]), float(upper[column])],
        }
    return summary


def zero_log_likelihood(theta):
    """Baseline: zero likelihood, dropping the data term entirely."""
    return theta.new_zeros(len(theta))


def find_map(prior, log_likelihood, starts, device, box=None):
    """Maximum a posteriori of the tempered posterior, from several starts.

    The surrogates predict under ``torch.no_grad`` (their predictive
    gradient is not exposed), so this is derivative-free: Nelder-Mead on
    the unconstrained ``eta`` with ``theta = sigmoid(eta)`` plus the logit
    Jacobian, started from prior draws, CALVADOS2's optimum and the best
    chain samples, so the result is never worse than the best sample.
    ``box`` applies the structural box of the sampled target; starts
    outside it score ``-inf`` and simply lose to interior optima.
    """
    from scipy.optimize import minimize

    target = TemperedPrior(prior, PRIOR_EXPONENT, box=box)

    def log_target(theta):
        return log_posterior(
            theta, priors=target, log_likelihood_fn=log_likelihood
        )

    def neg_target_eta(eta_flat):
        eta = torch.tensor(eta_flat, dtype=torch.float32).unsqueeze(0)
        theta = torch.sigmoid(eta).clamp(1e-6, 1 - 1e-6)
        jacobian = (torch.log(theta) + torch.log1p(-theta)).sum()
        with torch.inference_mode():
            value = (log_target(theta) + jacobian).item()
        return -value

    best_theta, best_value = None, -torch.inf
    for start in starts:
        eta0 = torch.logit(start.clamp(0.01, 0.99)).numpy().ravel()
        result = minimize(neg_target_eta, eta0, method="Nelder-Mead",
                          options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-4})
        theta = torch.sigmoid(torch.tensor(result.x)).numpy()
        with torch.inference_mode():
            value = log_target(torch.tensor(theta, dtype=torch.float32)).item()
        if value > best_value:
            best_theta, best_value = theta, value
    return best_theta, best_value


def run_prior(name, prior, models, log_likelihood, device, output, seeds=None):
    """Sample one tempered posterior; write chain, summary, report."""
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    box = StructuralBox(seeds) if seeds is not None else None
    p0 = initial_walkers(prior, N_WALKERS, generator, seeds, box)
    if seeds is not None:
        target = TemperedPrior(prior, PRIOR_EXPONENT, box=box)
    else:
        target = TemperedPrior(prior, PRIOR_EXPONENT)
    log_probability = partial(
        log_posterior, priors=target, log_likelihood_fn=log_likelihood
    )
    n_dim = len(prior.parameter_names)
    proposal = AdaptiveGaussianProposal(
        torch.full((n_dim,), PROPOSAL_SCALE**2).diag(),
        device=device,
    )
    sampler = Sampler(
        log_prob=log_probability,
        proposal=proposal,
        device=device,
        dtype=torch.float32,
    )
    checkpoint_path = output / "mcmc.ckpt"
    with torch.inference_mode():
        for checkpoint in sampler.run(
            p0,
            total_steps=TOTAL_STEPS,
            warmup=WARMUP,
            thin=THIN,
            progress_stride=PROGRESS_STRIDE,
            fn_checkpoint=checkpoint_path,
            checkpoint_compatibility={
                "prior": name,
                "prior_exponent": PRIOR_EXPONENT,
                "weights": (
                    "none (prior-only baseline)"
                    if log_likelihood is zero_log_likelihood
                    else {"chi2_rg": WEIGHT_RG, "chi2_pre": WEIGHT_PRE}
                ),
                "run": "M1_24",
            },
        ):
            rate = (
                "n/a" if checkpoint.acceptance_rate is None
                else f"{checkpoint.acceptance_rate:.3f}"
            )
            max_rhat = (
                "n/a" if checkpoint.convergence is None
                else f"{checkpoint.convergence.max_rhat:.3f}"
            )
            print(
                f"  step {checkpoint.step}/{checkpoint.total_steps} "
                f"[{checkpoint.phase}] acceptance {rate}, R-hat {max_rhat}",
                flush=True,
            )
    diagnostics = sampler.diagnostics
    report = {
        "prior": name,
        "prior_exponent": PRIOR_EXPONENT,
        "seed": SEED,
        "weights": (
            "none (prior-only baseline)"
            if log_likelihood is zero_log_likelihood
            else {"chi2_rg": WEIGHT_RG, "chi2_pre": WEIGHT_PRE}
        ),
        "n_walkers": N_WALKERS,
        "total_steps": TOTAL_STEPS,
        "warmup": WARMUP,
        "thin": THIN,
        "converged": sampler.converged,
        "acceptance_rate": (
            None if checkpoint.acceptance_rate is None
            else float(checkpoint.acceptance_rate)
        ),
        "max_rhat": None if diagnostics is None else diagnostics.max_rhat,
        "min_ess": None if diagnostics is None else diagnostics.min_ess,
        "caveat": (
            "Preliminary: the 80/20 split validates interpolation along "
            "optimization paths, not accuracy in unexplored parameter space."
        ),
    }
    chain = sampler.chain.detach().cpu().numpy()
    log_prob_chain = sampler.chain_logp.detach().cpu().numpy()
    np.savez_compressed(output / "chain.npz", chain=chain, log_prob=log_prob_chain)
    summary = summarize(chain, log_prob_chain, prior.parameter_names)
    (output / "posterior_summary.json").write_text(
        json.dumps({"report": report, "parameters": summary}, indent=2) + "\n"
    )
    return chain, summary


def optimized_lambdas(parameter_names):
    """The CALVADOS2-optimized stickiness vector, in ``parameter_names`` order.

    The ``lambdas`` column of the upstream ``residues.csv`` holds the final
    parameters after optimization cycle 11 of run M1_24.
    """
    import io

    import pandas as pd

    from examples.calvados.fetch_data import RAW, verified_bytes

    residues = pd.read_csv(io.BytesIO(verified_bytes(RAW / "residues.csv")))
    if residues["three"].duplicated().any():
        raise ValueError("CALVADOS posterior: duplicate residue IDs")
    lambdas = residues.set_index("three")["lambdas"]
    return np.array(
        [float(lambdas[name.split()[1]]) for name in parameter_names]
    )


def prior_marginal(prior, column, grid):
    """Marginal prior density of one parameter evaluated on ``grid``."""
    if prior.kind == "uniform":
        return np.ones_like(grid)
    if prior.kind == "kde":
        centers = prior.centers[:, column]
        return (
            scipy.stats.norm.pdf(grid[:, None], loc=centers, scale=prior.bandwidth)
            / prior.component_masses[:, column]
            * prior.component_weights
        ).sum(axis=1)
    return scipy.stats.beta.pdf(
        grid, prior.alpha[column], prior.beta[column]
    )


def tempered_marginal(prior, column, grid, exponent):
    """Marginal of the tempered prior ``p**exponent`` on one parameter.

    Uniform stays flat. Each tempered beta column is exactly
    ``Beta(exponent*(alpha-1)+1, exponent*(beta-1)+1)``. The tempered KDE
    is approximated by inflating each kernel bandwidth to
    ``bandwidth / sqrt(exponent)`` and renormalizing the per-column masses;
    exact for a single dominant kernel, approximate where kernels overlap.
    """
    if prior.kind == "uniform":
        return np.ones_like(grid)
    if prior.kind == "beta":
        alpha = exponent * (prior.alpha[column] - 1) + 1
        beta_shape = exponent * (prior.beta[column] - 1) + 1
        return scipy.stats.beta.pdf(grid, alpha, beta_shape)
    bandwidth = prior.bandwidth / math.sqrt(exponent)
    centers = prior.centers[:, column]
    masses = scipy.stats.norm.cdf((1 - centers) / bandwidth) - scipy.stats.norm.cdf(
        -centers / bandwidth
    )
    weights = masses / masses.sum()
    return (
        weights
        * scipy.stats.norm.pdf(grid[:, None], loc=centers, scale=bandwidth)
    ).sum(axis=1)


def plot_marginals(chains, priors, parameter_names, optimized, maps=None,
                   output=OUTPUT, filename="posterior_marginals.png"):
    """Posterior histograms per parameter with the prior, tempered prior,
    CALVADOS2-optimized values and (post-run) MAP values overlaid."""
    n_priors = len(chains)
    figure, axes = plt.subplots(
        n_priors,
        len(parameter_names),
        figsize=(3 * len(parameter_names), 3 * n_priors),
        sharex="col",
        squeeze=False,
    )
    grid = np.linspace(0.001, 0.999, 400)
    for row, (name, chain) in enumerate(chains.items()):
        flat = chain.reshape(-1, chain.shape[-1])
        prior = priors[name]
        for column, axis in enumerate(axes[row]):
            axis.hist(flat[:, column], bins=40, range=(0, 1), density=True,
                      alpha=0.5, color="C0", label="Posterior")
            axis.plot(
                grid,
                prior_marginal(prior, column, grid),
                color="C1",
                linewidth=1.5,
                label="Prior",
            )
            axis.plot(
                grid,
                tempered_marginal(prior, column, grid, PRIOR_EXPONENT),
                color="C3",
                linewidth=1.2,
                linestyle=":",
                label=f"Tempered prior (x^{PRIOR_EXPONENT:g})",
            )
            axis.axvline(
                optimized[column], color="k", linestyle="--", linewidth=1,
                label="CALVADOS2 optimum",
            )
            if maps is not None and name in maps:
                axis.axvline(
                    maps[name][column], color="C2", linestyle="-.",
                    linewidth=1.2, label="MAP (this posterior)",
                )
            if row == 0:
                axis.set_title(parameter_names[column], fontsize=8)
            if row == n_priors - 1:
                axis.set_xlabel("lambda")
            if column == 0:
                axis.set_ylabel(f"{name}\n density", fontsize=9)
            axis.set_xlim(0, 1)
    axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / filename, dpi=150)
    plt.close(figure)


def main(plot_only=False, no_likelihood=False, output=None):
    plt.switch_backend("Agg")
    torch.set_num_threads(2)
    device = resolve_device("auto")
    priors = load_priors()
    models = {
        qoi: LGPCommittee.load(MODELS / f"{qoi}.lgp").to(device, torch.float32)
        for qoi in ("chi2_rg", "chi2_pre")
    }
    parameter_names = models["chi2_rg"].parameter_names
    for prior in priors.values():
        if tuple(prior.parameter_names) != tuple(parameter_names):
            raise ValueError(
                "CALVADOS posterior: prior parameters must match the GP order"
            )
    optimized = optimized_lambdas(parameter_names)
    if output is None:
        output = OUTPUT / "prior_only_baseline" if no_likelihood else OUTPUT
    global OUTPUT_DIR
    OUTPUT_DIR = output
    if plot_only:
        chains = {}
        maps = {}
        for name in PRIOR_NAMES:
            saved = np.load(output / name / "chain.npz")
            chains[name] = saved["chain"]
            map_file = output / name / "map.txt"
            if map_file.is_file():
                maps[name] = np.loadtxt(map_file)
        map_log_file = output / "map_log_target.json"
        map_log = (
            json.loads(map_log_file.read_text())
            if map_log_file.is_file() and maps
            else {}
        )
        print(f"Replotting saved chains: {output}")
    else:
        log_likelihood = (
            zero_log_likelihood if no_likelihood
            else discrepancy_log_likelihood(
                models, {"chi2_rg": WEIGHT_RG, "chi2_pre": WEIGHT_PRE}
            )
        )
        seeds = (
            None if no_likelihood
            else models["chi2_rg"].members[0].X_train.to(device, torch.float32)
        )
        print(
            f"Sampling {len(PRIOR_NAMES)} "
            f"{'tempered priors (no likelihood)' if no_likelihood else 'posteriors'} "
            f"on {device}",
            flush=True,
        )
        chains = {}
        maps = {}
        map_log = {}
        output.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(SEED)
        for name in PRIOR_NAMES:
            print(f"Prior: {name}", flush=True)
            per_prior = output / name
            per_prior.mkdir(exist_ok=True)
            chains[name] = run_prior(
                name, priors[name], models, log_likelihood, device, per_prior,
                seeds,
            )[0]
            if not no_likelihood:
                box = StructuralBox(seeds)
                chain = np.load(per_prior / "chain.npz")
                flat = chain["chain"].reshape(-1, 20)
                flat_logp = chain["log_prob"].reshape(-1)
                finite = np.isfinite(flat_logp)
                top_samples = [
                    torch.tensor(flat[i], dtype=torch.float32)
                    for i in np.argsort(np.where(finite, flat_logp, -np.inf))[-5:]
                ]
                prior_starts = initial_walkers(
                    priors[name], 8, generator, seeds, box
                )
                starts = [prior_starts[i] for i in range(8)] + list(top_samples)
                if box.is_valid(
                    torch.tensor(optimized, dtype=torch.float32).unsqueeze(0)
                )[0]:
                    starts.append(
                        torch.tensor(optimized, dtype=torch.float32)
                    )
                theta_map, value = find_map(
                    priors[name], log_likelihood, starts, device, box=box
                )
                maps[name] = theta_map
                map_log[name] = float(value)
                print(f"  MAP log target {value:.2f}", flush=True)
    plot_marginals(chains, priors, parameter_names, optimized,
                   maps=maps if not no_likelihood else None, output=output)
    if maps:
        for name, theta_map in maps.items():
            np.savetxt(output / name / "map.txt", theta_map)
        (output / "map_log_target.json").write_text(
            json.dumps(map_log, indent=2) + "\n"
        )
    print(f"Samples and summaries: {output}")


if __name__ == "__main__":
    main(
        plot_only="--plot" in sys.argv,
        no_likelihood="--no-likelihood" in sys.argv,
    )
