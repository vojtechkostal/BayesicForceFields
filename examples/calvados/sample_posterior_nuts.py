"""NUTS sampling (Pyro) of the CALVADOS stickiness posteriors, 4 chains.

Same tempered target as ``sample_posterior``: the paper's weighting of the
log prior (``theta = 0.05``) plus the direct-discrepancy likelihood

    log target(eta) = PRIOR_EXPONENT * log prior(sigmoid(eta))
                      - 0.5 * (w_rg * chi2_rg + w_pre * chi2_pre)(sigmoid(eta))
                      + log |d theta / d eta|          (logit Jacobian)

Sampling runs in the unconstrained space ``eta = logit(theta)`` so the hard
[0,1] walls become smooth tails that NUTS can handle. Each of the 4 chains
runs in its own process (spawn), with its own seed and its own prior-based
initialization: for the KDE the 70 components are split round-robin across
chains, so no chain starts in another's modes. Requires the artifacts of
``train_discrepancy`` and ``prepare_priors``:

```bash
python -m examples.calvados.sample_posterior_nuts
```

Results go to ``generated/posterior_nuts/<prior>/`` and the marginal plot
to ``generated/posterior_nuts/posterior_marginals.png``. Convergence uses
BFF's own walker-chain diagnostics on the (n_saved, n_chains, n_dim) chain,
plus, for the KDE, the per-mode occupancy against the weights the tempered
target actually prescribes.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.constraints as constraints

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

from bff.bayes.gaussian_process import LGPCommittee
from bff.mcmc.convergence import diagnose
from examples.calvados.prepare_priors import load_priors
from examples.calvados.sample_posterior import (
    MODELS,
    PRIOR_EXPONENT,
    PRIOR_NAMES,
    SEED,
    WEIGHT_PRE,
    WEIGHT_RG,
    discrepancy_log_likelihood,
    optimized_lambdas,
    plot_marginals,
)

OUTPUT = Path(__file__).resolve().parent / "generated" / "posterior_nuts"
N_CHAINS = 4
WARMUP_STEPS = 600
NUM_SAMPLES = 10000
MAX_TREE_DEPTH = 9


@dataclass(frozen=True)
class ChainRequest:
    """Everything a worker process needs to run one NUTS chain."""

    prior_name: str
    chain: int
    seed: int
    initial_theta: list[list[float]]
    warmup: int
    num_samples: int


def prior_start_theta(prior, chain, generator):
    """One start per chain, drawn from the prior; KDE modes split
    round-robin across chains so the inits are mutually far apart."""
    n_dim = len(prior.parameter_names)
    if prior.kind == "uniform":
        return torch.rand((n_dim,), generator=generator)
    if prior.kind == "kde":
        centers = torch.tensor(prior.centers, dtype=torch.float32)
        pool = centers[chain::N_CHAINS]
        start = pool[torch.randint(len(pool), (1,), generator=generator)[0]]
        return (start + torch.randn(n_dim, generator=generator) * prior.bandwidth).clamp(
            0.01, 0.99
        )
    draws = torch.distributions.Beta(
        torch.tensor(prior.alpha, dtype=torch.float32),
        torch.tensor(prior.beta, dtype=torch.float32),
    ).sample((64,))
    finite = torch.isfinite(prior.log_prob(draws))
    return draws[finite][chain % int(finite.sum())]


def run_chain(request: ChainRequest) -> dict:
    """One NUTS chain in its own process; returns saved thetas and log-densities."""
    torch.set_num_threads(2)
    pyro.set_rng_seed(request.seed)
    prior = load_priors()[request.prior_name]
    models = {
        qoi: LGPCommittee.load(MODELS / f"{qoi}.lgp") for qoi in ("chi2_rg", "chi2_pre")
    }
    log_likelihood = discrepancy_log_likelihood(
        models, {"chi2_rg": WEIGHT_RG, "chi2_pre": WEIGHT_PRE}
    )
    n_dim = len(prior.parameter_names)

    # A sampleable Normal "carrier" keeps Pyro's model introspection happy;
    # its log-density is subtracted in the factor, so the sampled joint is
    # exactly exp(tempered_target + jacobian).
    carrier = dist.Normal(
        torch.zeros(n_dim), torch.full((n_dim,), 10.0)
    ).to_event(1)

    def model():
        eta = pyro.sample("eta", carrier)
        theta = torch.sigmoid(eta).clamp(1e-6, 1 - 1e-6)
        log_jacobian = (torch.log(theta) + torch.log1p(-theta)).sum()
        target = (
            PRIOR_EXPONENT * prior.log_prob(theta.unsqueeze(0))[0]
            + log_likelihood(theta.unsqueeze(0))[0]
            + log_jacobian
        )
        pyro.factor(
            "tempered_posterior", target - carrier.log_prob(eta).sum()
        )

    initial_eta = torch.logit(torch.tensor(request.initial_theta, dtype=torch.float32))
    kernel = NUTS(
        model,
        max_tree_depth=MAX_TREE_DEPTH,
        target_accept_prob=0.8,
        init_strategy=pyro.infer.autoguide.initialization.init_to_value(
            values={"eta": initial_eta}
        ),
    )
    mcmc = MCMC(
        kernel,
        num_samples=request.num_samples,
        warmup_steps=request.warmup,
        disable_progbar=True,
    )
    mcmc.run()
    eta = mcmc.get_samples()["eta"]
    theta = torch.sigmoid(eta)
    with torch.inference_mode():
        log_density = (
            PRIOR_EXPONENT * prior.log_prob(theta)
            + log_likelihood(theta)
        )
    return {
        "theta": theta.numpy().tolist(),
        "log_density": log_density.numpy().tolist(),
    }


def kde_equilibrium_weights(kde, rng):
    """Analytic-importance check: tempered-prior draw reweighted by the
    likelihood; with the flat GP landscape these are ~uniform (1/70 each)."""
    from examples.calvados.sample_posterior import TemperedPrior

    models = {
        qoi: LGPCommittee.load(MODELS / f"{qoi}.lgp") for qoi in ("chi2_rg", "chi2_pre")
    }
    ll = discrepancy_log_likelihood(
        models, {"chi2_rg": WEIGHT_RG, "chi2_pre": WEIGHT_PRE}
    )
    draws_idx = rng.integers(0, len(kde.centers), 7000)
    scale = kde.bandwidth / np.sqrt(PRIOR_EXPONENT)
    draws = np.clip(
        kde.centers[draws_idx] + rng.normal(0, scale, (7000, len(kde.centers[0]))),
        0.01,
        0.99,
    )
    with torch.inference_mode():
        logw = ll(torch.tensor(draws, dtype=torch.float32)).numpy()
    weights = np.exp(logw - logw.max())
    return np.bincount(draws_idx, weights=weights, minlength=len(kde.centers)) / weights.sum()


def mode_occupancy(theta_flat, centers):
    """Fraction of samples nearest each KDE component center."""
    nearest = np.sqrt(
        ((theta_flat[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    ).argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(centers)).astype(float)
    return counts / counts.sum()


def summarize_and_save(prior_name, prior, chain_tensor, logp_tensor, report_extra, output):
    """BFF's per-dimension diagnostics on the (n_saved, n_chains, n) chain.

    A tempered target whose log-density is constant (tophat: flat prior,
    flat likelihood) makes the appended log-probability column non-varying,
    which BFF reports as NaN; the coordinate columns are diagnosed through
    a stand-in column in that case and the summary says so.
    """
    constant_logp = bool(
        logp_tensor.std() < 1e-12
    )
    diagnostic_logp = logp_tensor
    if constant_logp:
        diagnostic_logp = chain_tensor[:, :, 0]
    diagnostics = diagnose(chain_tensor, diagnostic_logp)
    flat = chain_tensor.reshape(-1, chain_tensor.shape[-1])
    flat_logp = logp_tensor.reshape(-1)
    map_row = flat[int(np.argmax(flat_logp))]
    lower, upper = np.percentile(flat, [5, 95], axis=0)
    summary = {}
    for column, name in enumerate(prior.parameter_names):
        values = flat[:, column]
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "map": float(map_row[column]),
            "interval_5_95": [float(lower[column]), float(upper[column])],
        }
    report = {
        "prior": prior_name,
        "prior_exponent": PRIOR_EXPONENT,
        "sampler": "pyro NUTS",
        "n_chains": N_CHAINS,
        "warmup_steps": WARMUP_STEPS,
        "num_samples": NUM_SAMPLES,
        "max_tree_depth": MAX_TREE_DEPTH,
        "seed": SEED,
        "weights": {"chi2_rg": WEIGHT_RG, "chi2_pre": WEIGHT_PRE},
        "converged": diagnostics.converged(1.01, 400),
        "max_rhat": diagnostics.max_rhat,
        "min_ess": diagnostics.min_ess,
        "constant_log_density": constant_logp,
        **report_extra,
        "caveat": (
            "Preliminary: the 80/20 split validates interpolation along "
            "optimization paths, not accuracy in unexplored parameter space."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "chain.npz", chain=chain_tensor.numpy(), log_prob=logp_tensor.numpy()
    )

    def plain(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    (output / "posterior_summary.json").write_text(
        json.dumps(
            {
                "report": report,
                "parameters": summary,
                "diagnostics": {
                    key: plain(value)
                    for key, value in diagnostics.to_dict().items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return report, flat


def main():
    plt.switch_backend("Agg")
    torch.set_num_threads(2)
    priors = load_priors()
    rng = np.random.default_rng(SEED)
    requests = []
    for prior_name in PRIOR_NAMES:
        prior = priors[prior_name]
        for chain in range(N_CHAINS):
            generator = torch.Generator().manual_seed(
                SEED + 11 * chain + 97 * PRIOR_NAMES.index(prior_name)
            )
            start = prior_start_theta(prior, chain, generator)
            requests.append(
                ChainRequest(
                    prior_name=prior_name,
                    chain=chain,
                    seed=SEED + 17 * chain,
                    initial_theta=start.numpy().tolist(),
                    warmup=WARMUP_STEPS,
                    num_samples=NUM_SAMPLES,
                )
            )
    print(
        f"Running {len(requests)} NUTS chains ({N_CHAINS} per prior, "
        f"parallel) with warmup {WARMUP_STEPS} + {NUM_SAMPLES} samples",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=N_CHAINS) as pool:
        results = list(pool.map(run_chain, requests))

    chains = {}
    reports = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = 0
    for prior_name in PRIOR_NAMES:
        block = results[index : index + N_CHAINS]
        index += N_CHAINS
        theta = torch.tensor(
            [r["theta"] for r in block], dtype=torch.float32
        ).transpose(0, 1)
        logp = torch.tensor(
            [r["log_density"] for r in block], dtype=torch.float32
        ).transpose(0, 1)
        chain_tensor = theta.detach().cpu()
        logp_tensor = logp.detach().cpu()
        extra = {}
        if prior_name == "kde":
            centers = priors["kde"].centers
            occupancy = mode_occupancy(chain_tensor.reshape(-1, 20).numpy(), centers)
            weights = kde_equilibrium_weights(priors["kde"], rng)
            ratio = occupancy / np.maximum(weights, 1e-12)
            extra["kde_modes_occupied"] = int((occupancy > 0).sum())
            extra["kde_equilibrium_ratio_p10_median_p90"] = [
                float(np.percentile(ratio, 10)),
                float(np.median(ratio)),
                float(np.percentile(ratio, 90)),
            ]
        output = OUTPUT / prior_name
        report, flat = summarize_and_save(
            prior_name, priors[prior_name], chain_tensor, logp_tensor, extra, output
        )
        chains[prior_name] = flat.numpy()
        reports[prior_name] = report
        print(
            f"{prior_name}: R-hat {report['max_rhat']:.3f}, "
            f"min ESS {report['min_ess']:.0f}, "
            f"converged {report['converged']}",
            flush=True,
        )
    parameter_names = list(priors[PRIOR_NAMES[0]].parameter_names)
    optimized = optimized_lambdas(parameter_names)
    plot_marginals(
        chains,
        priors,
        parameter_names,
        optimized,
        output=OUTPUT,
        filename="posterior_marginals.png",
    )
    for name, report in reports.items():
        if "kde_modes_occupied" in report:
            print(
                f"kde: {report['kde_modes_occupied']} modes occupied, "
                f"occupancy/equilibrium "
                f"{report['kde_equilibrium_ratio_p10_median_p90']}",
                flush=True,
            )
    print(f"NUTS posteriors: {OUTPUT}")


if __name__ == "__main__":
    main()
