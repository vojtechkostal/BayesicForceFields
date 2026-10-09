# Results

`bff learn` writes one file, `outputs/results.pt`. It holds everything needed to
inspect, summarize, plot, and reuse a run, so a notebook needs nothing else:

```python
import bff

results = bff.Results.load("06-learn/outputs/results.pt")

results.summary()          # mean, std, median, 16/84 % quantiles, mode, MAP per parameter
results.map                # maximum a posteriori parameters
results["charge C1"]       # posterior samples of one parameter
results.draw(10)           # parameter sets from an approximate posterior
results.plot_marginals()   # figures, see below
```

## What it contains

| Attribute | Meaning |
| --- | --- |
| `results.names` | Names of the columns of `results.samples`: every parameter of the specification (sampled parameters and implicit charges, sorted by name), then `noise <qoi>` for each QoI whose noise was learned. |
| `results.samples` | Posterior samples in physical units, shape `(n_samples, len(names))`: all states after warmup, implicit charges computed, noise as sigma (not its logarithm). |
| `results[name]` | One column of `samples`. |
| `results.map` | The MAP: the saved state of highest log posterior, as a mapping from every name to its value. |
| `results.prior` | The priors of the sampled columns, next to the posterior rather than inside it. |
| `results.specs` | The parameter `Specs` of the run. |
| `results.chain`, `results.log_prob` | The raw chain `(n_saved, n_walkers, n_dim)` in sampler space (noise as its logarithm) and the log posterior of each state. |
| `results.qoi` | Per QoI: `n_eff`, `tolerance`, and the fixed `nuisance` if there is one. |
| `results.qoi_index`, `results.qoi_log_likelihood` | Samples (indices into the flattened chain) at which each QoI's log likelihood was evaluated, and those values; they feed `plot_qoi_marginals`. |
| `results.info` | MCMC settings, whether it converged, specification and model fingerprints. |

`results.explicit_names`, `results.implicit_names`, and `results.nuisance_names`
split `names` by role.

## Point estimates and diagnostics

`results.summary()` returns, for every name, the `mean`, `std`, `median`, the
`q16` and `q84` quantiles, the `mode` of the marginal density, and the `map`.
`results.write_summary("summary.yaml")` saves it. The MAP is the best single
state the sampler visited, including noise; it can differ from the marginal
modes in correlated posteriors.

`results.diagnostics()` returns the rank-normalized split R-hat, the bulk ESS,
and the tail ESS (Vehtari et al., 2021) of every sampled column and of the log
probability. These are the numbers the sampler stops on: `bff learn` ends early
when the largest R-hat is below `mcmc.rhat_tol` and the smallest ESS is at least
`mcmc.ess_min` at two consecutive checks (spaced by 10 % chain growth, so the cost stays a fixed fraction of the sampling). `results.info["mcmc"]` keeps the final
`max_rhat` and `min_ess`.

## Drawing parameter sets

```python
draws = results.draw(
    10, distribution="normal", seed=1, include_mean=True, include_map=True
)
results.draw(10, fn_out="posterior-samples.yaml")   # explicit-mode validation input
```

`draw` fits an approximation to the posterior samples of the sampled parameters
(`normal`, `kde`, `uniform`, or `empirical`), redraws sets whose implicit charges
leave their bounds, and prepends the mean and the MAP if requested. It returns a
mapping from parameter name to values; `implicit=True` adds the implicit charges.
`bff validate` uses the same function.

## Figures

Every plot function takes the results and returns the matplotlib `Figure`; save it
with `fig.savefig("marginals.pdf", bbox_inches="tight")`. The same functions are
methods of `Results`.

| Function | Shows |
| --- | --- |
| `plot_marginals(results, plot_metadata=None, max_samples=None)` | Prior and posterior marginal of every parameter with its bounds, mode, and MAP. |
| `plot_qoi_marginals(results, plot_metadata=None, temperature=0.7)` | The posterior marginals colored by the QoI that supports each region (from the stored likelihoods). |
| `plot_corner(results, names=None, max_samples=None)` | Corner plot of the chosen columns, noise included. |

`plot_metadata` maps parameter names to `{"xlabel": ..., "ylabel": ...}`, as in the
`plots.plot_metadata` option of the [learn configuration](configuration/learn.md).

## Building results yourself

`LearningProblem.learn(...)` returns a `Results` and writes `fn_results`. To wrap
your own chain, construct it directly:

```python
results = bff.Results(chain, log_prob, specs, prior=prior, nuisances=["rdf"])
results.save("results.pt")
```

Results of earlier BFF versions (`posterior.pt` and `prior.pt`) cannot be read;
rerun `bff learn`.
