# Learn Configuration

Source code:

- `bff/workflows/learn/config.py`
- `bff/workflows/learn/main.py`
- `bff/bayes/learning.py`

## Purpose

`bff learn` runs posterior learning from previously trained surrogate models.
The reference observation vectors are read from the surrogate files themselves.
Effective observation counts are assigned when the models are loaded for
learning.

## Minimal Example

```yaml
log: ./out.log
specs: ../03-sample/specs.yaml
models:
  rdf:
    model_path: ../05-fit/models/rdf.lgp
    tolerance: 0.1
  density:
    model_path: ../05-fit/models/density.lgp
    independent_observations: true
  pmf:
    model_path: ../05-fit/models/pmf.lgp
    n_eff: 5
mcmc:
  total_steps: 10000
  warmup: 2000
  checkpoint: ./mcmc-checkpoint.pt
  posterior: ./posterior.pt
  priors: ./priors.pt
  restart: false
  device: cuda
```

## Top-Level Keys

- `log`
  Workflow log file.
- `specs`
  Force-field specification file from the sample stage.
- `models`
  Non-empty mapping from QoI name to trained `.lgp` model settings.
- `mcmc`
  Posterior-sampling settings.

## `models` Keys

Each key under `models` is the QoI name that should appear in logs and plots.
Each value is a mapping with:

- `model_path`
  Path to the corresponding trained `.lgp` file.
- `independent_observations`
  If `true`, every reference value counts as one effective observation.
- `n_eff`
  Positive effective observation count to use directly. It cannot be combined
  with `independent_observations` or `tolerance`.
- `tolerance`
  Required when neither `independent_observations: true` nor `n_eff` is
  specified. The reference values are split into their stored curves and the
  effective count is estimated from each curve.

The Gaussian contribution for one QoI uses the mean squared curve residual:

```text
-0.5 * n_eff * MSE / sigma^2 - n_eff * log(sigma)
```

Using MSE keeps the likelihood insensitive to merely duplicating or refining
the curve bins. `n_eff` controls how many independently resolved observations
the complete curve represents.

!!! warning
    BFF 0.3.0 removes the path-only model syntax and cannot load pre-0.3.0
    `.lgp` files. Refit the surrogate models and use the nested model mappings
    shown above.

## `mcmc` Keys

- `priors_disttype`
  Prior family, currently defaulting to `normal`.
- `total_steps`
  Total MCMC steps.
- `warmup`
  Burn-in length.
- `thin`
  Chain thinning factor.
- `progress_stride`
  Logging interval.
- `n_walkers`
  Optional walker count. If omitted, BFF chooses a default.
- `checkpoint`
  Checkpoint file path.
- `posterior`
  Posterior chain output path.
- `priors`
  Prior output path.
- `restart`
  Restart from checkpoint if possible.
- `device`
  Torch device for MCMC.
- `rhat_tol`
  R-hat convergence threshold.
- `ess_min`
  Minimum effective sample size target.
- `include_implicit_charge`
  If `true`, include reconstructed implicit charges in prepared posterior samples.

## Plots

`bff learn` writes:

- `marginals.pdf`
  Prior and posterior parameter marginals.
- `qoi-marginals.pdf`
  Posterior marginals partitioned into contrastive QoI-attribution colors.
  The colored regions are diagnostic responsibilities, not an additive
  decomposition of posterior probability. The first legend row identifies the
  prior, total posterior, and parameter bounds; the second identifies QoIs.
- `corner.pdf`
  Joint posterior parameter and nuisance distributions.
