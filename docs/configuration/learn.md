# Learn Configuration

```yaml
specs: ../03-sample/specs.yaml
models:
  rdf:
    model_path: ../05-lgpfit/models/rdf.lgp
    tolerance: 0.1
  pmf:
    model_path: ../05-lgpfit/models/pmf.lgp
    n_eff: 5
mcmc:
  total_steps: 10000
  warmup: 2000
  thin: 1
  resume: false
  device: cuda
output:
  directory: ./
  overwrite: false
```

Each model selects exactly one effective-observation mode: positive `n_eff`,
`independent_observations: true`, or a positive curve `tolerance`. MCMC options
also include `priors_disttype`, `progress_stride`, `n_walkers`, `rhat_tol`,
`ess_min`, and `include_implicit_charge`.

Learning exposes one output root and owns these fixed paths:

```text
learn.log
plots/
  marginals.pdf
  qoi-marginals.pdf
  corner.pdf
output/
  prior.pt
  posterior.pt
  mcmc.ckpt
```

Existing owned files are rejected by default. `output.overwrite: true` removes
only the paths listed above. `mcmc.resume: true` requires a checkpoint,
regenerates posterior and plots, and appends a delimited run to `learn.log`.
Resume and overwrite cannot be combined.

The prior, posterior, and checkpoint are written atomically. Resume validates
the specification fingerprint, ordered models and their hashes, target
settings, dimensions, walkers, warmup, thinning, prior family, and proposal.
All three artifacts and all three plots are mandatory for command success.
Marginal annotations show posterior means and are placed from rendered bounds.
