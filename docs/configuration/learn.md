# Learn Configuration

```yaml
specs: ../03-sample/specs.yaml
models:
  rdf:
    model_path: ../05-lgp/models/rdf.lgp
    tolerance: 0.1
  pmf:
    model_path: ../05-lgp/models/pmf.lgp
    n_eff: 5
mcmc:
  total_steps: 10000
  warmup: 2000
  thin: 1
  resume: false
  device: cuda
plots:
  max_corner_samples: 2000
  max_marginal_samples: 10000
  max_qoi_samples: 10000
  qoi_batch_size: 256
  plot_metadata:
    define VSA:
      xlabel: O-VS
      ylabel: angle [degree]
    define VSD:
      xlabel: C-O-VS
      ylabel: distance [nm]
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
outputs/
  specs.yaml
  prior.pt
  posterior.pt
  mcmc.ckpt
```

Existing owned files are rejected by default. `output.overwrite: true` removes
only the paths listed above. `mcmc.resume: true` requires a checkpoint,
regenerates posterior and plots, and appends a delimited run to `learn.log`.
Resume and overwrite cannot be combined.

The configured `specs.yaml` is copied unchanged into `outputs/`. The prior,
posterior, and checkpoint are written atomically. Resume requires the copied
specifications and validates
the specification fingerprint, ordered models and their hashes, target
settings, dimensions, walkers, warmup, thinning, prior family, and proposal.
All three artifacts and all three plots are mandatory for command success.
Marginal annotations show the modes of the plotted posterior KDEs directly
below their lower bounds.

Plot generation uses deterministic, evenly spaced subsets of the prepared
posterior. `max_corner_samples` limits the samples used by the corner-plot KDE,
`max_marginal_samples` limits the samples used by the standard marginal KDEs,
and `max_qoi_samples` limits both QoI likelihood evaluation and attribution.
Set `max_marginal_samples` to `-1` or `null` to use all posterior samples.
QoI likelihoods are evaluated in batches of at most `qoi_batch_size`;
if a CUDA allocation still fails, the batch size is halved automatically and
retried. These settings affect only plots, not the saved posterior.

Each parameter kind forms its own marginal-plot section, arranged from left to
right in specification order. Charges, sigmas, epsilons, and other same-kind
parameters share a panel, with at most five parameters per row. Larger groups
wrap internally and split evenly, so six charges are shown as two rows of three.
Defines occupy the next horizontal section, but every `define` receives an
independent subpanel and y-axis. Two defines are stacked vertically; four form a
2-by-2 grid; larger sets use a square-like grid capped at five subpanels per row.
Optional `plots.plot_metadata` entries replace parameter tick labels and set
the x- and y-axis labels of individual define panels. Metadata keys must
exactly match names in `specs.yaml`; each entry supports `xlabel` and `ylabel`.
