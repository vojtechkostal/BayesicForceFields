# Learn Configuration

```yaml
specs: ../03-sample-parameters/specs.yaml
models:
  rdf:
    model_path: ../05-fit-lgp/models/rdf.lgp
    tolerance: 0.1
  pmf:
    model_path: ../05-fit-lgp/models/pmf.lgp
    tolerance: 0.5
mcmc:
  total_steps: 10000
  warmup: 2000
  thin: 1
  resume: false
  device: auto
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

## Options

General rules for all options are on the
[conventions page](index.md).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `specs` | path | *required* | `specs.yaml` of the sampling campaign. |
| `models` | mapping | *required* | QoI name to surrogate model; see below. |

### `models.<name>`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | path | *required* | `.lgp` model written by `bff fit-lgp`. |
| `tolerance` | number >= 0 | `0` | Deviation from the reference that is acceptable, in the QoI's units, for example `0.1` for an RDF. See [Likelihood](#likelihood). |

## Likelihood

Each QoI contributes a Gaussian likelihood of the mean squared deviation
between the surrogate prediction and the reference:

```text
log L = -(n_eff / 2) * MSE / (sigma^2 + tolerance^2) - (n_eff / 2) * log(sigma^2 + tolerance^2)
```

- `sigma` is the noise of the data, learned over the whole QoI as a nuisance
  parameter (or fixed by `nuisance` in fit-lgp).
- `tolerance` is the deviation you accept: a model that stays within about
  `tolerance` of the reference is nearly as good as a perfect one. It matters
  when it is larger than the learned `sigma`, and leaves the result unchanged
  when it is much smaller.
- `n_eff` is the number of independent observations; BFF infers it. Values of
  a curve are not independent, because a model that deviates at one bin
  deviates at its neighbours too. BFF fits the correlation length `l` of each
  reference curve (a Gaussian process fitted to the curve) and counts
  `n_eff = (tr R)^2 / tr(R^2)` for the correlation `R` of deviations along the
  curve, about one observation per `sqrt(pi) * l` of curve. It does not depend
  on the binning. Scalar QoIs count one observation per value, and a curve of
  several labels adds up its curves.

The learn log reports each QoI's `n_eff` and tolerance.

### `mcmc`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `mcmc.total_steps` | integer >= 1 | `1500` | MCMC steps per walker. |
| `mcmc.warmup` | integer >= 0 | `500` | Discarded initial steps; smaller than `total_steps`. |
| `mcmc.thin` | integer >= 1 | `1` | Keep every this many steps after warmup. |
| `mcmc.n_walkers` | integer >= 2 | 5 x number of sampled parameters | Parallel walkers. |
| `mcmc.priors_disttype` | `normal` or `uniform` | `normal` | Prior family over the parameter bounds. |
| `mcmc.rhat_tol` | number > 1 | `1.01` | Convergence: sampling stops early once the largest rank-normalized split R-hat (over all parameters and the log probability) is below `rhat_tol` and the smallest bulk and tail effective sample size is at least `ess_min`, at two consecutive checks (made when the chain has grown by 10 %). |
| `mcmc.ess_min` | integer >= 1 | `400` | See `rhat_tol`. |
| `mcmc.progress_stride` | integer >= 1 | `100` | Steps between progress reports and checkpoints; convergence is diagnosed at these reports once the chain has grown by 10 %. |
| `mcmc.resume` | boolean | `false` | Continue from `outputs/mcmc.ckpt`; cannot be combined with `output.overwrite`. |
| `mcmc.device` | `auto`, `cpu`, `cuda`, `cuda:<index>`, or `mps` | `auto` | PyTorch device, chosen once for the whole run: `auto` is `cuda` when available and `cpu` otherwise. The surrogate models move to it at the start. |

### `plots`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `plots.max_corner_samples` | integer >= 1 | `2000` | Posterior samples used by the corner-plot KDE. |
| `plots.max_marginal_samples` | integer >= 1 or -1 | `10000` | Posterior samples used by the marginal KDEs; `-1` uses all. |
| `plots.max_qoi_samples` | integer >= 1 | `10000` | Posterior samples used for QoI likelihood attribution. |
| `plots.qoi_batch_size` | integer >= 1 | `256` | Batch size of QoI likelihood evaluation; halved automatically on CUDA out-of-memory. |
| `plots.plot_metadata.<parameter>.xlabel` | string | parameter name | Axis or tick label for a parameter in `specs.yaml`. |
| `plots.plot_metadata.<parameter>.ylabel` | string | `Value` | y-axis label of a `define` panel. |

### `output`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output.directory` | path | `./` | Root of `learn.log`, `plots/`, and `outputs/`. |
| `output.overwrite` | boolean | `false` | Replace existing outputs listed below instead of failing. |

## Outputs

Learning exposes one output root and owns these fixed paths:

```text
learn.log
plots/
  marginals.pdf
  qoi-marginals.pdf
  corner.pdf
outputs/
  specs.yaml
  results.pt
  mcmc.ckpt
```

Existing owned files are rejected by default. `output.overwrite: true` removes
only the paths listed above. `mcmc.resume: true` requires a checkpoint,
regenerates posterior and plots, and appends a delimited run to `learn.log`.
Resume and overwrite cannot be combined.

The configured `specs.yaml` is copied unchanged into `outputs/`. `results.pt`
holds everything about the run in one file: the prior and the posterior chain
side by side, the log posterior of every state (so the MAP), the
specification, each QoI's `n_eff` and tolerance, and the log likelihood of each
QoI at the samples used for the QoI plot; see [Results](../results.md). The
results and the checkpoint are written atomically. Resume requires the copied
specifications and validates
the specification fingerprint, ordered models and their hashes, target
settings, dimensions, walkers, warmup, thinning, prior family, and proposal.
The results, the checkpoint, the copied specifications, and all three plots are
mandatory for command success. Marginal plots mark the MAP with a diamond and
show the mode of each posterior density directly below its lower bound. The
learn log lists the posterior mean, standard deviation, and MAP of every
parameter.

Plots use every posterior sample after warmup, without discarding outliers, and
deterministic, evenly spaced subsets for the expensive parts.
`max_corner_samples` limits the samples used by the corner-plot KDE,
`max_marginal_samples` limits the samples used by the standard marginal KDEs,
and `max_qoi_samples` is the number of samples whose per-QoI log likelihood is
evaluated and stored for the QoI plot. Set `max_marginal_samples` to `-1` to use
all posterior samples. QoI likelihoods are evaluated in batches of at most
`qoi_batch_size`; if a CUDA allocation still fails, the batch size is halved
automatically and retried. Except for the stored QoI likelihoods, these settings
affect only plots, not the saved results.

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
