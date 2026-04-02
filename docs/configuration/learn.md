# Learn Configuration

Source code:

- `bff/workflows/configs.py`
- `bff/workflows/learn.py`
- `bff/bayes/learning.py`

## Purpose

`bff learn` runs posterior inference from previously trained surrogate models.
The reference observation vectors and effective observation counts are read
from the surrogate files themselves, not from the original QoI datasets.

## Minimal Example

```yaml
log: out.log
specs: ../03-training-trjs/trainset/specs.yaml
models:
  rdf: ../05-train-lgp/models/rdf.lgp
  hb: ../05-train-lgp/models/hb.lgp
mcmc:
  total_steps: 10000
  warmup: 2000
  checkpoint: mcmc-checkpoint.pt
  posterior: posterior.pt
  priors: priors.pt
  restart: false
  device: cuda
```

## Top-Level Keys

- `log`
  Workflow log file.
- `specs`
  Force-field specification file from the simulate stage.
- `models`
  Non-empty mapping from QoI name to trained `.lgp` model file.
- `mcmc`
  Posterior-sampling settings.

## `models` Keys

Each key under `models` is the QoI name that should appear in logs and plots.
Each value is a path to the corresponding trained `.lgp` file.

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
  If `true`, include the implicit charge in prepared posterior samples.
