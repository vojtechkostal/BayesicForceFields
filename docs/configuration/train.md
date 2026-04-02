# Train Configuration

Source code:

- `bff/workflows/configs.py`
- `bff/workflows/train.py`
- `bff/bayes/learning.py`

## Purpose

`bff train` fits one surrogate model per QoI dataset and writes one `.lgp` file
per QoI.

## Minimal Example

```yaml
log: out.log
datasets:
  rdf:
    data: ../04-qoi/qoi-rdf.pt
    mean: sigmoid
training:
  model_dir: ./models
  reuse_models: true
  committee_size: 1
  device: cuda
```

## Top-Level Keys

- `log`
  Workflow log file.
- `datasets`
  Non-empty mapping from QoI name to dataset settings.
- `training`
  Surrogate-training settings.

## `datasets.<name>` Keys

- `data`
  QoI dataset file produced by `bff qoi`.
- `mean`
  Mean-function setting passed into surrogate training.
- `nuisance`
  Optional fixed nuisance standard deviation.
- `observation_scale`
  Optional scaling applied to the effective observation count in the likelihood.
- `model`
  Optional model output path. Defaults to `training.model_dir/<name>.lgp`.

## `training` Keys

- `model_dir`
  Output directory for surrogate model files.
- `reuse_models`
  Reuse existing model files if present.
- `n_hyper_max`
  Maximum number of hyperparameter optimization steps.
- `committee_size`
  Number of committee members for model averaging.
- `test_fraction`
  Held-out validation fraction.
- `device`
  Torch device, for example `cpu` or `cuda`.
- any additional keys
  Forwarded as optimization keyword arguments.

## Outputs

`bff train` writes one `.lgp` file per QoI.

Each model file stores:

- the trained committee
- the effective observation count used in the likelihood
- the reference observation vector needed later by `bff learn`
