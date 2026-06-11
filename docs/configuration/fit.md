# Fit Configuration

Source code:

- `bff/workflows/fit/config.py`
- `bff/workflows/fit/main.py`
- `bff/bayes/learning.py`

## Purpose

`bff fit` trains one surrogate model per QoI dataset and writes one `.lgp` file
per QoI.

## Minimal Example

```yaml
log: ./out.log
datasets:
  rdf:
    data: ../04-analyze/qoi-rdf.pt
    mean: sigmoid
fit:
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
- `fit`
  Surrogate-fitting settings.

## `datasets.<name>` Keys

- `data`
  QoI dataset file produced by `bff analyze`.
- `mean`
  Mean-function setting passed into surrogate training. Use a numeric constant
  or `sigmoid` for RDF datasets. Python callers of `fit_surrogates()` may also
  pass a callable mean.
- `nuisance`
  Optional fixed nuisance standard deviation.
- `model`
  Optional model output path. Defaults to `fit.model_dir/<name>.lgp`.

## `fit` Keys

- `model_dir`
  Output directory for surrogate model files.
- `reuse_models`
  Reuse existing model files if present.
- `n_hyper_max`
  Maximum number of training rows used during hyperparameter optimization.
- `committee_size`
  Number of committee members for model averaging.
- `test_fraction`
  Held-out validation fraction.
- `device`
  Torch device, for example `cpu` or `cuda`.
- any additional keys
  Forwarded as optimization keyword arguments.

## Outputs

`bff fit` writes one `.lgp` file per QoI.
Each model file stores the trained committee together with the reference values,
and the number of reference curves. `bff learn` assigns the effective
observation count from its model configuration.

!!! warning
    Models written before BFF 0.3.0 do not contain the required curve metadata.
    Refit existing QoI datasets before using them with BFF 0.3.0.
