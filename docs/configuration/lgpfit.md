# LGP Fit Configuration

`bff lgpfit` trains one local Gaussian-process committee for each analyzed QoI.
There is no `bff fit` compatibility command.

```yaml
datasets:
  rdf:
    data: ../04-analyze/qoi/rdf.pt
    mean: sigmoid
lgpfit:
  model_dir: ./models
  reuse_models: true
  n_hyper_max: 200
  committee_size: 1
  test_fraction: 0.2
  device: cuda
```

`datasets.<name>.data` must contain a dataset with the same name. Optional
dataset keys are `mean`, `nuisance`, and `model`; the model path defaults to
`lgpfit.model_dir/<name>.lgp`. Optimizer overrides are `lr`, `max_iter`, and
`tol_grad`. Unknown keys are rejected.

The default log is `lgpfit.log` next to `models/`. Cached models include an
exact fingerprint of the QoI dataset. Reuse is rejected if any training input,
output, reference value, label, setting, or metadata differs, even when tensor
shapes are unchanged.
