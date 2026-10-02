# Fit LGP Configuration

`bff fit-lgp` trains one local Gaussian-process committee for each QoI dataset.

```yaml
datasets:
  rdf:
    data: ../04-qoi/qoi/rdf.pt
    mean: sigmoid
fit:
  model_dir: ./models
  reuse_models: true
  n_hyper_max: 200
  committee_size: 1
  test_fraction: 0.2
  device: cuda
```

`datasets.<name>.data` must contain a dataset with the same name. Optional
dataset keys are `mean`, `nuisance`, and `model`; the model path defaults to
`fit.model_dir/<name>.lgp`. Optimizer overrides are `lr`, `max_iter`, and
`tol_grad`. Unknown keys are rejected.

By default, log-scale GP hyperpriors are centered on the data used for
hyperparameter fitting. Each kernel length scale follows the corresponding
input-column standard deviation, the kernel width follows the residual target
standard deviation after subtracting `mean`, and diagonal noise starts at 10%
of the residual variance. The priors remain broad so optimization can move away
from these empirical centers. This scaling is especially important for scalar
QoIs and for parameter dimensions expressed on very different numerical scales.

The default log is `fit-lgp.log` next to `models/`. Cached models include an
exact fingerprint of the QoI dataset. Reuse is rejected if any training input,
output, reference value, label, setting, or metadata differs, even when tensor
shapes are unchanged.
