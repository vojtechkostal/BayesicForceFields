# Fit LGP Configuration

`bff fit-lgp` trains one local Gaussian-process committee for each QoI dataset.

```yaml
datasets:
  rdf:
    data: ../04-build-qoi-datasets/qoi/rdf.pt
    mean: sigmoid
fit:
  model_dir: ./models
  reuse_models: true
  n_hyper_max: 200
  committee_size: 1
  test_fraction: 0.2
```

## Options

General rules for all options are on the
[conventions page](index.md).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `datasets` | mapping | *required* | QoI name to dataset settings; see below. |
| `log` | path | `<fit.model_dir>/../fit-lgp.log` | Workflow log file. |

### `datasets.<name>`

`<name>` is the QoI name stored in the dataset file.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | path | *required* | `qoi/<name>.pt` written by `bff build-qoi-datasets`. |
| `mean` | `data`, number, `sigmoid`, or `file.py:function` | `data` | What the surrogate predicts away from the training samples; see [Surrogate mean](#surrogate-mean). |
| `nuisance` | number > 0 | none | Fixed observation-noise standard deviation used by `bff learn`; without it, `learn` samples the noise as a nuisance parameter. |
| `model` | path | `<fit.model_dir>/<name>.lgp` | Output model file. |

### `fit`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `fit.model_dir` | path | `./models` | Directory for model files. |
| `fit.reuse_models` | boolean | `true` | Reuse an existing model fitted to identical data with the same mean instead of refitting. |
| `fit.n_hyper_max` | integer >= 1 | `200` | Maximum training points used to fit the hyperparameters. |
| `fit.committee_size` | integer >= 1 | `1` | Local GPs per committee. |
| `fit.test_fraction` | number in (0, 1) | `0.2` | Fraction of samples held out to report the model's test error (symmetric mean absolute percentage error, sMAPE). |
| `fit.max_iter` | integer >= 1 | `500` | Maximum L-BFGS-B iterations of the hyperparameter search. |
| `fit.tol_grad` | number > 0 | `1e-4` | Convergence of the hyperparameter search: largest component of the gradient of the log posterior. |

Fitting always runs on the CPU in float64: it uses at most `n_hyper_max`
points, which is too small for a GPU to help. `bff learn` moves the models to
its device once, in float32.

## Surrogate Mean

Each committee member is a Gaussian process: it predicts `mean(X)` plus a
correction learned from the training samples, and far from the samples only
`mean(X)` remains. Choose the mean that is a sensible prediction where there
are no simulations:

| `mean` | Prediction away from the samples | Use it for |
| --- | --- | --- |
| `data` (default) | The average training output, separately for each output value. | Most QoIs. |
| a number, e.g. `0` | That constant for every output value. | QoIs with a known baseline. |
| `sigmoid` | Per RDF curve, a smooth step from 0 to 1 centred where the reference RDF first reaches 0.5. Needs RDF datasets (`bins` and `range` settings). | RDFs whose sampled region is small. |
| `path/to/file.py:function` or `module:function` | A custom, possibly parameter-dependent model. | Physics-informed means, such as an RDF from a pair potential. |

A custom mean is a function of the parameter vectors, a torch tensor of shape
`(n_samples, n_parameters)` in the order of `specs.yaml`, that returns the
outputs, shape `(n_samples, n_outputs)` or `(n_outputs,)`, preferably as a
torch tensor (it receives the model's device and precision):

```python
import torch

def linear_density(X: torch.Tensor) -> torch.Tensor:
    return 997.0 + 300.0 * (X[:, :1] - 0.3)
```

Paths are relative to this configuration file. The model stores only the
reference `path:function`, so the file must stay importable wherever the model
is used. The mean is built from the training samples only, never from the
held-out test samples.

## Hyperparameter Priors

By default, log-scale GP hyperpriors are centered on the data used for
hyperparameter fitting. Each kernel length scale follows the corresponding
input-column standard deviation, the kernel amplitude follows the residual
standard deviation after subtracting `mean`, and the noise variance starts at
10% of the residual variance. The priors remain broad so optimization can move away
from these empirical centers. This scaling is especially important for scalar
QoIs and for parameter dimensions expressed on very different numerical scales.

## Model Reuse

A model records an exact fingerprint of its QoI dataset and the `mean` it was
fitted with. With `reuse_models: true`, a model file is reused only if both
match; otherwise it is refitted and replaced, and the log says why. Means given
as Python objects through the Python API are always refitted.

## Model Contents

An `.lgp` file holds an `LGPCommittee`. Its attributes follow common
Gaussian-process notation: `members` (the `LocalGaussianProcess` objects, each
with `X_train`, `y_train`, `mean`, `lengthscales`, `amplitude`, and
`noise_variance`), `y_ref` (the reference outputs), `n_inputs`, `n_outputs`,
`n_curves`, `test_error` (sMAPE in percent), `parameter_names`, and
`mean_spec`. QoI datasets use the same names: `X`, `y`, and `y_ref`.
