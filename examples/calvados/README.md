# CALVADOS2 surrogate training data

## Aggregate-discrepancy surrogates

From the repository root, in an environment with BFF, PyTorch and pandas:

```bash
python examples/calvados/fetch_data.py
python -m examples.calvados.train_discrepancy
```

This fits two BFF scalar GPs directly to `chi2_rg` and `chi2_pre`
from all 11 M1_24 optimization histories: 1,318 distinct parameter vectors,
each with 20 residue stickiness inputs. The regularized `cost` and prior term
are not training targets. Parameter columns are matched by residue ID.

Both fits use seed 2026 and the same random 80/20 split: 1,055 training rows
and 263 test rows. BFF optimizes GP hyperparameters using 200 training rows,
then constructs each GP using all 1,055 training rows. The held-out rows are
never used for fitting. Training uses two CPU threads and one committee member.

`generated/discrepancy_direct/` contains the datasets, `.lgp` models, fit logs,
`split.csv` (cycle and iteration provenance), individual test predictions,
`metrics.json`, and `held_out_predictions.png`. Metrics include RMSE, MAE,
R², and a training-mean baseline, all on the original chi-squared scale.
Earlier square-root fits, if present, remain in `generated/discrepancy/`.
The 90/10 comparison remains in `generated/discrepancy_direct_90_10/`.

To evaluate those saved 80/20 models after excluding test samples with any
stickiness parameter at or below 0.005 or at or above 0.995:

```bash
python -m examples.calvados.evaluate_interior_test
```

This requires the saved 80/20 models, datasets and split manifest. The 1,055
training samples remain unchanged. Results and the exclusion audit are written to
`generated/discrepancy_direct_80_20_interior_005/`. The earlier 0.01-threshold
evaluation remains in `generated/discrepancy_direct_80_20_interior/`.
These scores describe only
this small interior subset, not the complete held-out distribution.

This random split measures interpolation along correlated optimization paths;
it does not establish accuracy in unexplored parameter space. A cycle- or
run-held-out assessment is needed before relying on posterior inference.
The zero reference in these datasets is only a placeholder for zero discrepancy:
likelihood scaling, prior choice and uncertainty treatment are not configured,
and the models must not be passed directly to `learn` without those decisions.
Direct chi-squared predictions need a discrepancy-based log likelihood;
BFF's ordinary observable-versus-reference Gaussian likelihood would square
the discrepancy again.
No posterior sampling or MD is performed.

## Three alternative priors

Keep the original, unfiltered 80/20 GPs. Prepare priors in exactly their
20-parameter column order with:

```bash
python -m examples.calvados.prepare_priors
```

The checksum-verified `selHPS.csv` supplies 70 min-max normalized scales.
The alternatives are:

- **Tophat:** independent uniform densities on [0, 1], joint density 1 on
  the 20-dimensional cube and zero outside.
- **KDE:** an equal-weight mixture of 70 joint Gaussian kernels, centered on
  the scale vectors, with isotropic bandwidth 0.05 as in the paper. This
  retains cross-residue dependence. The whole mixture is conditioned on
  [0, 1]^20 and normalized, rather than truncating and equally reweighting
  each kernel separately. Inside the cube its log density differs from the
  paper's KDE only by an additive constant, irrelevant to MCMC ratios.
- **Independent beta:** a product of 20 beta densities, fitted separately
  by matching each residue's scale mean and population variance. If mean is
  m and variance is v, c = m(1-m)/v - 1, alpha = mc, beta = (1-m)c.
  Exact 0 and 1 entries remain in the moment calculation; no clipping is
  needed. This intentionally discards cross-residue correlations.

All priors use exponent 1: the paper's 0.05 weighting of log prior is not
automatically applied. Beta densities are evaluated on the open interval
(0, 1); endpoint values do not affect this continuous probability model.

`generated/priors/` contains `priors.json` (complete reproducible definitions
and provenance), `beta_parameters.csv`, and `prior_marginals.png`. The plotted
KDE curves are marginals of the bounded joint mixture, not separately fitted
one-dimensional KDEs.

```python
from examples.calvados.prepare_priors import load_priors

priors = load_priors()
log_prior = priors["kde"].log_prob(theta)
```

`theta` is a floating-point torch tensor with 20 columns in the recorded
`parameter_names` order. Each object supplies a batch `log_prob` compatible
with `bff.bayes.posterior.log_posterior`. These are example-specific priors;
the `bff learn` YAML prior selector still supports only its existing families.
No posterior is sampled here. The direct-discrepancy likelihood and its Rg/PRE
weights remain to be specified before MCMC.

## Original per-protein snapshots

From the repository root, using an environment with BFF, PyTorch and pandas:

```bash
python examples/calvados/fetch_data.py
python -m examples.calvados.convert_data
```

The downloader uses only the Python standard library. Conversion additionally
requires pandas to read the upstream snapshots (`pip install pandas` if needed).
Paths resolve relative to the scripts, independently of the working directory.

`raw/M1_24/` holds unchanged, checksum-verified upstream files. Conversion writes
one BFF `QoIDataset` per protein to `generated/qoi/`, plus `generated/summary.json`.
Each dataset has 11 rows, 20 parameter columns sorted by amino-acid one-letter
code, and one Rg output. The experimental targets come from the upstream
snapshots. Your `c2_rg.csv` remains separate.

Downloaded data and generated outputs are ignored by Git; scripts and provenance
are versionable. See [SOURCE.md](SOURCE.md) for the fixed commit, cycle mapping,
units, and limitations.

The per-protein import is not a validated inference example. Eleven
parameter sets cannot broadly cover 20 parameter dimensions; the centered
training matrix has rank at most ten. The 51 proteins provide separate outputs,
not additional parameter-space locations. No surrogate fitting, posterior
sampling, or new MD is performed. Additional compatible parameter sets or a
scientifically justified reduction of parameter dimension are needed before
reliable inference. Different-cutoff runs must not be pooled as the same model.
