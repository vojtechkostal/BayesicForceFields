# Changelog

The public reproduction snapshot for the published study is archived as
[`v0.0.1`](https://github.com/vojtechkostal/BayesicForceFields/tree/v0.0.1).
Use that tag for exact reproduction of the paper results. The current
`0.3.0` release is the current workflow release.

## `0.3.0` - 2026-06-11

### Highlights

- Effective observation counts are now configured during `bff learn` as
  explicit counts, independent scalar observations, or tolerance-derived curve
  features.
- The Gaussian likelihood uses an `n_eff`-weighted mean squared residual, so
  posterior width is not determined by arbitrary curve binning.
- `bff learn` writes `qoi-marginals.pdf` to show which QoIs support different
  posterior regions.
- Different QoIs may use different numbers of surrogate-training rows.
- Build templates are optional when a system does not need them, and the CLI
  uses the single `bff <command>` entry point.
- User-defined analysis routines now work with multiprocessing workers.
- `pytest` is included in the standard installation.

!!! warning
    The learn configuration and `.lgp` model format changed in 0.3.0. Refit
    models created by earlier releases and migrate path-only model entries to
    the nested format documented in the
    [learn configuration](configuration/learn.md).

## `0.2.1` - 2026-06-01

### Fixed

- Restored Python 3.10 compatibility by replacing the Python 3.11-only
  `typing.Self` annotation in Gaussian-process model loading.

## `0.2.0` - 2026-06-01

## Reference Points

The code history contains two useful paper-era comparison points:

- Earlier branches used ParmEd to parse and modify GROMACS topologies.
- The later public `v0.0.1` snapshot had already moved to an intermediate
  `gmxtop` parser together with MDAnalysis.
- The current line uses `gmxtopology` and MDAnalysis selections.

## Publication Snippet

Relative to the paper-era implementation, Bayesic Force Fields has been
refactored into a staged, configuration-driven workflow for system preparation,
reference-data generation, force-field sampling, surrogate fitting, posterior
learning, and validation. The refactor replaces the external `emcee` sampling
backend with an in-package Torch MCMC implementation, moves topology handling
away from the earlier ParmEd-based path, and adds reusable quantity-of-interest
datasets, local and Slurm execution, hierarchical charge constraints, broader
GROMACS topology updates, and notebook-first examples for externally generated
data.

## Architecture Changes

| Area | Paper-Era Implementation | `0.2.0` |
| --- | --- | --- |
| Posterior sampling | `emcee.EnsembleSampler` and its HDF backend | Torch parallel Metropolis-Hastings sampler with adaptive proposals |
| Diagnostics | Autocorrelation handling through `emcee` results | Checkpoints, restart support, split R-hat, autocorrelation time, and effective sample size |
| Topologies | ParmEd in earlier branches; intermediate `gmxtop` in `v0.0.1` | `gmxtopology` with MDAnalysis selections |
| Constraints | One molecular total charge and one implicit charge parameter | Hierarchical residue- or system-level constraints with compatibility checks |
| Data model | Monolithic training arrays and YAML sidecars | Reusable serialized `QoIDataset` objects |
| API | Broad workflow commands | Focused stages from `build` through `validate` |

## Highlights

- Removed the runtime `emcee` dependency in favor of the Torch MCMC stack.
- Replaced the earlier ParmEd topology path with GROMACS-native handling.
- Split monolithic structures and inference code into focused modules.
- Reconstructable `specs.yaml` files and hierarchical charge constraints.
- CP2K snapshot collection, bias inputs, and local or Slurm campaigns.
- Reusable quantity-of-interest datasets and notebook-first examples.
- More stable Gaussian-process fitting, posterior learning, and plotting.
- Function-9 GROMACS dihedral updates using labels such as
  `dihedraltype9_3_180`.

The repository [CHANGELOG.md](https://github.com/vojtechkostal/BayesicForceFields/blob/main/CHANGELOG.md)
keeps the complete grouped history.
