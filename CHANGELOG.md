# Changelog

## Unreleased

## `0.3.0` - 2026-06-11

### Breaking Changes

- Moved effective-observation configuration from `bff fit` to each model under
  `bff learn`. Model entries now require `model_path` and exactly one strategy:
  explicit `n_eff`, `independent_observations: true`, or curve inference with
  `tolerance`.
- Changed `.lgp` serialization to store reference-curve metadata. Models
  written by earlier releases must be refitted before use with 0.3.0.

### Added

- Added prominence-based effective-observation estimates for smooth,
  curve-valued QoIs while retaining direct counts for independent scalar data.
- Added `qoi-marginals.pdf`, which colors posterior parameter marginals by the
  local contribution of each QoI and shows priors and parameter bounds.
- Added per-QoI Gaussian log-likelihood evaluation for post-processing thinned
  posterior samples.

### Changed

- Made the Gaussian likelihood insensitive to curve binning by applying
  `n_eff` to a mean squared residual rather than treating every bin as an
  independent observation.
- Allowed surrogate models for different QoIs to use different numbers of
  training rows, provided their parameter dimensions agree.
- Simplified the `bff` command entry point and shell completion while retaining
  subcommands such as `bff build`.
- Made build templates optional for systems that do not require molecule
  templates.
- Updated the packaged examples and documentation for effective observations
  and QoI-attributed posterior plots.
- Included `pytest` in the standard installation so packaged test and example
  checks are available without installing the developer extra.

### Fixed

- Fixed multiprocessing imports for analysis routines loaded from user Python
  files.

## `0.2.1` - 2026-06-01

### Fixed

- Restored Python 3.10 compatibility by replacing the Python 3.11-only
  `typing.Self` annotation in Gaussian-process model loading.

## `0.2.0` - 2026-06-01

The public reproduction snapshot for the published study is archived as
[`v0.0.1`](https://github.com/vojtechkostal/BayesicForceFields/tree/v0.0.1).
Use that tag for exact reproduction of the paper results. The current
`0.2.0` release is a substantial workflow refactor.

### Reference Points

The code history contains two useful paper-era reference points:

- Earlier paper-era branches parsed and modified GROMACS topologies through
  ParmEd.
- The later public `v0.0.1` reproduction snapshot had already removed ParmEd
  and used an intermediate `gmxtop` parser together with MDAnalysis.
- The current development line uses `gmxtopology` for GROMACS topology
  handling and MDAnalysis selections for flexible atom grouping.

### Publication Snippet

Relative to the paper-era implementation, Bayesic Force Fields has been
refactored into a staged, configuration-driven workflow for system preparation,
reference-data generation, force-field sampling, surrogate fitting, posterior
learning, and validation. The refactor replaces the external `emcee` sampling
backend with an in-package Torch MCMC implementation, moves topology handling
away from the earlier ParmEd-based path, and adds reusable quantity-of-interest
datasets, local and Slurm execution, hierarchical charge constraints, broader
GROMACS topology updates, and notebook-first examples for externally generated
data.

### Architecture Changes

| Area | Paper-Era Implementation | `0.2.0` |
| --- | --- | --- |
| Posterior sampling | `emcee.EnsembleSampler` with an `emcee` HDF backend | In-package Torch parallel Metropolis-Hastings sampler with an adaptive Gaussian proposal |
| MCMC diagnostics | Autocorrelation handling through `emcee` results | Checkpoint and restart support, rank-normalized split R-hat, integrated autocorrelation time, and effective sample size diagnostics |
| Topology handling | Earlier branches used ParmEd; `v0.0.1` used the intermediate `gmxtop` parser | `gmxtopology` topology model with MDAnalysis selections and complete-topology updates |
| Charge constraints | One molecule of interest, one total molecular charge, and one implicit charge parameter | Reconstructable residue- or system-level hierarchical constraints with separate implicit parameters and compatibility checks |
| Training data | Monolithic `TrainData` arrays and YAML sidecar files | Serialized `QoIDataset` objects that can also be assembled from external data |
| Learning results | `MCMCResults` and `InferenceResults` inside a broad `structures.py` module | Focused `LearningProblem`, `PosteriorResults`, priors, proposal, sampler, and convergence modules |
| Workflow API | Broad `initialize`, `runsims`, `analyze`, and `learn` commands | Focused `build`, `prepare-assets`, `evaluate-snapshots`, `sample`, `analyze`, `fit`, `learn`, and `validate` stages |

### Highlights

- Removed the runtime `emcee` dependency and added a Torch-native MCMC stack
  with adaptive proposals, checkpoints, restart support, and convergence
  diagnostics.
- Replaced the earlier ParmEd topology path with a GROMACS-native topology
  model and MDAnalysis selections.
- Split the former monolithic structures and inference code into focused
  domain, MCMC, learning, and results modules.
- Added reconstructable `specs.yaml` files and hierarchical residue- or
  system-level charge constraints with explicit compatibility checks.
- Added CP2K snapshot staging and collection, EXTXYZ output, PLUMED and Colvars
  bias inputs, and local or Slurm execution.
- Unified analyzed observables as reusable quantity-of-interest datasets and
  added notebook workflows for arbitrary tabular data and Neon RDF inference.
- Added callable Gaussian-process means, numerical-stability improvements, and
  more reliable posterior plotting and MAP selection.
- Added GROMACS function-9 dihedral force-constant sampling through labels such
  as `dihedraltype9_3_180`, updating every matching topology term.
- Expanded documentation, packaged examples, regression coverage, and release
  automation.

## `v0.0.1`: Publication Snapshot

Archived public snapshot for reproducing the results reported in:

Kostal, V.; Shanks, B. L.; Jungwirth, P.; Martinez-Seara, H.
*Bayesian Learning for Accurate and Robust Biomolecular Force Fields.*
J. Chem. Theory Comput. **2026**, 22 (5), 2652-2663.
[https://doi.org/10.1021/acs.jctc.5c02051](https://doi.org/10.1021/acs.jctc.5c02051)
