# Changelog

The public reproduction snapshot for the published study is archived as
[`v0.0.1`](https://github.com/vojtechkostal/BayesicForceFields/tree/v0.0.1).
Use that tag for exact reproduction of the paper results. The current workflow
release is `0.4.2`.

## Unreleased

### Changed

- Simulation campaigns can run GROMACS in a node-local `scratch_dir` and copy
  results back, and on Slurm stop production runs cleanly before
  `sbatch.time` (`mdrun -maxh`). With `max_restarts`, samples stopped by the
  time limit are resubmitted and continue from their checkpoints.
- The production step count is written into a generated
  `production-run.mdp` (which replaces `production-colvars.mdp`) instead of
  being passed as `mdrun -nsteps`.
- Built-in `rdf` and `hydrogen_bonds` routines now use the same interface as
  custom routines and validate their own options when they run. Their outputs
  are unchanged; hydrogen-bond analysis is substantially faster.
- Moved `bff.qoi.data` to `bff.qoi.dataset` and `bff.tools.get_unitcell` to
  `bff.qoi`. Custom routines import `QoI`, `get_unitcell`, and `select_atoms`
  from `bff.qoi`.
- Routines are called as `routine(universe, *, frames, options)` or
  `routine(*, inputs, options)`; `system_id` and `sample_id` are no longer
  passed.
- Moved surrogate fitting (`fit_surrogates`, MAP search) to `bff.bayes.fit`;
  `bff.bayes.learning` holds posterior learning only.
- Simulation campaigns (`sample-parameters`, `validate`) and snapshot labeling
  run on Slurm as job arrays of one `run.sh` instead of one submission per
  job. `slurm.max_parallel_jobs` limits running tasks; the new
  `slurm.max_array_size` (default 1000) splits large campaigns into
  consecutive arrays.
- Campaign jobs write directly to `samples/<sample_id>/`; the `outputs/`
  working directory is gone. GROMACS runs inside each system's run directory,
  so `mdout.mdp` and bias outputs no longer land in input directories.
- A sample whose MD fails no longer aborts a local campaign; it is recorded
  as `failed`, as on Slurm.
- `sample-parameters` draws all samples from one Latin hypercube instead of
  drawing them one at a time, so samples now fill the parameter space evenly.
- `build` shares an equilibration only between systems with identical
  topology, templates, box, MDP files, and step counts.

- MCMC log-probabilities stay on the compute device (no per-step
  GPU-to-CPU copy), and local-GP predictions reuse a precomputed
  `K^-1 (y - mean)` instead of multiplying by the inverse kernel each call.
- A NaN parameter vector now gets `-inf` for its own walker only.

### Fixed

- The learning-rate search no longer accepts runs that diverged to NaN.
- Walkers can be initialized from uniform priors without charge constraints.
- `ChargeConstraint` objects can be copied and pickled.
- The adaptive proposal accumulates its covariance in float64, avoiding
  non-positive-definite proposals in long float32 runs.
- A transient `squeue` failure no longer counts submitted jobs as finished.
- Trajectory completeness uses the configured `n_steps`, so production MDP
  files with `nsteps <= 0` or `nstxout-compressed = 0` no longer mark every
  sample as failed or crash.
- Element guessing keeps name-based elements for deuterium and
  repartitioned hydrogens instead of failing.
- Molecule insertion stops with an error instead of looping forever when the
  box is too small.

## `0.4.2` - 2026-10-02

Local Gaussian-process fitting now derives its default hyperparameter-prior
centers from the input and residual target scales. This makes the defaults more
robust for scalar QoIs and mixed parameter scales.

Posterior plotting now limits expensive KDE and QoI-attribution work to
deterministic subsets by default, evaluates QoI likelihood contributions in
adaptive batches, and exposes these limits through the `plots` section of the
learn configuration. Marginal plots report KDE modes, use more legible panel
layouts and legends, and support per-parameter labels for arbitrary `define`
parameters. Local validation campaigns now display progress immediately and
advance the count after each completed MD job, including single-sample runs.

## `0.4.1` - 2026-08-25

RDF selection handling and atom-type expansion moved into the QoI adapter,
leaving the numerical kernel to operate directly on MDAnalysis AtomGroups.
Dynamic selections and numerical behavior are preserved.

## `0.4.0` - 2026-08-24

The pipeline now uses semantic system IDs and explicit role-based manifests.
Snapshot extraction and CP2K labeling are unified under `bff label-snapshots`;
all molecular and isolated-atom CP2K inputs are supplied explicitly by users;
the remaining stages are named `sample-parameters`, `build-qoi-datasets`, and
`fit-lgp`. This is a breaking config change; follow the
[pipeline migration guide](migration.md).

Build systems now include stable virtual-site-free reference assets,
`label-snapshots` writes a detailed results manifest, and validation can draw
samples directly from `outputs/posterior.pt`. Learning copies the authoritative
parameter specification into its fixed `outputs/` directory.

QoI analysis now parallelizes complete training samples, processes all systems
of a sample sequentially, and treats the reference through the same execution
path. Custom routines infer file-based execution from declared `inputs` rather
than a separate `loader` setting.

Static hydrogen-bond selections now reuse their donor topology and possible
labels across frames, substantially reducing reference-analysis time. QoI
dataset metadata is also guaranteed to remain acyclic during serialization.

Colvars-enabled sampling and validation jobs now rewrite
`colvars-configfile` relative to the GROMACS working directory, fixing missing
bias files in staged local and Slurm campaigns.

The numbered examples follow the new stage contract. The self-contained
notebooks use CUDA when available and otherwise fall back to CPU. `pytest` is
now installed through the `dev` extra instead of as a runtime dependency.

See the [repository changelog](https://github.com/vojtechkostal/BayesicForceFields/blob/main/CHANGELOG.md)
for the complete list of breaking changes and fixes.

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
