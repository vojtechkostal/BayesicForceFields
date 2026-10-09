# Changelog

The public reproduction snapshot for the published study is archived as
[`v0.0.1`](https://github.com/vojtechkostal/BayesicForceFields/tree/v0.0.1).
Use that tag for exact reproduction of the paper results. The current workflow
release is `0.5.0`.

## `0.5.0` - 2026-10-09

### Changed

- BFF pins `gmxtopology==0.3.0`, so upgrading `bfflearn` also upgrades `gmxtopology`. Topologies are read with their conditional
  blocks evaluated, bonded parameters are matched on bonded atom types, and
  written topologies keep only the molecule types, atom types and CMAP grids of
  the system.
- Examples are streamlined. The acetate example has one directory per stage,
  named after its command (`01-build`, `02-reference-md`,
  `03-sample-parameters`, `04-build-qoi-datasets`, `05-fit-lgp`, `06-learn`,
  `07-validate`), each holding one `config.yaml`; the old `configs/`
  directory and its copy step are gone. It has two systems: `acetate` (QoI:
  acetate-water RDFs) and `calcium-acetate`, whose Ca-C2 distance is biased by
  Colvars metadynamics (`inputs/colvars.dat`) and whose PMF is a QoI read by
  the custom file routine `inputs/pmf.py` and compared with an MLIP reference
  PMF (`02-reference-md/calcium-acetate.pmf`). The sampling config runs
  locally and carries the Slurm settings as a commented block. The CP2K inputs
  of the labeling script (revPBE0-D3 only) are in `02-reference-md/cp2k/`.
  The two acetate notebooks are merged into `06-learn/posterior.ipynb`, and
  the arbitrary-data and neon notebooks are shortened. `LGPCommittee.n_eff` is
  inferred from the reference curve when the committee is created.
- `fit-lgp` always runs on the CPU in float64 (`fit.device` is removed) and
  the hyperparameter search uses L-BFGS-B with autograd gradients inside box
  bounds derived from the priors, with up to two restarts from prior draws.
  `fit.lr` is removed; `fit.max_iter` (default `500`) and `fit.tol_grad`
  (default `1e-4`, largest gradient component) control the search. On the
  acetate data it needs 8-25 iterations instead of the former 10,000 steps.
  The Laplace covariance of committees floors non-positive curvatures.
  Model files hold CPU tensors and load on any machine (including models
  fitted on a GPU by earlier versions); `bff learn` moves them once to
  `mcmc.device`, which now defaults to `auto` (`cuda` if available). The
  sampling loop no longer synchronizes with the host: `Specs`, `Priors`, the
  likelihood, and the accept step run on the device. `gaussian_kernel` always
  uses explicit differences (no `cdist` switch), `LocalGaussianProcess` has no
  `device` argument (use `.to(device, dtype)`), and `fit_surrogates` has no
  `device` argument. `log_posterior` no longer takes `device`.
- MCMC convergence uses one diagnostic, `bff.mcmc.convergence.diagnose`:
  rank-normalized split R-hat plus bulk and tail effective sample sizes from
  the pooled multi-walker autocovariance (Geyer truncation), for every
  parameter and the log probability. Sampling stops when `rhat_tol` and
  `ess_min` are met at two consecutive checks; `ess_min` now defaults to
  `400`. `Results.diagnostics()` reports the same quantities (`rhat`,
  `ess_bulk`, `ess_tail`; no autocorrelation time) and `Results.info["mcmc"]`
  stores `max_rhat` and `min_ess`. `ConvergenceInfo` and the autocorrelation
  helpers are removed; older checkpoints cannot be resumed.
- `learn` writes one self-contained `outputs/results.pt` (prior, posterior
  chain, specifications, QoI information, per-QoI likelihoods) instead of
  `posterior.pt` plus `prior.pt`, and `bff.Results` replaces
  `PosteriorResults`. It provides `samples` in physical units, `map` (the state
  of highest log posterior), `summary()`, `diagnostics()`, `draw()`, and the
  plot methods. Samples are no longer stripped of outliers, and nuisance
  columns are named `noise <qoi>`. `validate` gains `posterior.include_map`.
  Plot functions take only a `Results` and return the figure. In Python,
  `LearningProblem(constraint=...)` is `specs=...`, and `fn_posterior`/
  `fn_priors` is `fn_results`. Checkpoints and posterior files of earlier
  versions cannot be read; rerun `bff learn`.
- `Specs` is the single parameter specification (`bounds`, `names`,
  `explicit_names`, `implicit_names`, `complete`, `is_valid`, `violations`).
  `latin_hypercube(specs, n, seed)` replaces `RandomParamsGenerator`.

- `learn` infers each QoI's effective number of observations from the
  correlation length of its reference curve (Satterthwaite degrees of freedom;
  scalars count once per value) instead of the peak-counting heuristic.
  `models.<name>.tolerance` is now the accepted deviation from the reference
  in the QoI's units and adds to the likelihood variance (`sigma^2 +
  tolerance^2`); it defaults to `0`. `bff.bayes.effective_observations`
  provides `effective_observations`, `curve_n_eff`, and `correlation_length`.
- Surrogates and datasets use common Gaussian-process names. `QoIDataset`:
  `X`, `y`, `y_ref` (were `inputs`, `outputs`, `outputs_ref`).
  `LocalGaussianProcess`: `mean`, `lengthscales`, `amplitude`,
  `noise_variance` (were `y_mean`, `lengths`, `width`, `sigma`; `sigma` was
  always a variance), `n_inputs`, `n_outputs`. `LGPCommittee`: `members`,
  `y_ref`, `test_error`, `n_members` (were `lgps`, `reference_values`,
  `error`, `size`). `fit_surrogates(y_means=...)` is now `means=...`.
- Surrogate means: `mean` defaults to `data` (the average training output per
  value) instead of `0`; it also accepts a number, `sigmoid`, or a custom
  `path.py:function`/`module:function`. The RDF sigmoid is centred where each
  reference RDF first reaches 0.5 instead of at a fixed 3 angstrom. Custom
  means are saved in models by reference. A saved model is reused only when
  both its dataset and its mean are unchanged; otherwise it is refitted.
- Charge constraints: `implicit` names an atom (name or type) of exactly one
  `charge ...` parameter in `bounds` instead of the parameter label. Each
  constraint is one linear equation, and all implicit charges are solved
  together; the rules for nested, non-overlapping constraints and their
  ordering are gone.
- Campaign artifacts: one `campaign.yaml` holds the job settings shared by all
  samples (jobs run as `bff md campaign.yaml <sample_id>`) instead of a
  `config.yaml` per sample. `samples.yaml` names its parameter columns
  (`parameter_names`), lists `systems` with their `n_steps`, and records each
  sample's outputs per system by role, including the sample's own `topology`,
  which `cleanup` now always keeps. YAML is read and written with libyaml when
  available.
- `build-qoi-datasets` analyzes each sample with its own topology, so routines
  see the sampled charges. A sample with missing files, a failing trajectory
  or routine, or mismatching QoIs is skipped with a warning instead of
  stopping the stage. In-memory trajectories that would not fit in the
  available memory are read from disk. Datasets record `sample_ids` and
  `parameter_names`; fitted models keep the parameter names, and `learn`
  rejects models fitted to other parameters than its `specs.yaml`.
- Stage logs share one layout: the stage title, `Config` and one line per
  input or setting, `<step>: Done. | ...` lines, warnings, and a final
  `<Stage>: Done. | summary | time | output`. Progress lines are shown on
  the console only.
- `sample-parameters` and `validate` share one campaign flow: both write
  `specs.yaml` into the campaign, check every parameter sample against the
  bounds before staging, list all samples in `samples.yaml` from the start,
  record the parameter source under `samples.yaml` `provenance`, and log the
  same parameter summary. New options: `resume` (continue a campaign, running
  only samples not yet completed), `overwrite`, `local.max_parallel_jobs`
  (parallel local samples), and `seed` for `sample-parameters` (a fresh seed
  is drawn and recorded otherwise).
- Charge-constraint compilation moved to `bff.domain.charge_constraints`
  (`compile_specs`, `check_specs_topologies`).
- All stage configurations are read by one shared reader with the same rules:
  unknown keys are errors, `null` means "use the default", integers must be
  whole numbers, numbers must be finite (exponent strings such as `1e-3` are
  accepted), booleans must be `true`/`false`, and every error names the full
  key path. Values are now range-checked consistently: for example
  `nsteps.prod >= 1`, `bounds` with `lower < upper`, `mcmc.rhat_tol > 1`,
  `mcmc.priors_disttype` in `normal`/`uniform`, `fit-lgp` `mean` a number or
  `sigmoid`, and devices `cpu`, `cuda[:N]`, or `mps`.
- The configuration reference lists every option of every stage with type,
  default, and meaning, plus a new conventions page.
- `charge_constraints` (sample-parameters), `fit` (fit-lgp), and `mcmc`
  (learn) may be omitted to use their defaults.
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
- Simulation campaigns (`sample-parameters`, `validate`) run on Slurm as job
  arrays of one `run.sh` instead of one submission per job. `slurm.max_parallel_jobs` limits running tasks; the new
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
- The source distribution on PyPI no longer contains the tests (the wheel never
  did); they stay in the repository.
- Documentation is reorganized: a settings overview links every option group
  of every stage, *How BFF Works* replaces the architecture page (the package
  map moved to *Development*), and the README and docs state that BFF does not
  generate reference data.
- `slurm.sbatch.time` must be a quoted string; YAML reads an unquoted
  `4:00:00` as the number 14400. `slurm.teardown` also runs after a failed
  task. `build.systems[].box` accepts only rectangular boxes, as molecule
  insertion always assumed. `validate.posterior.seed` must be >= 0.
- Errors of every stage are reported as one line naming the stage, instead of
  a usage message about the config argument.

### Removed

- The top-level `data/` directory of force-field, MDP, and water templates:
  provide your own inputs, for example from `examples/acetate/inputs/`.
- `mcmc.include_implicit_charge`, `Bounds`, `ChargeConstraint`,
  `RandomParamsGenerator`, `Priors` file I/O and `from_any`, `sample_posterior`,
  and `prepare_samples`.
- Removed `models.<name>.n_eff` and `models.<name>.independent_observations`
  from learn configs, and `estimate_curve_n_eff`.
- Removed the build stage's per-system `system.yaml` (no stage read it),
  `build-qoi-datasets`' `qoi/raw.json` with its `output.write_raw` option, and
  its `training_samples.progress_stride` option.
- Removed the `project: <path>` shorthand and the undocumented top-level
  `fn_log` from build configs (use `project.directory` and `project.log`), the
  `kind` key of build `bias` mappings, and boolean or single-string `store`
  values. `plots.max_marginal_samples: null` now means the default; use `-1`
  for all samples.
- Removed `bff label-snapshots`, the hidden `label-snapshot-job` command, the
  `bff.label_snapshots` and `Project.label_snapshots` Python API, and
  `bff.io.cp2k`/`bff.io.extxyz`. BFF no longer runs CP2K; the reference MD is
  an external step documented in the new "Reference trajectories" page. The
  standalone, Slurm-only `scripts/label_structures.py` labels foundation-model
  MD frames with CP2K for MLIP fine-tuning (train/valid/test EXTXYZ and
  isolated-atom energies keyed by atomic number).
- Removed the `charge` and `multiplicity` build keys and the matching
  `system.yaml` fields; only CP2K labeling used them.

### Fixed

- A campaign's final cleanup deleted the checkpoints of samples that were
  still incomplete, so `resume: true` had to restart them from scratch; it now
  prunes only systems whose production run finished. A rerun no longer fails
  on the topology left by an attempt that never reached a checkpoint.
- A Slurm task stopped by `SIGTERM` (time limit, `scancel`) now lets the
  running mdrun checkpoint, copies its files back, and is recorded as
  `incomplete`, so `max_restarts` continues it; before, the sample was lost as
  `failed`. A sample whose job writes no result is `failed` instead of
  reusing a result from an earlier round, and local reruns append to
  `run.out` instead of truncating it.
- MDP files with inline comments (`nsteps = 500 ; 1 ps`), indented comments,
  or `_` spellings of keys failed or were patched twice; MDP keys are now read
  the way GROMACS reads them.
- Custom trajectory routines receive the universe positionally, as
  documented, so its parameter may have any name. Samples whose QoI settings
  (for example a grid) differ from the reference are skipped. `~` is expanded
  in reference inputs and custom routine paths. Parallel analysis caps
  in-memory trajectories by the Slurm memory allocation, and a worker killed
  for lack of memory is reported with a hint.
- `scripts/label_structures.py run` collected results while array tasks were
  still running, because `sbatch --wait` returned early; it now polls
  `squeue` until the arrays leave the queue.
- `bff learn` resumes a checkpoint written on another device type (CPU vs.
  CUDA) instead of failing on the random-generator state.
- A charge-constraint selection that fails reports the underlying error
  instead of calling the selection invalid.
- `fit-lgp` refits a model file written by an older BFF version instead of
  stopping with `reuse_models: true`.
- `build-qoi-datasets` writes its progress to the log every 100 samples and
  announces the start of the reference and of the sample analysis; a batch
  job's console shows the same lines instead of one per sample.
- `mean: sigmoid` no longer fails when the sigmoid values are built into the
  surrogate.
- Running `sample-parameters` or `validate` again on an existing campaign no
  longer mixes newly drawn parameters with the old samples' trajectories; it
  is an error unless `resume` or `overwrite` is set.
- Validation campaigns in explicit `parameters` mode now contain
  `specs.yaml`, so `build-qoi-datasets` can analyze them.
- `validate` rejects parameter samples outside the bounds, unknown parameter
  names, and charge constraints that give a different equation in the
  validation systems before staging, instead of failing in every MD job.
- `max_restarts` with `job_scheduler: local` is an error instead of being
  ignored.

- Campaigns started from a build directory (`source`) no longer require the
  build's `production.xtc`, `npt.mdp`, or a loadable `system.yaml`; only the
  files they use must exist.
- A campaign sample is complete once its production run reaches `n_steps`,
  read from the final checkpoint in the GROMACS log. Runs whose MDP writes no
  compressed trajectory, or whose trajectory is not stored, are no longer
  reported as `incomplete` and resubmitted. A `production.done` marker keeps
  finished systems from rerunning after `cleanup` in a scratch directory.
- Samples whose campaign did not store a trajectory are usable by
  build-qoi-datasets when their routines only read stored files; a routine
  that needs a missing output fails before analysis with the sample, system,
  and missing role. A missing stored output file is reported per sample.

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
