## Unreleased

### Changed

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

### Changed

- Bounded the posterior sample counts used for default corner, standard
  marginal, and QoI marginal plots. The standard marginal cap accepts `-1` or
  `null` to use every sample. QoI likelihood attribution now runs in adaptive
  batches to avoid CUDA memory exhaustion after learning.
- Centered default LGP hyperparameter priors on the observed input and residual
  target scales, avoiding noise-dominated fits for scalar QoIs whose natural
  parameter or output scales differ substantially from one.
- Reported the mode of each posterior marginal instead of its mean, limited
  charge modes to three decimal places, and formatted other parameter modes
  with three significant digits.
- Balanced marginal plots across rows of at most five parameters, placed each
  arbitrary `define` parameter on an independent axis, and added configurable
  axis labels through `plots.plot_metadata`. Marginal figures now reserve
  measured space for legends so they cannot overlap single- or multi-panel
  plots, align y-axis labels within subplot columns, and leave sufficient
  gutters between neighboring panel sections.
- Made local validation progress visible immediately, including campaigns with
  a single sample, and advanced the displayed count only after each MD job has
  completed.

## `0.4.1` - 2026-08-25

### Changed

- Simplified RDF calculation so selection handling and atom-type expansion
  occur in the QoI adapter while the numerical kernel operates directly on
  MDAnalysis AtomGroups. Dynamic selections and numerical behavior are
  preserved.
