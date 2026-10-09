# Migration

## From 0.4.2 to 0.5.0

1. Custom QoI routines import from `bff.qoi`: replace
   `from bff.qoi.data import QoI` and `from bff.tools import get_unitcell`
   with `from bff.qoi import QoI, get_unitcell`. Python code using
   `bff.qoi.data.QoIDataset` imports `QoIDataset` from `bff.qoi`.
   Routines no longer receive `system_id` and `sample_id`; drop both
   parameters: `routine(universe, *, frames, options)` or
   `routine(*, inputs, options)`. Results in `raw.json` are already keyed
   by system and sample.
2. Invalid built-in routine options are reported when the reference trajectory
   is analyzed instead of when the configuration is loaded.
3. Campaign logs (`config.yaml`, `run.out`, `gmx.log`) now live in
   `samples/<sample_id>/`; there is no `outputs/` directory. Existing campaign
   manifests remain readable.
4. Remove any `array` entry from `slurm.sbatch`. If the cluster limits array
   size or submitted jobs per user below 1000, set `slurm.max_array_size`.
   Slurm output appears in `samples/<sample_id>/run.out` (campaigns) and
   `slurm/<job>_<task>.out`.
5. Remove `mdp_npt` from explicit campaign `inputs`; it was never used there.
6. Import `fit_surrogates` from `bff.bayes.fit`; `LearningProblem` stays in
   `bff.bayes.learning`.
7. Replace `cd "$WORKDIR"`-style `slurm.setup`/`teardown` lines with
   `scratch_dir`; GROMACS runs in its sample (or scratch) directory regardless
   of the shell's working directory.
8. `bff label-snapshots` is gone. Produce the reference trajectory outside
   BFF (see [Reference trajectories](reference-trajectories.md)). To label
   frames with CP2K, use `scripts/label_structures.py` on Slurm: rename
   `system_id` to `id`, replace `md_input`/`sp_input` with one single-point
   `cp2k_input` that reads `structure.xyz` and includes `cell.inc`, move
   `single_atom_inputs` to the top level, replace `train_fraction` with
   `split`, and drop `job_scheduler`, `single_atoms`, `cleanup_snapshots`, and
   `collection_wait_seconds`. The short CP2K MD before each single point is no
   longer run; label frames from foundation-model MD instead.
9. Delete `charge` and `multiplicity` from build configs. Existing build
   outputs, including their `system.yaml`, work unchanged.
10. Configuration files are checked more strictly; see the
    [conventions](configuration/index.md). In build configs, write
    `project: {directory: <path>}` instead of `project: <path>`, move a
    top-level `fn_log` to `project.log`, and drop `kind` from `bias`. Write
    `store` as a list (`store: [xtc]`). Replace
    `plots.max_marginal_samples: null` with `-1`. Values that were silently
    truncated or accepted before, such as `n_steps: 1.5`, equal parameter
    bounds, `rhat_tol: 1`, or an unknown `priors_disttype`, are now errors.
11. Rerunning `sample-parameters` or `validate` on an existing campaign
    directory now fails. Add `resume: true` to continue it or
    `overwrite: true` to replace it. Remove `max_restarts` from local
    campaigns. Python code importing `compile_specs` from
    `bff.workflows.sample_parameters.main` imports it from
    `bff.domain.charge_constraints` and passes bounds, constraints, and
    topology paths.
12. In `charge_constraints`, set `implicit` to the atom name or type instead of
    the parameter label: `implicit: C2` instead of `implicit: "charge C2"`.
    Constraints no longer need to be disjoint or nested.
13. Campaigns and QoI datasets written by earlier versions cannot be read:
    `samples.yaml` has a new layout and jobs read `campaign.yaml`. Rerun
    `sample-parameters`/`validate` (or `build-qoi-datasets` on a new campaign),
    then `fit-lgp` and `learn`. Remove `training_samples.progress_stride` and
    `output.write_raw` from build-qoi-datasets configs; `qoi/raw.json` is no
    longer written. Delete any `systems/<id>/system.yaml` you no longer need.
14. Rebuild QoI datasets and refit `.lgp` models: both use the new names.
    In Python code, use `QoIDataset(X=..., y=..., y_ref=...)`,
    `committee.members`, `committee.y_ref`, `committee.test_error`, and
    `fit_surrogates(means=...)`. In fit-lgp configs, the default `mean` is now
    `data`; write `mean: 0` to keep the old zero mean.
15. In learn configs, remove `n_eff` and `independent_observations`; BFF
    infers the effective observations. Keep or set `tolerance` as the
    deviation you accept in the QoI's units (for example `0.1` for an RDF);
    it now widens the likelihood instead of counting curve features, so
    posteriors change where the tolerance exceeds the learned noise.
16. Rerun `bff learn`: it now writes `outputs/results.pt` and old
    `posterior.pt`/`prior.pt`/checkpoints cannot be read. Point
    `validate.posterior.file` at `results.pt` (optionally `include_map: true`),
    and remove `mcmc.include_implicit_charge`. In Python, use
    `bff.Results.load(...)`, `results.draw(...)`, `results.map`, and
    `LearningProblem.from_models(models, specs=specs)` with `fn_results=`;
    plot functions take the `Results` and return figures.
17. `mcmc.ess_min` defaults to `400` and counts the smaller of the bulk and
    tail effective sample sizes; set `ess_min: 100` to keep the old looseness.
    `Results.diagnostics()` returns `rhat`, `ess_bulk`, and `ess_tail`
    (previously `ess` and `autocorr_time`). Restart runs with `mcmc.resume`
    only from checkpoints written by this version.
18. In fit-lgp configs remove `fit.device` and `fit.lr`; fitting always runs on
    the CPU. Optionally tune `fit.max_iter` and `fit.tol_grad` (new default
    `1e-4`). In learn configs, `mcmc.device` now defaults to `auto`. In Python
    code, drop `device=` from `fit_surrogates` and `LocalGaussianProcess`,
    use `model.to(device)` if you predict on a GPU yourself, and call
    `log_posterior(theta, priors, log_likelihood_fn)` without `device`.
    `find_map` takes `bounds` and returns a `MapResult`.
19. The acetate example moved to one directory per stage with its config inside
    (`cd 06-learn && bff learn config.yaml`); copy your own configs next to
    the stage they belong to. The `data/` directory of templates is removed;
    `LGPCommittee.n_eff` is inferred at creation, so Python code no longer
    needs `effective_observations` to set it.
20. The acetate example has two systems, `acetate` and `calcium-acetate`
    (formerly `acetate`, `acetate-contact`, `acetate-separated`), and one
    config per stage; Slurm settings are a commented block in
    `03-sample-parameters/config.yaml`. Its topologies are
    `inputs/acetate.top` and `inputs/calcium-acetate.top`.
21. Quote `slurm.sbatch.time` (`time: "04:00:00"`); a number is rejected.
    Give `build` boxes as three lengths: angles other than 90 are rejected.

# Pipeline Directory-Contract Migration

This release deliberately has no runtime compatibility shim for the previous
pipeline configuration. Keep old output directories intact and create new
stage directories for the convention-based workflow.

## Required Changes

1. Give every physical system a semantic `system_id`, such as `acetate` or
   `acetate-contact`. IDs must be unique, lowercase, file-safe, and match
   `[a-z0-9][a-z0-9._-]*`. Use the optional `system_name` only for display.
2. Replace `prepare-reference` and `evaluate-snapshots` with
   `label-snapshots`. Provide each system's topology, trajectory, CP2K MD and
   single-point inputs, and snapshot count explicitly. When isolated-atom
   energies are enabled, also provide one CP2K input per detected element in
   `single_atom_inputs`; BFF no longer generates these inputs. Regenerate
   labels; old prepare/evaluate stage layouts are not loaded.
3. Replace `sample`, `analyze`, and `lgpfit` with `sample-parameters`,
   `build-qoi-datasets`, and `fit-lgp`. The Python names use underscores.
4. Rename the analysis `sample` section to `training_samples`. Move routines
   out of reference-system entries into the top-level `routines` list, and
   give each routine its applicable `systems`.
5. Replace residue shortcuts with complete MDAnalysis selections. RDF routines
   require `group_a` and `group_b` and now emit one curve per atom type in
   `group_a`. Hydrogen-bond routines require `selection` and `water_selection`;
   donor hydrogens and both interaction directions are inferred automatically.
6. Remove `loader` from custom QoI routines. A callable with `inputs` is
   file-based; a callable without `inputs` is trajectory-based. Remove
   `run.gc_collect` and `run.maxtasksperchild`; `run.in_memory` is the only
   analysis runtime option.
7. In fit-LGP configs, rename the `lgpfit` options section to `fit`.
8. Replace learning `restart` and individual artifact paths with
   `mcmc.resume`, `output.overwrite`, and one `output.directory`. Learning now
   owns fixed `outputs/` and `plots/` children below that root. The plural
   `outputs/` replaces the earlier singular directory.
9. Regenerate QoI datasets and `.lgp` models. RDF normalization/PBC handling
   changed, and model reuse now verifies an exact dataset fingerprint.
10. Rerun `bff build` to create each system's stable
   `reference/topology.top` and `reference/coordinates.gro`. Configure
   `build-qoi-datasets` with those files and the external MLIP trajectory as
   separate explicit inputs.
11. Validation may now consume `outputs/posterior.pt` directly. Posterior mode
    rejects an external `specs` path and writes the artifact's embedded
    specifications into the validation campaign. Keep `parameters` plus
    `specs` only for explicitly exported YAML samples.

## Identity and Paths

Training and reference system sets must contain the same unique IDs. YAML list
order is irrelevant: QoI construction pairs records by ID. Build files follow
documented fixed names under `systems/<system_id>/`; label-snapshots inputs are
explicit. Paths in `samples.yaml` are relative to that file, while user paths
remain relative to their config.

`specs.yaml` remains limited to ordered parameter bounds and compiled charge
constraints. It does not carry system, path, scheduler, or learning metadata.

No `schema_version` or replacement version field was introduced.

## Learning Outputs

New learning runs reject existing stage-owned artifacts unless
`output.overwrite: true` is explicit. Resuming requires `mcmc.resume: true` and
a compatible `outputs/mcmc.ckpt` plus a matching `outputs/specs.yaml`; resume
and overwrite cannot be combined.
Only the known learning artifacts are replaced by overwrite, so unrelated
files in the output root are preserved.

Existing posterior files can still be loaded read-only where their contents
are compatible, but old checkpoint directories are not migrated in place.
