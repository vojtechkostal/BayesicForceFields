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
