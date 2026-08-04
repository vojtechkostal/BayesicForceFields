# Pipeline Directory-Contract Migration

This release deliberately has no runtime compatibility shim for the previous
pipeline configuration. Keep old output directories intact and create new
stage directories for the convention-based workflow.

## Required Changes

1. Give every physical system a semantic `system_id`, such as `acetate` or
   `acetate-contact`. IDs must be unique, lowercase, file-safe, and match
   `[a-z0-9][a-z0-9._-]*`. Use the optional `system_name` only for display.
2. Replace `prepare-assets` with `prepare-reference`. Point it at the build
   root with `source`; it creates only CP2K/reference inputs.
3. Point `sample` and `validate` directly at the build root with `source`, and
   point `evaluate-snapshots` at the prepare-reference root. Alternatively,
   provide complete per-system `inputs`; the two forms cannot be mixed.
4. Rename the analysis `sample` section to `training_samples`. Move routines
   out of reference-system entries into the top-level `routines` list, and
   give each routine its applicable `systems`.
5. Replace residue shortcuts with complete MDAnalysis selections. RDF routines
   require `group_a` and `group_b`; hydrogen-bond routines require `donors`,
   `hydrogens`, and `acceptors`.
6. Replace `bff fit`, `bff.fit`, and `Project.fit` with `bff lgpfit`,
   `bff.lgpfit`, and `Project.lgpfit`. There is no alias.
7. Replace learning `restart` and individual artifact paths with
   `mcmc.resume`, `output.overwrite`, and one `output.directory`. Learning now
   owns fixed `output/` and `plots/` children below that root.
8. Regenerate QoI datasets and `.lgp` models. RDF normalization/PBC handling
   changed, and model reuse now verifies an exact dataset fingerprint.

## Identity and Paths

Training and reference system sets must contain the same unique IDs. YAML list
order is irrelevant: analysis pairs records by ID. Build and reference file
roles follow documented fixed names under `systems/<system_id>/`. Local
`system.yaml` files contain metadata only. Paths in `samples.yaml` are relative
to that file; user paths remain relative to their config.

`specs.yaml` remains limited to ordered parameter bounds and compiled charge
constraints. It does not carry system, path, scheduler, or learning metadata.

No `schema_version` or replacement version field was introduced.

## Learning Outputs

New learning runs reject existing stage-owned artifacts unless
`output.overwrite: true` is explicit. Resuming requires `mcmc.resume: true` and
a compatible `output/mcmc.ckpt`; resume and overwrite cannot be combined.
Only the known learning artifacts are replaced by overwrite, so unrelated
files in the output root are preserved.

Existing posterior files can still be loaded read-only where their contents
are compatible, but old checkpoint directories are not migrated in place.
