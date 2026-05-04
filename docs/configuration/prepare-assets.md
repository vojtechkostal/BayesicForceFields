# Prepare Assets Configuration

Source code:

- `bff/workflows/prepare_assets/config.py`
- `bff/workflows/prepare_assets/main.py`

## Purpose

`bff prepare-assets` packages the seeded production trajectories from
`bff build` into both downstream asset trees:

- reusable FFMD assets for `bff sample` and `bff validate`
- staged CP2K snapshot assets for `bff evaluate-snapshots`

## Minimal Example

```yaml
manifest: ./build-manifest.yaml
ffmd_dir: ./ffmd
reference_dir: ./reference
n_single_point_snapshots: 1000
```

## Top-Level Keys

- `manifest`
  Build manifest written by `bff build`.
- `ffmd_dir`
  Output directory for `system-XXX/` FFMD asset folders. Defaults to `ffmd/`
  next to the manifest.
- `reference_dir`
  Output directory for staged CP2K reference asset folders. Defaults to
  `reference/` next to the manifest.
- `n_single_point_snapshots`
  Number of evenly spaced snapshots sampled from each seeded production
  trajectory for CP2K evaluation.
- `systems`
  Optional list of system ids to package. Items may be `0`, `000`,
  `system-000`, or mappings with `system_id`.

## Outputs

For FFMD, each selected system writes:

- `FFMD_DIR/system-XXX/system-XXX.top`
- `FFMD_DIR/system-XXX/system-XXX.gro`
- `FFMD_DIR/system-XXX/system-XXX.em.mdp`
- `FFMD_DIR/system-XXX/system-XXX.npt.mdp`
- `FFMD_DIR/system-XXX/system-XXX.mdp`
- `FFMD_DIR/system-XXX/system-XXX.ndx`
- optional copied bias input

For reference snapshots, each selected system writes:

- `REFERENCE_DIR/system-XXX/system.gro`
- `REFERENCE_DIR/system-XXX/system.top`
- `REFERENCE_DIR/system-XXX/system.xyz`
- `REFERENCE_DIR/system-XXX/md/`
- `REFERENCE_DIR/system-XXX/single-atoms/`
- `REFERENCE_DIR/system-XXX/snapshots/`

Run `bff evaluate-snapshots` on the staged reference assets to execute CP2K
and collect `train.extxyz`, `valid.extxyz`, and optional isolated-atom
energies.
