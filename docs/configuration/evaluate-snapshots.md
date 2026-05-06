# Evaluate Snapshots Configuration

Source code:

- `bff/workflows/evaluate_snapshots/config.py`
- `bff/workflows/evaluate_snapshots/main.py`

## Purpose

`bff evaluate-snapshots` runs staged CP2K snapshot jobs. The staged assets
normally come from `bff prepare-assets`.

## Run Example

```yaml
output_dir: ./snapshots
job_scheduler: local
cp2k_cmd: cp2k.psmp
single_atoms: true
snapshot_md_steps: 100
train_fraction: 0.8
seed: 2026

systems:
  - assets: ../02-assets/reference/system-000
```

The staged snapshot `md.inp` uses GFN1-xTB by default. Per-system CP2K input
overrides are optional:

```yaml
systems:
  - assets: ../02-assets/reference/system-000
    md: ../inputs/reference-inputs/md-0.inp
    sp: ../inputs/reference-inputs/revpbe0-sp.inp
```

## Top-Level Keys

- `output_dir`
  Output directory written by `bff evaluate-snapshots`.
- `systems`
  Non-empty list of systems to evaluate.
- `job_scheduler`
  Either `local` or `slurm`.
- `cp2k_cmd`
  CP2K executable used for local execution.
- `single_atoms`
  Whether to also run isolated single-atom reference jobs. Defaults to `true`.
- `snapshot_md_steps`
  Optional override for the staged short GFN1-xTB MD length.
  When CP2K older than 2025 is detected, BFF removes the `GFN_TYPE` keyword
  from that staged xTB input for compatibility.
- `train_fraction`
  Fraction of collected snapshot frames written into `train.extxyz`.
- `seed`
  Deterministic shuffle seed used before the train/validation split.
- `cleanup_snapshots`
  Remove collected snapshot run directories and the staged `single-atoms/`
  run tree after successful collection.
- `collection_wait_seconds`
  Grace period for delayed `sp.extxyz` files on shared filesystems.
- `slurm`
  Required when `job_scheduler: slurm`.

## `systems[]` Keys

- `assets`
  Path to one staged reference system directory, usually
  `reference/system-XXX/` written by `bff prepare-assets`.
- `md`
  Optional CP2K MD input override for this system.
- `sp`
  Optional CP2K single-point input override for this system.

## Outputs

`bff evaluate-snapshots` writes or refreshes one directory per system under
`output_dir`, for example `snapshots/system-000/`, containing:

- `snapshots/snapshot-XXXX/`
- `train.extxyz`
- `valid.extxyz`
- optional `single-atoms.yaml` with atomic-number keys

Reference trajectories for `bff analyze` are user-provided. A convenient
convention is to place them under `03-reference/trajectories/system-*/`
alongside matching `system.top` and `system.gro` files.
