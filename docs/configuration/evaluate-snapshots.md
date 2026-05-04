# Evaluate Snapshots Configuration

Source code:

- `bff/workflows/evaluate_snapshots/config.py`
- `bff/workflows/evaluate_snapshots/main.py`

## Purpose

`bff evaluate-snapshots` either runs staged CP2K snapshot jobs or imports
externally generated trajectories into BFF's canonical evaluated-snapshot
layout.

In `run` mode, the staged assets normally come from
`bff prepare-assets`.

Supported modes:

- `mode: run`
  execute the staged CP2K snapshot and optional single-atom jobs
- `mode: import`
  copy externally generated `.top`, `.gro`, and trajectory files into canonical assets

## Run Example

```yaml
mode: run
output_dir: ./
job_scheduler: local
cp2k_cmd: cp2k.psmp
single_atoms: true
snapshot_md_steps: 100
train_fraction: 0.8
seed: 2026

systems:
  - assets: ../01-build-colvars/reference/system-000
```

The staged snapshot `md.inp` uses GFN1-xTB by default. Per-system CP2K input
overrides are optional:

```yaml
systems:
  - assets: ../01-build-colvars/reference/system-000
    md: ../inputs/reference-inputs/md-0.inp
    sp: ../inputs/reference-inputs/revpbe0-sp.inp
```

## Import Example

```yaml
mode: import
output_dir: ./

systems:
  - topology: ../01-build-colvars/reference/system-000/system.top
    coordinates: ../01-build-colvars/reference/system-000/system.gro
    trajectory: ../inputs/reference-trajectories/pos-000.xtc
```

Import mode requires all three files for each system:

- `topology`: a `.top` file
- `coordinates`: a `.gro` file
- `trajectory`: a trajectory file such as `.xtc`

## Top-Level Keys

- `mode`
  Either `run` or `import`.
- `output_dir`
  Output directory written by `bff evaluate-snapshots`.
- `systems`
  Non-empty list of systems to run or import.
- `job_scheduler`
  Required in `run` mode. Either `local` or `slurm`.
- `cp2k_cmd`
  Required in `run` mode. CP2K executable used for local execution.
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

## `systems[]` Keys In `run` Mode

- `assets`
  Path to one staged reference system directory, usually
  `reference/system-XXX/` written by `bff prepare-assets`.
- `md`
  Optional CP2K MD input override for this system.
- `sp`
  Optional CP2K single-point input override for this system.

## `systems[]` Keys In `import` Mode

- `topology`
  Topology file for the imported system. Must be `.top`.
- `coordinates`
  Coordinate file for the imported system. Must be `.gro`.
- `trajectory`
  Trajectory file copied into the canonical `trajectory.*` asset name.

## Outputs

In `run` mode, `bff evaluate-snapshots` writes or refreshes:

- `snapshots/snapshot-XXXX/`
- `train.extxyz`
- `valid.extxyz`
- optional `single-atoms.yaml` with atomic-number keys

In `import` mode, it writes one canonical directory per system containing:

- `system.top`
- `system.gro`
- `trajectory.*`
- `imported.yaml`
