# Reference Configuration

Source code:

- [configs.py][reference-configs]
- [reference.py][reference-workflow]

## Purpose

`bff reference` executes the CP2K reference assets staged by `bff prepare`.

It can:

- stage one run directory per snapshot under a separate output tree such as
  `02-reference-data/reference-assets/system-XXX/snapshots/snapshot-XXXX/`
- run the short GFN1-xTB MD plus final single-point evaluation
- collect `sp.extxyz` files into `train.extxyz` and `valid.extxyz`
- optionally run isolated single-atom reference calculations
- run either locally or through Slurm

Staged short-MD inputs request only one CP2K restart backup. After each CP2K
reference job finishes, BFF removes `*.wfn*` and `*.restart*` files from the
run directory to avoid keeping large restart artifacts.

## Minimal Example

```yaml
reference_dir: ./reference-assets
job_scheduler: local
cp2k_cmd: cp2k.psmp
single_atoms: true
snapshot_md_steps: 100
train_fraction: 0.8
seed: 2026

systems:
  - assets: ../01-prepare/colvars/reference/system-000
  - assets: ../01-prepare/colvars/reference/system-001
```

Per-system CP2K input overrides are optional:

```yaml
systems:
  - assets: ../01-prepare/colvars/reference/system-000
    md: ./inputs/system-000-md.inp
    sp: ./inputs/revpbe0-sp.inp
```

## Slurm Example

```yaml
reference_dir: ./reference-assets
job_scheduler: slurm
cp2k_cmd: cp2k.psmp
single_atoms: true
snapshot_md_steps: 100

systems:
  - assets: ../01-prepare/colvars/reference/system-000

slurm:
  max_parallel_jobs: 20
  sbatch:
    partition: cpu
    time: "04:00:00"
    mem: 10G
  setup:
    - module load cp2k
```

In Slurm mode, `cp2k_cmd` is kept for config symmetry but not injected into the
batch job. Make `cp2k.psmp` available through `slurm.setup`, your shell
environment, or your site scheduler defaults.

## Top-Level Keys

- `reference_dir`
  Output directory written by `bff reference`. In the acetate example this is
  `./reference-assets` inside `02-reference-data/`.
- `job_scheduler`
  Either `local` or `slurm`.
- `cp2k_cmd`
  CP2K executable used for local execution. This should be a single executable
  name or path such as `cp2k.psmp`.
- `single_atoms`
  Whether to also run the isolated single-atom reference jobs. Defaults to
  `true`.
- `snapshot_md_steps`
  Optional override for the number of short GFN1-xTB MD steps run for each
  snapshot. If omitted, `bff reference` uses the staged `snapshots/md.inp`
  from `bff prepare` unchanged.
- `train_fraction`
  Fraction of collected snapshot frames written into `train.extxyz`. The rest
  goes to `valid.extxyz`.
- `seed`
  Deterministic shuffle seed used before the train/validation split.
- `systems`
  Non-empty list of prepared reference-asset directories.
- `slurm`
  Required when `job_scheduler: slurm`.

## `systems[]` Keys

- `assets`
  Path to one prepared reference system directory such as
  `../01-prepare/colvars/reference/system-000/`.
- `md`
  Optional CP2K MD input for this system's short snapshot relaxations. If
  omitted, `bff reference` uses `assets/snapshots/md.inp`.
- `sp`
  Optional CP2K single-point input for this system's final energy/force
  calculation. If omitted, `bff reference` uses `assets/snapshots/sp.inp`.

Each referenced system must already contain:

- `system.top`
- `system.gro`
- `system.xyz`
- `snapshots/md.inp`, unless `systems[].md` is provided
- `snapshots/sp.inp`, unless `systems[].sp` is provided
- `snapshots/xyz/snapshot-*.xyz`
- `single-atoms/*/input.inp`
- `single-atoms/*/pos.xyz`

Those are written by `bff prepare`.

## `slurm` Keys

The `slurm` block intentionally matches the simulation workflow shape:

- `slurm.max_parallel_jobs`
  Maximum number of simultaneously active jobs. Use `-1` for no client-side
  throttling.
- `slurm.sbatch`
  Mapping of `sbatch` options such as `partition`, `time`, `mem`,
  `ntasks_per_node`, or `cpus_per_task`.
- `slurm.setup`
  Optional shell commands inserted before each batch job.
- `slurm.teardown`
  Optional shell commands inserted after each batch job.

## Outputs

For each configured reference system, `bff reference` writes or refreshes:

- `snapshots/snapshot-XXXX/`
  Per-snapshot run directories with staged `md.inp`, `sp.inp`, `pos.xyz`, and
  the resulting `sp.extxyz`.
- `train.extxyz`
- `valid.extxyz`

If `single_atoms: true`, it also writes:

- `single-atoms/energies.yaml`

[reference-configs]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/workflows/configs.py
[reference-workflow]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/workflows/reference.py
