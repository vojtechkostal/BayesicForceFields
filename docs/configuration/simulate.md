# Simulate Configuration

Source code:

- `bff/workflows/configs.py`
- `bff/workflows/simulate.py`
- `bff/workflows/runsims.py`

## Purpose

`bff simulate` samples parameter vectors, writes a trainset directory, and runs
the corresponding GROMACS training campaign.

## Minimal Example

```yaml
mol_resname: ACE
trainset_dir: ./trainset
systems:
  - assets: ../01-prepare/colvars/ace-colvars/training/system-000
    n_steps: 1000
  - assets: ../01-prepare/colvars/ace-colvars/training/system-001
    n_steps: 1000
bounds:
  charge C2: [0.0, 1.0]
  charge O1 O2: [-0.8, -0.3]
implicit_atoms: [C2]
total_charge: -0.8
n_samples: 10
gmx_cmd: gmx
job_scheduler: local
```

## Top-Level Keys

- `mol_resname`
  Residue name of the parameterized molecule in the GROMACS topology.
- `trainset_dir`
  Output directory for sampled topologies, metadata, and trajectories.
- `systems`
  Non-empty list of prepared asset directories plus system-specific MD lengths.
- `bounds`
  Mapping from parameter label to lower and upper bounds.
- `implicit_atoms`
  Atom group whose charge is determined by the total-charge constraint.
- `total_charge`
  Target total molecular charge used when building `specs.yaml`.
- `n_samples`
  Number of force-field vectors sampled for the campaign.
- `gmx_cmd`
  GROMACS executable.
- `job_scheduler`
  Either `local` or `slurm`.
- `dispatch`
  If `true`, launch jobs immediately after staging them.
- `compress`
  If `true`, compress finished simulation outputs.
- `cleanup`
  If `true`, remove temporary files after successful runs.
- `store`
  Which trajectory outputs to keep. Defaults to `["xtc"]`.
- `slurm`
  Slurm-only runtime settings.

## `systems[]` Keys

- `assets`
  Directory created by `bff prepare`, for example
  `training/system-000`.
- `n_steps`
  Production MD length for this prepared system within the sampled campaign.

## `slurm` Keys

- `slurm.max_parallel_jobs`
  Maximum number of submitted jobs. `-1` means no client-side limit.
- `slurm.sbatch`
  Mapping passed to the submission script generator.
- `slurm.setup`
  Shell commands executed before the BFF MD job.
- `slurm.teardown`
  Shell commands executed after the BFF MD job.

## Outputs

`bff simulate` writes:

- `trainset/specs.yaml`
- `trainset/samples.yaml`
- per-sample job configs
- per-sample modified topologies
- stored trajectories and energy files
