# Sample Configuration

Source code:

- `bff/workflows/sample/config.py`
- `bff/workflows/sample/main.py`
- `bff/workflows/_shared/campaign.py`

## Purpose

`bff sample` draws parameter vectors, stages a sampled FFMD campaign, and runs
the corresponding GROMACS jobs.

## Minimal Example

```yaml
mol_resname: ACE
campaign_dir: ./
systems:
  - assets: ../02-assets/ffmd/system-000
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
- `campaign_dir`
  Output directory for sampled topologies, metadata, and trajectories.
- `systems`
  Non-empty list of staged asset directories plus system-specific MD lengths.
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
  Which trajectory outputs to keep. Defaults to `['xtc']`.
- `slurm`
  Slurm-only runtime settings.

## `systems[]` Keys

- `assets`
  Directory created by `bff prepare-assets`, for example `ffmd/system-000`.
- `n_steps`
  Production MD length for this prepared system within the sampled campaign.

## Outputs

`bff sample` writes:

- `campaign_dir/specs.yaml`
- `campaign_dir/samples.yaml`
- per-sample job configs
- per-sample modified topologies
- stored trajectories and energy files
