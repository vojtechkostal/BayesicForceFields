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
campaign_dir: ./
systems:
  - assets: ../02-assets/ffmd/system-000
    n_steps: 1000
bounds:
  charge C2: [0.0, 1.0]
  charge O1 O2: [-0.8, -0.3]
charge_constraints:
  - selection: "resname ACE"
    target: -0.8
    scope: residue
    implicit: "charge C2"
n_samples: 10
gmx_cmd: gmx
job_scheduler: local
```

## Top-Level Keys

- `campaign_dir`
  Output directory for sampled topologies, metadata, and trajectories.
- `systems`
  Non-empty list of staged asset directories plus system-specific MD lengths.
- `bounds`
  Mapping from parameter label to lower and upper bounds.
- `charge_constraints`
  Charge equations compiled from MDAnalysis selections. Each constraint defines
  `selection`, `target`, `scope`, and a distinct bounded `implicit` parameter.
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

## `charge_constraints[]` Keys

- `selection`
  MDAnalysis atom selection. Selections must be disjoint or strictly nested;
  partial overlaps are rejected.
- `target`
  Required total charge for the selected group.
- `scope`
  Either `system` for one complete-system sum or `residue` to apply the target
  independently to each selected residue.
- `implicit`
  Parameter reconstructed to satisfy this equation. It must exist in `bounds`,
  belong to this selection, and not occur in a descendant selection.

Nested constraints are reconstructed from the smallest selected groups outward.

## Outputs

`bff sample` writes:

- `campaign_dir/specs.yaml`
- `campaign_dir/samples.yaml`
- per-sample job configs
- per-sample modified topologies
- stored trajectories and energy files
