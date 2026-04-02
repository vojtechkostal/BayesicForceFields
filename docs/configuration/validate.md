# Validate Configuration

Source code:

- `bff/workflows/configs.py`
- `bff/workflows/validate.py`
- `bff/workflows/runsims.py`

## Purpose

`bff validate` reruns selected parameter samples, usually drawn from the
posterior learned by `bff learn`.

The campaign runtime keys intentionally match `bff simulate` as closely as
possible.

## Minimal Example

```yaml
trainset_dir: ./validation-trainset
parameters: ../06-learn/posterior-samples.yaml
specs: ../03-training-trjs/trainset/specs.yaml
systems:
  - assets: ../01-prepare/colvars/ace-colvars/training/window-000
    n_steps: 1000
  - assets: ../01-prepare/colvars/ace-colvars/training/window-001
    n_steps: 1000
gmx_cmd: gmx
job_scheduler: local
```

## Top-Level Keys

- `trainset_dir`
  Output directory for the validation campaign.
- `parameters`
  YAML file containing explicit parameter samples.
- `specs`
  Force-field specification file used to reconstruct constrained parameters.
- `systems`
  Non-empty list of prepared asset directories plus validation MD lengths.
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
  Optional Slurm runtime configuration.

## `systems[]` Keys

- `assets`
  Directory created by `bff prepare`, for example `training/window-001`.
- `n_steps`
  Production MD length for this validation run.

## Parameter File Format

Validation consumes YAML only. The expected structure is a mapping from
explicit parameter name to a list of sampled values:

```yaml
charge C1: [-0.5, -0.4, -0.3]
charge O1 O2: [-0.7, -0.6, -0.5]
```

The implicit charge is reconstructed from `specs.yaml`, so it does not need to
appear in the file.
