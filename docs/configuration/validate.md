# Validate Configuration

Source code:

- `bff/workflows/validate/config.py`
- `bff/workflows/validate/main.py`
- `bff/workflows/_shared/campaign.py`

## Purpose

`bff validate` reruns selected parameter samples, usually drawn from the
posterior learned by `bff learn`.

The campaign runtime keys intentionally match `bff sample` as closely as
possible.

## Minimal Example

```yaml
campaign_dir: ./
parameters: ../06-learn/posterior-samples.yaml
specs: ../03-sample/specs.yaml
source: ../01-build
systems:
  - system_id: acetate
    n_steps: 1000
gmx_cmd: gmx
job_scheduler: local
```

## Top-Level Keys

- `campaign_dir`
  Output directory for the validation campaign.
- `parameters`
  YAML file containing explicit parameter samples.
- `specs`
  Force-field specification file used to reconstruct constrained parameters.
- `systems`
  Non-empty list of build-stage system IDs plus validation MD lengths, or systems
  with fully explicit FFMD role mappings.
- `source`
  Build stage root containing the selected system directories. Do not combine
  it with explicit per-system `inputs`.
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
  Optional Slurm runtime configuration.

## `systems[]` Keys

- `system_id`
  Stable ID selected from the build-stage `source`.
- `inputs`
  Alternatively, the explicit FFMD roles documented for `bff sample`.
- `n_steps`
  Production MD length for this validation run.

## Parameter File Format

Validation consumes YAML only. The expected structure is a mapping from
explicit parameter name to a list of sampled values:

```yaml
charge C1: [-0.5, -0.4, -0.3]
charge O1 O2: [-0.7, -0.6, -0.5]
```

Implicit charges are reconstructed from `specs.yaml`, so they do not need to
appear in the file.

The fixed learning artifact is `output/posterior.pt`. Load it with
`PosteriorResults`, prepare/select the desired draws, and call
`sample_posterior(..., fn_out="posterior-samples.yaml")` to create validation
input in the format above.
