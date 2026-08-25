# Validate Configuration

Source code:

- `bff/workflows/validate/config.py`
- `bff/workflows/validate/main.py`
- `bff/workflows/_shared/campaign.py`

## Purpose

`bff validate` reruns selected parameter samples, usually drawn from the
posterior learned by `bff learn`.

The campaign runtime keys intentionally match `bff sample-parameters` as closely as
possible.

## Posterior Mode

```yaml
campaign_dir: ./
posterior:
  file: ../06-learn/outputs/posterior.pt
  n_samples: 10
  include_mean: true
  distribution: normal
  confidence: 0.9
  seed: null
source: ../01-build
systems:
  - system_id: acetate
    n_steps: 1000
gmx_cmd: gmx
job_scheduler: local
```

The posterior's embedded specifications are authoritative and are copied to
`campaign_dir/specs.yaml`. `n_samples` counts random draws; `include_mean`
prepends one additional deterministic sample. Supported distributions are
`empirical`, `normal`, `uniform`, and `kde`. A null or omitted seed uses
fresh randomness.

## Explicit Mode

```yaml
campaign_dir: ./
parameters: ./selected-parameters.yaml
specs: ../06-learn/outputs/specs.yaml
source: ../01-build
systems:
  - system_id: acetate
    n_steps: 1000
gmx_cmd: gmx
job_scheduler: local
```

Exactly one of `parameters` or `posterior` is required. `specs` is required
only with `parameters` and is rejected with `posterior`.

## Top-Level Keys

- `campaign_dir`
  Output directory for the validation campaign.
- `parameters`
  YAML file containing explicit parameter samples.
- `posterior`
  Learned posterior source and draw settings. Strict keys are `file`,
  `n_samples`, `include_mean`, `distribution`, `confidence`, and `seed`.
  At least one random or mean sample must be requested.
- `specs`
  Force-field specification file used only with explicit parameter samples.
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
  If `true`, retain only extensions listed in `store` inside each system result
  directory and delete `outputs/` after successful collection. If `false`,
  retain all generated files.
- `store`
  File extensions to retain, without leading dots. Defaults to `['xtc']`.
  Working-directory outputs are moved into the system's sample directory and
  recorded as named manifest inputs.
- `slurm`
  Optional Slurm runtime configuration.

## `systems[]` Keys

- `system_id`
  Stable ID selected from the build-stage `source`.
- `inputs`
  Alternatively, the explicit FFMD roles documented for `bff sample-parameters`.
- `n_steps`
  Production MD length for this validation run.

## Explicit Parameter File Format

The explicit mode consumes YAML only. The expected structure is a mapping from
explicit parameter name to a list of sampled values:

```yaml
charge C1: [-0.5, -0.4, -0.3]
charge O1 O2: [-0.7, -0.6, -0.5]
```

Implicit charges are reconstructed from `specs.yaml`, so they do not need to
appear in the file.

Operational files use the same layout as sampling. The job `config.yaml`,
`run.sh`, and Slurm `run.out` are copied to `samples/<sample_id>/`. The live
`outputs/` working tree is retained only when cleanup is disabled; requested
validation data remain in per-system directories below `samples/`.

For the alternative explicit workflow, load `outputs/posterior.pt` with
`PosteriorResults` and call
`sample_posterior(..., fn_out="outputs/posterior-samples.yaml")`. Posterior
mode performs the same preparation, nuisance exclusion, parameter ordering,
bounds enforcement, and implicit-charge reconstruction automatically.
