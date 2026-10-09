# Validate Configuration

Source code:

- `bff/workflows/validate/config.py`
- `bff/workflows/validate/main.py`
- `bff/workflows/campaign/`

## Purpose

`bff validate` reruns selected parameter samples, usually drawn from the
posterior learned by `bff learn`.

The campaign runtime keys intentionally match `bff sample-parameters` as closely as
possible.

## Posterior Mode

```yaml
campaign_dir: ./
posterior:
  file: ../06-learn/outputs/results.pt
  n_samples: 10
  include_mean: true
  include_map: true
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

The specification stored in the results is authoritative and is copied to
`campaign_dir/specs.yaml`. `n_samples` counts random draws; `include_mean` and
`include_map` prepend the posterior mean and the MAP (the sampled state of
highest log posterior) as deterministic samples. Supported distributions are
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

Before anything is staged, validation checks that every parameter sample
satisfies the bounds once implicit charges are reconstructed, and that every
charge constraint gives the same equation in the validation systems as in the
systems the specification was compiled for. Validation systems may lack some
parameters or constraints, for example a system without calcium, but a
constraint that selects atoms must still reconstruct the learned charges.

## Options

General rules for all options are on the
[conventions page](index.md). Set exactly one of `posterior` and
`parameters`.

### Validation options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `posterior` | mapping | none | Draw parameters from a learned posterior; see [posterior](#posterior). |
| `parameters` | path | none | YAML file with explicit parameter samples; see [the format](#explicit-parameter-file-format). |
| `specs` | path | *required with `parameters`* | Parameter specification for `parameters`; not allowed with `posterior`, which embeds its own. |

### `posterior`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `posterior.file` | path | *required* | `outputs/results.pt` written by `bff learn`. |
| `posterior.n_samples` | integer >= 0 | `10` | Random draws. |
| `posterior.include_mean` | boolean | `false` | Prepend one deterministic sample at the posterior mean. |
| `posterior.include_map` | boolean | `false` | Prepend one deterministic sample at the MAP, after the mean. At least one sample must result from `n_samples`, `include_mean`, and `include_map`. |
| `posterior.distribution` | `empirical`, `normal`, `uniform`, or `kde` | `normal` | `empirical` resamples posterior samples, `normal` draws from a fitted multivariate normal, `uniform` from the per-parameter central `confidence` interval, and `kde` from a Gaussian KDE. |
| `posterior.confidence` | number in (0, 1) | `0.9` | Central interval width; used only by `uniform`. |
| `posterior.seed` | integer | none | Random seed; omitted means fresh randomness. |

--8<-- "campaign-options.md"

## Explicit Parameter File Format

The explicit mode consumes YAML only. The expected structure is a mapping from
explicit parameter name to a list of sampled values; names not in the
specification are rejected:

```yaml
charge C1: [-0.5, -0.4, -0.3]
charge O1 O2: [-0.7, -0.6, -0.5]
```

Implicit charges are reconstructed from `specs.yaml`, so they do not need to
appear in the file; columns for them, as written by
`results.draw(..., implicit=True)`, are ignored.

Validation campaigns use the same layout as sampling: each
`samples/<sample_id>/` holds `run.out` and `gmx.log`,
with one directory per system below it.
Locally dispatched campaigns show console-only progress from `0/N` while the
first MD job is running and advance after each completed parameter sample.

For the alternative explicit workflow, load `outputs/results.pt` with
`Results.load` and call `results.draw(10, fn_out="posterior-samples.yaml")`.
Posterior mode performs the same parameter ordering, bounds enforcement, and
implicit-charge reconstruction automatically.

## Analyzing a Validation Campaign

A validation campaign has the same layout as a sampling campaign: it always
contains `specs.yaml` (copied from `specs` or from the posterior) next to
`samples.yaml`, so `bff build-qoi-datasets` can analyze it like a sampling
campaign, for example to compare validation QoIs with the reference:

```yaml
training_samples:
  manifest: ../07-validate/samples.yaml
  systems:
    - system_id: acetate
```

`samples.yaml` records the parameter source under `provenance`: `source:
posterior` with the results path and SHA-256, `distribution`, `confidence`,
`n_samples`, `include_mean`, `include_map`, and the `seed` actually used, or `source:
parameters` with the parameter file, its SHA-256, and `specs`.
