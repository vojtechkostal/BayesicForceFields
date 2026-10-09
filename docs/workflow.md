# How BFF Works

BFF is a chain of stages. Each stage is one command that reads one YAML
config and writes files the next stage reads, so every step can be inspected,
rerun, or replaced.

## Stages

| Stage | Reads | Writes |
| --- | --- | --- |
| [`bff build`](configuration/build.md) | topologies, coordinate templates, MDP files, optional Colvars/PLUMED bias | `systems/<system_id>/` with equilibrated inputs and a virtual-site-free `reference/` topology and coordinates |
| reference MD (yours) | `systems/<system_id>/reference/` | reference trajectories, or files such as a PMF |
| [`bff sample-parameters`](configuration/sample-parameters.md) | `systems/`, parameter `bounds` | `specs.yaml`, `samples.yaml`, MD outputs per sample |
| [`bff build-qoi-datasets`](configuration/build-qoi-datasets.md) | `samples.yaml`, reference data, QoI routines | `qoi/<name>.pt`, one dataset per QoI |
| [`bff fit-lgp`](configuration/fit-lgp.md) | QoI datasets | `models/<name>.lgp`, one surrogate per QoI |
| [`bff learn`](configuration/learn.md) | surrogates, `specs.yaml` | `outputs/results.pt` (posterior), `plots/` |
| [`bff validate`](configuration/validate.md) | `results.pt` or explicit parameters, `systems/` | a new MD campaign for the chosen parameters |

## A project

A project is one directory per stage; paths in a config are relative to it.
The [acetate example](examples/acetate.md) looks like this:

```text
acetate/
├── inputs/                    topologies, force field, MDP files, bias, QoI routine
├── 01-build/                  config.yaml → systems/<system_id>/
├── 02-reference-md/           your reference data
├── 03-sample-parameters/      config.yaml → specs.yaml, samples.yaml, samples/
├── 04-build-qoi-datasets/     config.yaml → qoi/<name>.pt
├── 05-fit-lgp/                config.yaml → models/<name>.lgp
├── 06-learn/                  config.yaml → outputs/results.pt, plots/
└── 07-validate/               config.yaml → a validation campaign
```

## Rules that hold everywhere

- **Systems are paired by `system_id`**, never by list order or display name.
- **The reference must match the build.** A reference trajectory has the same
  atoms in the same order as `systems/<system_id>/reference/`. See
  [Reference data](reference-trajectories.md).
- **Configs are strict.** Unknown keys and invalid values are errors that
  name the offending key; see the [settings overview](configuration/index.md).
- **Long runs can be continued.** A campaign continues with `resume: true`
  (Slurm samples stopped by the time limit continue from their checkpoints,
  automatically with `max_restarts`), and `bff learn` with `mcmc.resume: true`.

## Main files

| File | Content |
| --- | --- |
| `specs.yaml` | Parameter names, bounds, and the charge equations that fix implicit charges. |
| `samples.yaml` | Every sample's parameters, status (`completed`, `incomplete`, `failed`), and output files per system. |
| `samples/<sample_id>/` | The sample's MD: `run.out`, `gmx.log`, and one directory per system. |
| `qoi/<name>.pt` | A `QoIDataset`: parameters `X`, sample QoIs `y`, reference `y_ref`. |
| `models/<name>.lgp` | The surrogate of one QoI. |
| `outputs/results.pt` | Prior, posterior chain, MAP, and per-QoI likelihoods; open it with `bff.Results.load`. See [Results](results.md). |

## Without MD

The inference part works on any data: a `QoIDataset` built from your own
tables can be fitted and learned from Python. See the
[arbitrary-data example](examples/arbitrary-data.md).
