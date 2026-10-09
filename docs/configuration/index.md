# Settings Overview

Every stage reads one YAML file:

```bash
bff <stage> config.yaml
```

This page lists all groups of settings; each link leads to the full option
table. The [conventions](#conventions) below apply to every stage.

## [`bff build`](build.md)

- [`project`, `gromacs`](build.md#options): output directory, log, GROMACS executable.
- [`systems[]`](build.md#systems): per system its ID, topology, coordinate
  templates, box, MDP files, equilibration and production steps, and an
  optional Colvars or PLUMED bias.

## [`bff sample-parameters`](sample-parameters.md)

- [`bounds`, `n_samples`, `seed`](sample-parameters.md#sampling-options): which
  parameters are sampled, in which ranges, and how many sets.
- [Parameter labels](sample-parameters.md#parameter-labels): the syntax for
  charges, Lennard-Jones sigma and epsilon, and dihedrals.
- [`charge_constraints[]`](sample-parameters.md#charge_constraints): total
  charges that fix one charge parameter.
- [Campaign options](sample-parameters.md#campaign-options): `campaign_dir`,
  `source`, `gmx_cmd`, `job_scheduler`, `store`, `cleanup`, `scratch_dir`,
  `max_restarts`, `resume`, `overwrite`.
- [`systems[]`](sample-parameters.md#systems): which built systems to simulate,
  and for how many steps.
- [`slurm`](sample-parameters.md#slurm-options): job arrays, `sbatch` options,
  setup and teardown lines; see also
  [restarts after the time limit](sample-parameters.md#time-limits-and-restarts).

## [`bff build-qoi-datasets`](build-qoi-datasets.md)

- [`training_samples`](build-qoi-datasets.md#training_samples): the campaign,
  its systems, analyzed frames, and parallel workers.
- [`reference`](build-qoi-datasets.md#reference): per system the reference
  files (trajectory, or files such as a PMF).
- [`routines[]`](build-qoi-datasets.md#routines): the QoIs; built-in
  [`rdf` and `hydrogen_bonds`](build-qoi-datasets.md#built-in-routines) or
  [your own function](build-qoi-datasets.md#routine-interface).
- [`run`, `output`](build-qoi-datasets.md#run-and-output): in-memory analysis
  and the output directory.

## [`bff fit-lgp`](fit-lgp.md)

- [`datasets.<name>`](fit-lgp.md#datasetsname): per QoI its dataset, the
  [surrogate mean](fit-lgp.md#surrogate-mean), and an optional fixed noise.
- [`fit`](fit-lgp.md#fit): model directory, model reuse, committee size,
  test fraction, and the hyperparameter search.

## [`bff learn`](learn.md)

- [`specs`, `models.<name>`](learn.md#modelsname): the surrogates and the
  [tolerance](learn.md#likelihood) per QoI.
- [`mcmc`](learn.md#mcmc): steps, warmup, walkers, prior, convergence, resume,
  device.
- [`plots`](learn.md#plots), [`output`](learn.md#output): plot sampling and labels,
  output directory.

## [`bff validate`](validate.md)

- [`posterior`](validate.md#posterior) or `parameters`: which parameter sets
  to simulate.
- All [campaign options](validate.md#campaign-options) of `sample-parameters`.

## Conventions

- **Unknown keys are errors.** A misspelled key is reported, never ignored.
- **`null` means omitted.** An option set to `null` takes its default; a
  required option set to `null` is reported as missing.
- **Paths are relative to the config file**, not to the working directory,
  and `~` is expanded. Input files must exist when the config is loaded;
  output directories and logs are created as needed.
- **Integers must be whole numbers** and **numbers must be finite**; `1e-3`
  and `1e5` work although YAML reads them as strings.
- **Booleans are `true` or `false`**, not `yes`, `"false"`, or `0`.
- **`-1` means unlimited** where an option allows it.
- **IDs** (`system_id`, routine, dataset, and model names) match
  `[a-z0-9][a-z0-9._-]*` and are unique within their list.
- **Devices** are `cpu`, `cuda`, `cuda:<index>`, or `mps`.

Errors name the offending key by its full path, for example
`systems[0].nsteps.prod must be an integer >= 1, got 0.` In the option tables,
*required* marks options without a default, and nested keys are written with
dots: `mcmc.warmup` is `warmup` inside `mcmc:`.
