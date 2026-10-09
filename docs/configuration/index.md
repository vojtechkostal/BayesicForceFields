# Configuration Conventions

Every BFF stage reads one YAML file, passed on the command line:

```bash
bff <stage> config.yaml
```

The pages in this section list every option of every stage. All stage
configurations follow the same rules:

- **Unknown keys are errors.** A misspelled key is reported, never ignored.
- **`null` means omitted.** An option set to `null` takes its default; a
  required option set to `null` is reported as missing.
- **Paths are relative to the configuration file**, not to the working
  directory. `~` is expanded. Input files must exist when the configuration is
  loaded; output directories and logs are created as needed.
- **Integers must be whole numbers** and **numbers must be finite**. Both may
  be written in exponent notation; `1e-3` and `1e5` work although YAML reads
  them as strings.
- **Booleans are `true` or `false`**, not `yes`, `"false"`, or `0`.
- **`-1` means unlimited** where an option allows it (`workers`,
  `slurm.max_parallel_jobs`, `plots.max_marginal_samples`).
- **IDs** (`system_id`, routine, dataset, and model names) match
  `[a-z0-9][a-z0-9._-]*` and must be unique within their list.
- **Devices** are `cpu`, `cuda`, `cuda:<index>`, or `mps`.

Errors name the offending key by its full path, for example
`systems[0].nsteps.prod must be an integer >= 1, got 0.`

In the option tables, *required* marks options without a default, and nested
keys are written with dots, so `mcmc.warmup` is `warmup` inside `mcmc:`.

| Stage | Command | Options |
| --- | --- | --- |
| Build | `bff build` | [build](build.md#options) |
| Sample parameters | `bff sample-parameters` | [sample-parameters](sample-parameters.md#options) |
| Build QoI datasets | `bff build-qoi-datasets` | [build-qoi-datasets](build-qoi-datasets.md#options) |
| Fit LGP | `bff fit-lgp` | [fit-lgp](fit-lgp.md#options) |
| Learn | `bff learn` | [learn](learn.md#options) |
| Validate | `bff validate` | [validate](validate.md#options) |
