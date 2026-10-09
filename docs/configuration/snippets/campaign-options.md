### Campaign options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `campaign_dir` | path | *required* | Output directory of the campaign; created if missing. An existing campaign in it must be resumed or overwritten; see [Rerunning a campaign](#rerunning-a-campaign). |
| `log` | path | `<campaign_dir>/<stage>.log` | Workflow log file. |
| `gmx_cmd` | string | *required* | GROMACS executable, for example `gmx` or `gmx_mpi`. |
| `job_scheduler` | `local` or `slurm` | *required* | Run samples on this machine (`local.max_parallel_jobs` at once) or as Slurm job arrays. |
| `source` | path | none | `bff build` project directory. With it, `systems[]` lists only IDs and lengths, and each `systems/<system_id>/` must contain `topology.top`, `production.gro`, `index.ndx`, `em.mdp`, `production.mdp`, and any `bias.*.dat`; the build trajectory is not needed. Without it, every system gives explicit `inputs`. |
| `systems` | list | *required* | Systems simulated for every sample; see [systems](#systems). |
| `dispatch` | boolean | `true` | Run the jobs after staging them. With `false`, stage only (and print the `sbatch` command on Slurm). |
| `compress` | boolean | `false` | Pack `.tpr`, `.xtc`, `.yaml`, `.top`, and `.gro` files of the finished campaign into `<campaign_dir>.tar.gz`. |
| `cleanup` | boolean | `false` | Once a system's production run is complete, keep only the `store` suffixes and the sample's topology in its result directory. Unfinished systems keep everything a restart needs. |
| `store` | list of strings | `[xtc]` | File suffixes, without dots, kept by `cleanup` and recorded as sample outputs (for example `[xtc, pmf]`). |
| `scratch_dir` | string | none | Directory, usually node-local, in which GROMACS runs; may contain `$VARIABLES` expanded on the compute node. See [Scratch directory](sample-parameters.md#scratch-directory). |
| `max_restarts` | integer >= 0 | `0` | Slurm only (an error with `local`): how often samples stopped by the time limit are resubmitted. See [Time limits and restarts](sample-parameters.md#time-limits-and-restarts). |
| `overwrite` | boolean | `false` | Delete an existing campaign in `campaign_dir` (only the files the campaign owns) and start a new one. |
| `resume` | boolean | `false` | Continue the existing campaign: reuse its parameter samples and run only samples not yet completed. Cannot be combined with `overwrite`. |
| `local.max_parallel_jobs` | integer >= 1 | `1` | With `job_scheduler: local`, samples run at once; mind the CPU/GPU use of each GROMACS run. |
| `slurm` | mapping | none | Required with `job_scheduler: slurm`; see [slurm](#slurm-options). |

### `systems[]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `system_id` | ID | *required* | System ID; with `source`, the directory `systems/<system_id>/` of the build. |
| `n_steps` | integer >= 1 | *required* | Production MD steps per sample. |
| `inputs` | mapping | *required without `source`* | Explicit input files; not allowed with `source`. |
| `inputs.topology` | path | *required* | GROMACS topology. |
| `inputs.coordinates` | path | *required* | Starting coordinates. |
| `inputs.index` | path | *required* | GROMACS index file. |
| `inputs.mdp_production` | path | *required* | Production MDP file. |
| `inputs.mdp_em` | path | none | Energy-minimization MDP file; minimization is skipped without it. |
| `inputs.bias` | path | none | Bias input named `*.colvars.dat` or `*.plumed.dat`. |

### `slurm` options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `slurm.sbatch` | mapping | *required* | `sbatch` options written as `#SBATCH --<key>=<value>` (underscores become dashes). Must not set `array`. Quote `time` (`"04:00:00"`): YAML reads an unquoted `4:00:00` as a number. |
| `slurm.max_parallel_jobs` | integer >= 1 or -1 | `1` | Array tasks running at once (`%` limit); `-1` for no limit. |
| `slurm.max_array_size` | integer >= 1 | `1000` | Tasks per submitted array; keep below the cluster's `MaxArraySize`. |
| `slurm.setup` | list of strings | `[]` | Shell lines run before each task, such as `module load gromacs`. |
| `slurm.teardown` | list of strings | `[]` | Shell lines run after each task, also when it failed. |

### Rerunning a campaign

A campaign directory holds `specs.yaml`, `samples.yaml`, `campaign.yaml`,
`run.sh`, `tasks.txt`, `systems/`, `samples/`, and `slurm/`. Running the stage again
on a directory that contains them is an error, because new parameter draws
would otherwise be mixed with old results:

- `resume: true` continues the campaign, for example after the submitting
  process was killed or samples failed. It reuses the staged parameter samples,
  reruns only samples whose status is not `completed`, and appends to the log.
  The specification, system IDs, and `n_steps` must be unchanged; runtime
  options such as `gmx_cmd`, `slurm`, or `scratch_dir` may change.
- `overwrite: true` deletes those campaign files (nothing else in the
  directory) and starts a new campaign.
