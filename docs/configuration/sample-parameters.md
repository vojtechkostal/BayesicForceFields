# Sample Parameters Configuration

Source code:

- `bff/workflows/sample_parameters/config.py`
- `bff/workflows/sample_parameters/main.py`
- `bff/workflows/campaign/` (staging, running, and the per-sample job)

## Purpose

`bff sample-parameters` draws parameter vectors, stages a sampled FFMD campaign, and runs
the corresponding GROMACS jobs.

## Minimal Example

```yaml
campaign_dir: ./
source: ../01-build
systems:
  - system_id: acetate
    n_steps: 1000
bounds:
  charge C2: [0.0, 1.0]
  charge O1 O2: [-0.8, -0.3]
charge_constraints:
  - selection: "resname ACE"
    target: -0.8
    scope: residue
    implicit: C2
n_samples: 10
gmx_cmd: gmx
job_scheduler: local
```

## Options

General rules for all options are on the
[conventions page](index.md).

### Sampling options

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `bounds` | mapping | *required* | Parameter label to `[lower, upper]` with finite `lower < upper`; see [Parameter labels](#parameter-labels). |
| `n_samples` | integer >= 1 | *required* | Number of parameter vectors, drawn as one Latin hypercube. |
| `seed` | integer >= 0 | none | Seed of the Latin hypercube. Without it a fresh seed is drawn; either way the seed is recorded in `samples.yaml`, so the draw can be repeated. |
| `charge_constraints` | list | `[]` | Charge equations; see [charge_constraints](#charge_constraints). |

--8<-- "campaign-options.md"

### `charge_constraints[]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `selection` | string | *required* | MDAnalysis selection of the group whose total charge is fixed. |
| `target` | number | *required* | Total charge of the group. |
| `scope` | `residue` or `system` | *required* | `residue`: every selected residue on its own has the total charge `target`. `system`: all selected atoms of a system together have it. |
| `implicit` | string | *required* | Atom name or type, inside the group, whose charge parameter is computed from the constraint instead of being sampled. It must belong to exactly one `charge ...` parameter in `bounds`, and each constraint needs a different one. |

## Charge Constraints

A charge constraint fixes the total charge of a group of atoms. Its implicit
charge parameter is not sampled; it is computed so that the group has exactly
the target charge. Atoms that no `charge ...` parameter controls keep their
topology charge.

Each selected residue of an acetate solution carries a charge of -0.8; `C2` is
computed from the sampled `C1`, `O1 O2`, and `H1 H2 H3` charges:

```yaml
bounds:
  charge C1: [-0.5, -0.1]
  charge C2: [0.4, 1.0]
  charge O1 O2: [-0.9, -0.5]
  charge H1 H2 H3: [0.0, 0.2]
charge_constraints:
  - selection: "resname ACE"
    target: -0.8
    scope: residue
    implicit: C2
```

With `scope: system`, the constraint holds for the selected atoms of a system
as a whole. For a lithium chloride solution in which only the lithium charge is
sampled, the chloride charge follows from requiring that all ion pairs are
neutral together:

```yaml
bounds:
  charge LI: [0.6, 1.0]
  charge CL: [-1.0, -0.6]
charge_constraints:
  - selection: "resname LI CL"
    target: 0.0
    scope: system
    implicit: CL
```

Every constraint becomes one linear equation in the charge parameters: the
number of group atoms each parameter controls, plus the fixed charge of the
other group atoms, equals `target`. The equation must be the same in every
configured system (and every residue, for `scope: residue`); with 1000 or 10
ion pairs, `system` scope gives the same equation. The implicit charges of all
constraints are solved together, so constraints may overlap or depend on each
other as long as they determine their implicit charges uniquely. The compiled
equations are written to `specs.yaml`.

## Biased systems

When a system carries a Colvars bias, BFF copies the bias file into each
`samples/<sample_id>/<system_id>/` run directory and writes a
`production-run.mdp`. Its `colvars-configfile` value is generated relative
to the run directory, so users do not need to edit paths for local or Slurm
campaigns.

## Slurm

With `job_scheduler: slurm`, all samples run from one script, `run.sh`, as
Slurm job arrays. Each array task runs one sample; its output goes to
`samples/<sample_id>/run.out`, and Slurm's own messages (for example time-limit
cancellations) go to `slurm/<job>_<task>.out`.

```yaml
slurm:
  max_parallel_jobs: 200   # tasks running at once (array %limit); -1 = no limit
  max_array_size: 1000     # tasks per submitted array, default 1000
  sbatch: {time: "00:40:00", mem: 1G, cpus_per_task: 1}
  setup: [module load gromacs]
  teardown: []
```

Campaigns larger than `max_array_size` are submitted as consecutive arrays;
the next array is submitted once the previous one has finished. Keep
`max_array_size` below the cluster's `MaxArraySize` and per-user submit limit.
`sbatch` must not set `array`. Array task `i` runs the sample on line `i + 1`
of `tasks.txt`. With `dispatch: false`, BFF stages the campaign and prints the
`sbatch --array=...` command for manual submission.

### Scratch directory

Hundreds of simultaneous MD runs writing to a shared filesystem can overload
it. With `scratch_dir`, each sample runs GROMACS in a fresh directory below
`scratch_dir` and copies its files back to `samples/<sample_id>/<system_id>/`
when a system finishes: only the `store` suffixes when `cleanup: true`,
everything otherwise or when the run stopped early. The scratch copy is then
removed.

```yaml
scratch_dir: $SLURM_TMPDIR    # or $TMPDIR, /scratch/$USER, ...
```

`$VARIABLES` and `~` are expanded on the compute node. If a variable is not
defined there, the sample runs in its campaign directory and `run.out` says so.

### Time limits and restarts

When `slurm.sbatch.time` is set, every production run gets `mdrun -maxh` with
90% of that limit minus the time the sample has already used, so GROMACS stops
cleanly and writes a checkpoint before Slurm would cancel the task. Such a
sample is reported as `incomplete`. With `max_restarts: N`, BFF resubmits the
incomplete samples, as a new throttled job array, up to `N` times:

```yaml
max_restarts: 3
slurm:
  sbatch: {time: "04:00:00"}
```

A rerun sample skips systems whose trajectory is complete and continues an
interrupted production run from `production.cpt`, appending to its output
files (`run.out` and `gmx.log` are appended as well). The total step count is
written into the `.tpr` (via the generated `production-run.mdp`), so a
continued run stops exactly at `n_steps`. Colvars biases continue from the
checkpoint; PLUMED biases must be restartable by PLUMED itself (keep `HILLS`
next to the run). Samples still incomplete after the last restart are recorded
with status `incomplete` and are not used for QoI datasets.

## Parameter Labels

The `bounds` keys determine which GROMACS force-field parameters are sampled
and subsequently learned. BFF currently supports:

| Parameter | Label syntax | Example | GROMACS quantity |
| --- | --- | --- | --- |
| Partial charge | `charge <name-or-type> [<name-or-type> ...]` | `charge O1 O2` | Atomic charge |
| Lennard-Jones sigma | `sigma <atom-type> [<atom-type> ...]` | `sigma OW` | LJ sigma |
| Lennard-Jones epsilon | `epsilon <atom-type> [<atom-type> ...]` | `epsilon OW` | LJ epsilon |
| Function-9 dihedral force constant | `dihedraltype9_<multiplicity>_<phase>` | `dihedraltype9_3_180` | Periodic-dihedral force constant |

Values use the native units of the GROMACS topology: elementary charge for
partial charges, nm for sigma, kJ mol^-1 for epsilon, degrees for the
dihedral phase, and kJ mol^-1 for the dihedral force constant.

### Charges

Charge labels resolve each token by atom name first and fall back to atom type
when no atom has that name. Multiple tokens in one label tie all matching atoms
to one sampled value:

```yaml
bounds:
  charge O1 O2: [-0.8, -0.3]
  charge HW: [0.1, 0.6]
```

Here, `O1` and `O2` share one charge parameter. If there is no atom named
`HW`, all atoms of type `HW` share the second parameter. Charge labels must not
overlap: one topology atom cannot be controlled by two entries in `bounds`.

Charge parameters may be fixed by the
[charge constraints](#charge-constraints) described above. Parameters of the other supported families are sampled
directly.

### Lennard-Jones Parameters

Sigma and epsilon labels address GROMACS atom types. Multiple atom types in one
label tie those types to one sampled value:

```yaml
bounds:
  sigma OW: [0.25, 0.38]
  epsilon OW: [0.58, 0.72]
  sigma NA CL: [0.20, 0.45]
```

### Function-9 Dihedrals

To sample a GROMACS function-9 dihedral force constant, use
`dihedraltype9_<multiplicity>_<phase>`:

```yaml
bounds:
  dihedraltype9_3_180: [0.0, 10.0]
```

This updates the force constant of every matching function-9 dihedral term in
the topology while preserving its multiplicity and phase. If the topology has
several function-9 terms with the same multiplicity and phase, the label ties
all of them to the same sampled value.

## Outputs

`bff sample-parameters` writes:

- `campaign_dir/specs.yaml`: the parameter specification
- `campaign_dir/samples.yaml`: parameters, status, and outputs per sample
- `campaign_dir/campaign.yaml`: job settings shared by all samples
- `campaign_dir/systems/<system_id>/`: staged inputs shared by all samples
- `campaign_dir/samples/<sample_id>/`: the job output `run.out` and the GROMACS
  log `gmx.log`
- `campaign_dir/samples/<sample_id>/<system_id>/`: the sample's topology and
  all MD outputs of that system
- `campaign_dir/run.sh`, `tasks.txt`, and `slurm/` for Slurm campaigns

`samples.yaml` looks like this (paths are relative to the campaign):

```yaml
parameter_names: [charge C1, charge O1 O2]   # order of every params list
provenance: {source: latin_hypercube, n_samples: 20, seed: 3602319095}
systems:
  acetate: {n_steps: 100000}
samples:
  "00":
    params: [-0.37, -0.76]
    status: completed
    outputs:
      acetate:
        topology: samples/00/acetate/topology.top
        trajectory: samples/00/acetate/production.xtc
        pmf: samples/00/acetate/production.pmf
```

The sample's `topology` is always kept; `trajectory` is recorded when `xtc` is
stored, and every other stored suffix under its own name, so `store: [xtc, pmf]`
adds `pmf`. With `cleanup: true`, system directories keep only the stored
suffixes and the topology; the files directly in `samples/<sample_id>/` are
always kept.

`samples.yaml` lists every sample from the start with its status (`staged`,
`completed`, `incomplete`, or `failed`) and records the parameter draw under
`provenance` (`source: latin_hypercube`, `n_samples`, `seed`).

A sample whose MD fails is recorded as `failed` in `samples.yaml`; the rest of
the campaign continues, locally and on Slurm. Rerun such samples with
`resume: true`.
