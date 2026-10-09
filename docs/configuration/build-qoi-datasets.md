# Build QoI Datasets Configuration

`bff build-qoi-datasets` joins sampled and reference systems by stable `system_id`, never
by list position. Both sections must contain the same unique ID set.

```yaml
training_samples:
  manifest: ../03-sample-parameters/samples.yaml
  systems:
    - system_id: acetate
    - system_id: calcium-acetate
  frames: {start: 1, stop: null, step: 1}
  workers: -1

reference:
  systems:
    - system_id: acetate
      inputs:
        topology: ../01-build/systems/acetate/reference/topology.top
        coordinates: ../01-build/systems/acetate/reference/coordinates.gro
        trajectory: ../02-reference-md/acetate.xtc
    - system_id: calcium-acetate
      inputs:
        pmf: ../02-reference-md/calcium-acetate.pmf
  frames: {start: 1, stop: null, step: 1}

routines:
  - name: rdf
    type: rdf
    systems: [acetate]
    selections:
      group_a: "resname ACE and name O1 O2 H1 H2 H3"
      group_b: "resname SOL and name O*"
    options: {range: [1.0, 7.0], bins: 200}
  - name: pmf
    callable: ../inputs/pmf.py:read_pmf
    systems: [calcium-acetate]
    inputs: [pmf]
    options: {range: [0.27, 0.65], points: 39}

run:
  in_memory: true
output:
  directory: ./qoi
```

## Options

General rules for all options are on the
[conventions page](index.md).

### `training_samples`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `training_samples.manifest` | path | *required* | `samples.yaml` of a `bff sample-parameters` campaign. |
| `training_samples.systems` | list | *required* | Entries `{system_id: <ID>}`; the same ID set as `reference.systems`. |
| `training_samples.frames` | mapping | see [frames](#frames) | Trajectory frames analyzed for every sample. |
| `training_samples.workers` | integer >= 1 or -1 | `-1` | Samples analyzed in parallel; `-1` uses every available CPU. |

### `reference`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `reference.systems` | list | *required* | One entry per system ID. |
| `reference.systems[].system_id` | ID | *required* | System ID, paired with the training samples by ID. |
| `reference.systems[].inputs` | mapping | *required* | Role to path (or list of paths). Trajectory routines need `topology`, `coordinates`, and `trajectory`; file routines need the roles in their `inputs`. Roles no routine of this system uses are rejected. |
| `reference.frames` | mapping | see [frames](#frames) | Reference trajectory frames analyzed. |

### `frames`

Both `training_samples.frames` and `reference.frames` take:

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `start` | integer >= 0 | `1` | First frame (0-based); the default skips the starting structure. |
| `stop` | integer > `start` | none (last frame) | Frame after the last analyzed one. |
| `step` | integer >= 1 | `1` | Stride between analyzed frames. |

### `routines[]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | ID | *required* | QoI name and output file `qoi/<name>.pt`; unique. |
| `systems` | list of IDs | *required* | Systems this routine analyzes. |
| `type` | `rdf` or `hydrogen_bonds` | none | Built-in routine; exactly one of `type` and `callable`. |
| `callable` | string | none | Custom routine as `module:function` or `path/to/file.py:function` (relative to this file). |
| `selections` | mapping | `{}` | Built-ins only: MDAnalysis selections merged into `options`. |
| `inputs` | list of strings | `[]` | Custom routines only: file roles passed to the routine; without them it analyzes a trajectory. |
| `options` | mapping | `{}` | Passed to the routine, which checks them itself; see below for the built-ins. |

### `run` and `output`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `run.in_memory` | boolean | `true` | Copy each analyzed trajectory slice into memory once, so every routine reads it without decompressing again. A slice larger than its share of the available memory is read from disk instead. |
| `output.directory` | path | `./qoi` | Directory for `<name>.pt` datasets. |
| `output.log` | path | `<output.directory>/../build-qoi-datasets.log` | Workflow log file. |

## Built-in Routines

Built-ins are `rdf` and `hydrogen_bonds`. RDF requires `group_a` and
`group_b`. It emits one curve for every atom type represented in `group_a`,
with labels sorted by atom type; `group_b` is the neighbor selection used for
every curve.

Hydrogen bonds require `selection`, the solute heavy-atom sites of interest;
`water_selection` defines the whole solvent group. BFF finds O/N/S sites, discovers donor hydrogens from topology
bonds, and evaluates both solute-to-water and water-to-solute combinations.
Override the candidate elements with `options.elements` when needed. An NH2
nitrogen can therefore contribute as both a donor and an acceptor.

| `rdf` option | Type | Default | Description |
| --- | --- | --- | --- |
| `group_a` | selection | *required* | Atoms whose RDFs are computed, one curve per atom type. |
| `group_b` | selection | *required* | Neighbor atoms. |
| `range` | `[min, max]` | `[0, 10]` | Distance range in angstrom. |
| `bins` | integer | `200` | Histogram bins per curve. |
| `pbc` | boolean | `true` | Use periodic boundary conditions; without them the curves are pair counts per shell, not normalized by the box volume. |
| `update_selections` | boolean | `false` | Reevaluate selections every frame. |
| `smooth` | boolean | `false` | Smooth each curve. |

| `hydrogen_bonds` option | Type | Default | Description |
| --- | --- | --- | --- |
| `selection` | selection | *required* | Solute heavy-atom sites of interest. |
| `water_selection` | selection | `resname SOL HOH WAT` | The whole solvent group. |
| `elements` | list of strings | `[O, N, S]` | Candidate donor and acceptor elements. |
| `donor_acceptor_cutoff` | number | `3.5` | Donor-acceptor distance cutoff in angstrom. |
| `angle_cutoff` | number | `150` | Minimum donor-hydrogen-acceptor angle in degrees. |
| `pbc` | boolean | `true` | Use periodic boundary conditions. |
| `update_selections` | boolean | `false` | Reevaluate selections every frame. |

Selections are full MDAnalysis expressions. Static selections are the default;
`update_selections: true` reevaluates them each frame. Empty selections,
missing bonds, and invalid PBC boxes are errors.

For reference MDAnalysis routines, use the virtual-site-free topology and
coordinates created by `bff build`. The external reference trajectory must
contain the same atoms in the same order. BFF keeps the trajectory explicit
because the reference simulation is outside this workflow; see
[Reference trajectories](../reference-trajectories.md).

## Routine Interface

Built-in and custom routines share one interface. A routine receives one
system of one sample (or the reference) and returns exactly one `QoI`:

```python
routine(universe, *, frames, options) -> QoI  # trajectory
routine(*, inputs, options) -> QoI            # files
```

Built-ins receive their `selections` merged into `options`. Every routine
validates its own options when it runs; the reference is analyzed before the
training samples, so configuration errors surface within seconds. The
built-ins in `bff/qoi/rdf.py` and `bff/qoi/hbonds.py` are complete examples of
trajectory routines.

Custom routines import their helpers from `bff.qoi`:

```python
from bff.qoi import QoI, get_unitcell, select_atoms
```

`get_unitcell(universe, ts)` returns the validated box of a frame; frames that
store no box fall back to the box of the configured coordinate file.
`select_atoms(universe, selection, field=...)` rejects empty or invalid
selections with an error naming the option.

Every routine returns exactly one `QoI`; the name configured under
`routines[].name` replaces the name returned by the callable. A sample whose
labels, `values_per_label`, number of values, or settings differ from the
reference's is skipped, so put everything that defines the QoI, such as a
grid, into `settings`. A routine applied to several systems must return the
same labels and length for each.

Declare `inputs` when the quantity is already stored in files such as a PMF:

```python
import numpy as np
from bff.qoi import QoI


def read_pmf(*, inputs, options) -> QoI:
    lower, upper = options.get("range", (0.27, 0.65))
    points = int(options.get("points", 39))
    distance, free_energy = np.loadtxt(
        inputs["pmf"], comments="#", usecols=(0, 1), unpack=True
    )
    grid = np.linspace(lower, upper, points)
    values = np.interp(grid, distance, free_energy)
    return QoI(
        name="pmf",
        values=values - values.mean(),  # a PMF is defined up to a constant
        labels=("Ca-C2",),
        values_per_label=points,
        settings={"distance_nm": grid.round(6).tolist()},
    )
```

This is `inputs/pmf.py` of the [acetate example](../examples/acetate.md).
Declare the roles a routine reads with `inputs`:

```yaml
- name: pmf
  callable: ../inputs/pmf.py:read_pmf
  systems: [calcium-acetate]
  inputs: [pmf]
  options: {range: [0.27, 0.65], points: 39}
```

For training samples, roles such as `pmf` come from `samples.yaml`, under
`samples.<sample_id>.outputs.<system_id>`: every suffix in the campaign's
`store` (for example `store: [xtc, pmf]`) is recorded there under its own
name. For reference systems, add the same
role under `reference.systems[].inputs`. Each declared role is passed as a
resolved `Path`; a role backed by multiple paths is passed as a tuple of paths.

A custom callable without `inputs` is trajectory-based:

```python
def trajectory_qoi(universe, *, frames, options) -> QoI:
    values = calculate(universe, frames, options)
    return QoI(name="custom", values=values)
```

The supplied `universe` already contains the configured topology, coordinates,
and trajectory. Iterate over `universe.trajectory[frames]`; do not reopen the
trajectory. File-based routines do not receive a universe or frame slice, and
trajectory-based routines do not receive the `inputs` mapping.

## How Samples Are Analyzed

Samples are analyzed in parallel, one worker process per sample. Within a
sample, the systems are analyzed one after another; each system's trajectory
is opened once and all of that system's routines run on it. The reference is
analyzed first, the same way, so configuration errors appear within seconds.

Each training sample is analyzed with its own topology,
`samples/<sample_id>/<system_id>/topology.top`, which carries the sampled
parameters; routines that need charges, such as a dipole, therefore see the
sample's charges. Coordinates come from the staged `systems/<system_id>/`.

A sample is skipped, with a warning naming it and the reason, when its files
are missing, its trajectory or a routine fails, or its QoIs do not match the
reference in shape. The remaining samples are still analyzed, and every
dataset contains the same samples. A failing reference stops the stage.

## Outputs

One `qoi/<routine-name>.pt` per routine and `build-qoi-datasets.log`. The log
marks the start of the reference and of the sample analysis and, every 100
samples, the progress and the estimated remaining time. Each
dataset records the `sample_ids` of its rows, the `parameter_names` of its
input columns (from the campaign), and the `system_ids` it combines.
`fit-lgp` stores the parameter names in the model, and `learn` checks them
against its `specs.yaml`.
