# Build QoI Datasets Configuration

`bff build-qoi-datasets` joins sampled and reference systems by stable `system_id`, never
by list position. Both sections must contain the same unique ID set.

```yaml
training_samples:
  manifest: ../03-sample/samples.yaml
  systems:
    - system_id: acetate
    - system_id: acetate-contact
  frames: {start: 1, stop: null, step: 1}
  workers: -1
  progress_stride: 10

reference:
  systems:
    - system_id: acetate
      inputs:
        topology: ../01-build/systems/acetate/reference/topology.top
        coordinates: ../01-build/systems/acetate/reference/coordinates.gro
        trajectory: ../02-reference-md/trajectories/acetate/trajectory.xtc
    - system_id: acetate-contact
      inputs:
        topology: ../01-build/systems/acetate-contact/reference/topology.top
        coordinates: ../01-build/systems/acetate-contact/reference/coordinates.gro
        trajectory: ../02-reference-md/trajectories/acetate-contact/trajectory.xtc
        pmf: ../02-reference-md/trajectories/acetate-contact/profile.pmf
  frames: {start: 1, stop: null, step: 1}

routines:
  - name: acetate-water-rdf
    type: rdf
    systems: [acetate, acetate-contact]
    selections:
      group_a: "resname ACE and name O1 O2 H1 H2 H3"
      group_b: "resname SOL and name O*"
    options:
      range: [1.0, 7.0]
      bins: 200
      pbc: true
      update_selections: false
      smooth: false
  - name: contact-pmf
    callable: ../inputs/pmf.py:load_profile
    systems: [acetate-contact]
    inputs: [pmf]

run:
  in_memory: true
output:
  directory: ./qoi
  write_raw: false
```

Built-ins are `rdf` and `hydrogen_bonds`. RDF requires `group_a` and
`group_b`. It emits one curve for every atom type represented in `group_a`,
with labels sorted by atom type; `group_b` is the neighbor selection used for
every curve.

Hydrogen bonds require `selection` and `water_selection`. `selection` defines
the solute heavy-atom sites of interest and `water_selection` defines the whole
solvent group. BFF finds O/N/S sites, discovers donor hydrogens from topology
bonds, and evaluates both solute-to-water and water-to-solute combinations.
Override the candidate elements with `options.elements` when needed. An NH2
nitrogen can therefore contribute as both a donor and an acceptor.

RDF options are `range` (default `[0, 10]` Angstrom), `bins` (200), `pbc`
(true), `update_selections` (false), and `smooth` (false). Hydrogen-bond
options are `elements` (`[O, N, S]`), `donor_acceptor_cutoff` (3.5 Angstrom),
`angle_cutoff` (150 degrees), `pbc`, and `update_selections`.

Selections are full MDAnalysis expressions. Static selections are the default;
`update_selections: true` reevaluates them each frame. Empty selections,
missing bonds, and invalid PBC boxes are errors.

For reference MDAnalysis routines, use the virtual-site-free topology and
coordinates created by `bff build`. The external MLIP trajectory must contain
the same atoms in the same order. BFF keeps the trajectory explicit because
training and running the MLIP are outside this workflow.

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
`routines[].name` replaces the name returned by the callable. Labels,
`values_per_label`, and settings must be identical for the reference and every
training sample.

Declare `inputs` when the quantity is already stored in files such as a PMF:

```python
def load_profile(*, inputs, options) -> QoI:
    pmf_path = inputs["pmf"]
    data = np.loadtxt(pmf_path, comments="#")
    coordinate = data[:, 0]
    values = data[:, 1] - data[:, 1].min()
    return QoI(
        name="pmf",
        values=values,
        labels=("PMF",),
        values_per_label=len(values),
        settings={"coordinate": tuple(float(value) for value in coordinate)},
    )
```

Declare every required role explicitly:

```yaml
- name: pmf
  callable: ./pmf.py:load_profile
  systems: [acetate-calcium]
  inputs: [pmf]
  options: {}
```

For training samples, roles such as `pmf` come from
`samples.yaml` under `outputs[].inputs`. For reference systems, add the same
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

Each worker analyzes one complete training sample and processes that sample's
systems sequentially. All trajectory routines for one system share the same
Universe. The reference follows the same path as one sample, while only the
training samples are processed in parallel.

Outputs are `qoi/<routine-name>.pt`, `build-qoi-datasets.log`, and optional
`qoi/raw.json`. Dataset metadata records `system_ids`.
