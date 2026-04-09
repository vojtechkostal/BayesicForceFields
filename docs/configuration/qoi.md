# QoI Configuration

Source code:

- `bff/workflows/configs.py`
- `bff/workflows/qoi.py`
- `bff/qoi/routines.py`

## Purpose

`bff qoi` computes quantities of interest for:

- the sampled trainset produced by `bff simulate`
- the reference trajectories provided by the user

## Minimal Example

```yaml
trainset:
  dir: ../03-training-trjs/trainset

refset:
  systems:
    - coordinates: ../01-prepare/colvars/ace-colvars/reference/system-000/system.gro
      topology: ../01-prepare/colvars/ace-colvars/reference/system-000/system.top
      trajectory: ../02-reference-trjs/pos-000.xtc
      routines:
        - routine: rdf
          mol_resname: ACE
        - routine: ./restraint.py:distance_distribution
          atom_pair: [C2, CAL]

run:
  in_memory: true
  gc_collect: false
  maxtasksperchild: 100

output:
  path: ./qoi
  log: ./out.log
  write_raw: true
```

## Top-Level Keys

- `trainset`
  Trainset slicing and worker settings.
- `refset`
  Reference systems, slicing, and routines.
- `run`
  Runtime settings for multiprocessing and memory handling.
- `output`
  Output prefix and logging settings.

## `trainset` Keys

- `dir`
  Trainset directory produced by `bff simulate`.
- `start`
  First frame index to analyze.
- `stop`
  Final frame index to analyze.
- `step`
  Frame stride.
- `workers`
  Worker count for trainset analysis. `-1` means the implementation default.
- `progress_stride`
  Progress update interval for trainset processing.

## `refset` Keys

- `start`
  First frame index for reference trajectories.
- `stop`
  Final frame index for reference trajectories.
- `step`
  Frame stride for reference trajectories.
- `systems`
  Non-empty list of reference systems.

## `refset.systems[]` Keys

- `coordinates`
  Coordinate file for the reference system.
- `topology`
  Topology file for the reference system.
- `trajectory`
  Reference trajectory file.
- `routines`
  Non-empty list of QoI routines to evaluate for that system.

## Routine Keys

- `routine`
  Either a builtin routine name such as `rdf` or `hb`, or a custom routine
  string in the form `path.py:function`.
- any additional keys
  Passed directly to the routine as keyword arguments.

Builtin routines currently live in:

- `bff/qoi/rdf.py`
- `bff/qoi/hbonds.py`

## `run` Keys

- `in_memory`
  Load trainset QoI results into memory before reduction.
- `gc_collect`
  Run explicit garbage collection between tasks.
- `maxtasksperchild`
  Worker recycling interval for multiprocessing.

## `output` Keys

- `path`
  Output prefix for the written QoI dataset files.
- `log`
  Log file path.
- `write_raw`
  If `true`, also write the raw per-block QoI data as JSON.
