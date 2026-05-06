# Analyze Configuration

Source code:

- `bff/workflows/analyze/config.py`
- `bff/workflows/analyze/main.py`
- `bff/qoi/routines.py`

## Purpose

`bff analyze` computes quantities of interest for:

- the sampled campaigns produced by `bff sample`
- the reference trajectories provided by the user

## Minimal Example

```yaml
sample:
  dir: ../03-sample

reference:
  systems:
    - coordinates: ../03-reference/trajectories/system-000/system.gro
      topology: ../03-reference/trajectories/system-000/system.top
      trajectory: ../03-reference/trajectories/system-000/trajectory.xtc
      routines:
        - routine: rdf
          mol_resname: ACE
        - routine: ../inputs/restraint.py:distance_distribution
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

- `sample`
  Sample slicing and worker settings.
- `reference`
  Reference systems, slicing, and routines.
- `run`
  Runtime settings for multiprocessing and memory handling.
- `output`
  Output prefix and logging settings.

## `sample` Keys

- `dir`
  Sample campaign directory produced by `bff sample`.
- `start`
  First frame index to analyze.
- `stop`
  Final frame index to analyze.
- `step`
  Frame stride.
- `workers`
  Worker count for sample analysis. `-1` means the implementation default.
- `progress_stride`
  Progress update interval for sample processing.

## `reference` Keys

- `start`
  First frame index for reference trajectories.
- `stop`
  Final frame index for reference trajectories.
- `step`
  Frame stride for reference trajectories.
- `systems`
  Non-empty list of reference systems.

## `reference.systems[]` Keys

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
  Load sample QoI results into memory before reduction.
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
