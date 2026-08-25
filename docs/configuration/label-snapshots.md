# Label Snapshots Configuration

`bff label-snapshots` extracts evenly spaced frames from user-provided
trajectories, runs the supplied CP2K MD and single-point inputs, and writes
MLIP-ready labeled datasets.

```yaml
output_dir: ./
job_scheduler: local
cp2k_cmd: cp2k.psmp
single_atoms: true
train_fraction: 0.8
seed: 2026

systems:
  - system_id: acetate
    topology: ../01-build/systems/acetate/production.gro
    trajectory: ../01-build/systems/acetate/production.xtc
    md_input: ../inputs/md.inp
    sp_input: ../inputs/sp.inp
    single_atom_inputs:
      H: ../inputs/atoms/h.inp
      C: ../inputs/atoms/c.inp
      O: ../inputs/atoms/o.inp
    n_snapshots: 100
    atom_selection: all
```

Each system requires a stable `system_id`, an MDAnalysis-compatible topology
and trajectory, CP2K inputs, and a positive snapshot count. Requests larger
than the number of trajectory frames are rejected. Optional `atom_selection`
is an MDAnalysis selection used to exclude virtual sites. CP2K inputs are copied
unchanged and must use the staged `pos.xyz` and `md-pos-1.xyz` coordinate
names expected by the two-step job.

When `single_atoms: true`, every system must provide `single_atom_inputs` with
exactly one CP2K input for each element selected from its topology. Element
keys are canonicalized (`ca` becomes `Ca`), then checked against the detected
set; missing and unused entries are rejected. BFF generates only the
single-atom `pos.xyz`, copies each supplied `input.inp` unchanged, and runs it.
The user is responsible for the functional, dispersion correction, basis,
potential, charge, multiplicity, and other CP2K settings. Set `single_atoms:
false` to disable these calculations and omit the mapping.

`train_fraction` must lie strictly between zero and one. Successfully labeled
frames are deterministically shuffled with `seed` and written to
`systems/<system_id>/train.extxyz` and `test.extxyz`.

Local and Slurm scheduling use `job_scheduler`, `cp2k_cmd`, `slurm`,
`collection_wait_seconds`, and `cleanup_snapshots`. The output root also
contains `label-results.yaml` with source hashes, selected trajectory indices,
split counts, artifact paths, and isolated-atom energies or failures, plus
`label-snapshots.log`.

The acetate example contains per-system xTB short-MD inputs, revPBE-D3 MD,
single-point, and isolated-atom inputs, and revPBE0-D3 single-point and
isolated-atom inputs. Its labeling configs select xTB for short MD and
revPBE0-D3 for single points and isolated atoms, so the atomic reference
energies match the final labels. The revPBE-D3 MD files are user-selectable
alternatives.
