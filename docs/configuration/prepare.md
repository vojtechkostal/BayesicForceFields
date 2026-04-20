# Prepare Configuration

Source code:

- [configs.py][prepare-configs]
- [prepare.py][prepare-workflow]
- [topology.py][prepare-topology]

## Purpose

`bff prepare` stages everything needed for downstream workflows:

- equilibrated GROMACS systems
- reusable training assets under `training/system-XXX/`
- CP2K reference assets under `reference/system-XXX/`

## Minimal Example

```yaml
project:
  directory: ./ace-colvars
  log: ./out.log

gromacs:
  command: gmx

defaults:
  nsteps:
    npt: 0
    prod: 100000

reference:
  n_single_point_snapshots: 1000

systems:
  - topology: ./topol.top
    templates:
      ACE: ./ace.gro
    mdp:
      em: ../../../data/mdp/em.mdp
      npt: ../../../data/mdp/npt.mdp
      prod: ../../../data/mdp/nvt.mdp
    charge: -1
    multiplicity: 1
    box: [15.7107, 15.7107, 15.7107, 90, 90, 90]
```

## Top-Level Keys

- `project`
  Project directory output. A string is accepted as shorthand for
  `project.directory`.
- `project.directory`
  Output directory for `equilibration/`, `training/`, and `reference/`.
- `project.log`
  Optional workflow log file.
- `gromacs.command`
  GROMACS executable, usually `gmx`.
- `defaults.nsteps.npt`
  Default NpT equilibration length for systems that do not override it.
- `defaults.nsteps.prod`
  Default production-preparation run length for systems that do not override it.
- `reference.n_single_point_snapshots`
  Number of evenly spaced snapshots staged for CP2K single-point and short-MD
  reference jobs.
- `systems`
  Non-empty list of prepared systems.

## `systems[]` Keys

- `topology`
  GROMACS topology describing residue counts.
- `templates`
  Mapping from residue name to coordinate template file for non-standard
  residues.
- `charge`
  Total system charge for staged CP2K reference inputs.
- `multiplicity`
  Spin multiplicity for staged CP2K reference inputs.
- `box`
  Optional box dimensions. Accepts 3 values or full 6-value triclinic format.
- `bias`
  Optional bias specification. Use either `plumed_file` or `colvars_file`.
- `nsteps.npt`
  Optional per-system NpT override.
- `nsteps.prod`
  Optional per-system production-preparation override.
- `mdp.em`
  Energy minimization MDP file.
- `mdp.npt`
  NpT equilibration MDP file.
- `mdp.prod`
  Production MDP file staged into the downstream training assets.

## Outputs

The most important output for downstream BFF workflows is:

- `PROJECT/training/system-XXX/`

Each such directory contains one prepared system with:

- `system-XXX.top`
- `system-XXX.gro`
- `system-XXX.ndx`
- `system-XXX.em.mdp`
- `system-XXX.npt.mdp`
- `system-XXX.mdp`
- optional bias file

Those directories are consumed directly by both `trainset` and `validate`.

The reference tree is system-centered:

- `PROJECT/reference/system-XXX/system.gro`
- `PROJECT/reference/system-XXX/system.top`
- `PROJECT/reference/system-XXX/system.xyz`
- `PROJECT/reference/system-XXX/md/`
  CP2K direct-MD input files.
- `PROJECT/reference/system-XXX/single-atoms/`
  One isolated-atom CP2K `input.inp` and centered `pos.xyz` per unique element
  in the prepared system. These jobs use a neutral atom in a nonperiodic
  `10 Å` cubic box with the same revPBE-based CP2K setup as the staged MD
  inputs.
- `PROJECT/reference/system-XXX/snapshots/`
  Decorrelated XYZ snapshots, `sp.inp`, and a 100-step `md.inp`.

`bff prepare` only stages these reusable reference inputs. The runnable output
tree, per-snapshot `runs/` directories, optional Slurm staging, and final
`train.extxyz` / `valid.extxyz` split are created later by `bff reference`,
typically from a separate `02-reference-data/` example stage.

[prepare-configs]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/workflows/configs.py
[prepare-workflow]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/workflows/prepare.py
[prepare-topology]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/topology.py
