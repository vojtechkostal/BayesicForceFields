# Build Configuration

Source code:

- `bff/workflows/build/config.py`
- `bff/workflows/build/main.py`
- `bff/topology.py`

## Purpose

`bff build` stages everything needed for downstream workflows:

- equilibrated GROMACS systems under `equilibration/`
- reusable FFMD assets under `ffmd/system-XXX/`
- staged CP2K reference assets under `reference/system-XXX/`

## Minimal Example

```yaml
project:
  directory: ./01-build-colvars
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
  Project output settings. A string is accepted as shorthand for `project.directory`.
- `project.directory`
  Output directory for `equilibration/`, `ffmd/`, and `reference/`.
- `project.log`
  Optional workflow log file.
- `gromacs.command`
  GROMACS executable, usually `gmx`.
- `defaults.nsteps.npt`
  Default NpT equilibration length for systems that do not override it.
- `defaults.nsteps.prod`
  Default production-preparation run length for systems that do not override it.
- `reference.n_single_point_snapshots`
  Number of evenly spaced snapshots staged for CP2K single-point and short-MD jobs.
- `systems`
  Non-empty list of systems to build.

## `systems[]` Keys

- `topology`
  GROMACS topology describing residue counts.
- `templates`
  Mapping from residue name to coordinate template file for non-standard residues.
- `charge`
  Total system charge for staged CP2K reference inputs.
- `multiplicity`
  Spin multiplicity for staged CP2K reference inputs.
- `box`
  Optional box dimensions. Accepts 3 values or full 6-value triclinic format.
- `bias`
  Optional opaque bias specification. Use either `plumed_file` or `colvars_file`.
- `nsteps.npt`
  Optional per-system NpT override.
- `nsteps.prod`
  Optional per-system production-preparation override.
- `mdp.em`
  Energy minimization MDP file.
- `mdp.npt`
  NpT equilibration MDP file.
- `mdp.prod`
  Production MDP file staged into the downstream FFMD assets.

## Outputs

The most important downstream output is `PROJECT/ffmd/system-XXX/`.
Each such directory contains one prepared system with topology, coordinates,
index file, staged MDP files, and an optional copied bias file.

The reference tree is system-centered:

- `PROJECT/reference/system-XXX/system.gro`
- `PROJECT/reference/system-XXX/system.top`
- `PROJECT/reference/system-XXX/system.xyz`
- `PROJECT/reference/system-XXX/md/`
- `PROJECT/reference/system-XXX/single-atoms/`
- `PROJECT/reference/system-XXX/snapshots/`

`bff build` only stages these reusable assets. Canonical `train.extxyz` and
`valid.extxyz` files are created later by `bff reference`.
