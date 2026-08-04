# Build Configuration

Source code:

- `bff/workflows/build/config.py`
- `bff/workflows/build/main.py`
- `bff/topology.py`

## Purpose

`bff build` prepares equilibrated systems and runs one seeded production
trajectory for each system. Its fixed system-directory layout is consumed
directly by sampling, validation, and reference preparation.

- equilibrated GROMACS systems under `systems/<system_id>/`
- seeded production outputs under each stable system-ID directory
- metadata-only `system.yaml` files colocated with each system

## Minimal Example

```yaml
project:
  directory: ./
  log: ./build.log

gromacs:
  command: gmx

defaults:
  nsteps:
    npt: 0
    prod: 100000

systems:
  - system_id: acetate
    system_name: Aqueous acetate
    topology: ../inputs/common/topol.top
    templates:
      ACE: ../inputs/common/ace.gro
    mdp:
      em: ../inputs/common/mdp/em.mdp
      npt: ../inputs/common/mdp/npt.mdp
      prod: ../inputs/common/mdp/nvt.mdp
    charge: -1
    multiplicity: 1
    box: [15.7107, 15.7107, 15.7107, 90, 90, 90]
```

## Top-Level Keys

- `project`
  Project output settings. A string is accepted as shorthand for `project.directory`.
- `project.directory`
  Output directory for `equilibration/` and `systems/`.
- `project.log`
  Optional workflow log file.
- `gromacs.command`
  GROMACS executable, usually `gmx`.
- `defaults.nsteps.npt`
  Default NpT equilibration length for systems that do not override it.
- `defaults.nsteps.prod`
  Default seeded production run length for systems that do not override it.
- `systems`
  Non-empty list of systems to build.

## `systems[]` Keys

- `system_id`
  Required lowercase file-safe ID matching `[a-z0-9][a-z0-9._-]*`.
- `system_name`
  Optional display-only name; never used for matching or paths.
- `topology`
  GROMACS topology describing residue counts.
- `templates`
  Optional mapping from residue name to coordinate template file for
  non-standard residues. Omit it when the system only contains built-in water
  or monoatomic-ion residues.
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
  Optional per-system seeded production run length. The seed trajectory is used
  later by `bff prepare-reference`.
- `mdp.em`
  Energy minimization MDP file.
- `mdp.npt`
  NpT equilibration MDP file.
- `mdp.prod`
  Production MDP file used for the seeded run and downstream FFMD assets.

## Outputs

The stage writes `build.log`, `gromacs.log`, and `systems/<system_id>/`.
Each directory uses fixed filenames for the topology, index, MDPs, optional
bias, and seeded production outputs. Its `system.yaml` contains only display
and physical metadata such as charge, multiplicity, box, and production length;
it contains no file paths or version field.

`bff sample` and `bff validate` consume this directory directly. Run
`bff prepare-reference` to create CP2K reference inputs.
