# Build Configuration

Source code:

- `bff/workflows/build/config.py`
- `bff/workflows/build/main.py`
- `bff/topology.py`

## Purpose

`bff build` prepares equilibrated systems and runs one seeded production
trajectory for each system. `bff prepare-assets` packages that seed into FFMD
and CP2K reference assets.

- equilibrated GROMACS systems under `equilibration/`
- seeded production outputs under `equilibration/system-XXX-prod.*`
- a `build-manifest.yaml` handoff file consumed by asset-preparation workflows

## Minimal Example

```yaml
project:
  directory: ./
  log: ./out.log

gromacs:
  command: gmx

defaults:
  nsteps:
    npt: 0
    prod: 100000

systems:
  - topology: ../inputs/common/topol.top
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
  Output directory for `equilibration/` and `build-manifest.yaml`.
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
  Optional per-system seeded production run length. The seed trajectory is used
  later by `bff prepare-assets`.
- `mdp.em`
  Energy minimization MDP file.
- `mdp.npt`
  NpT equilibration MDP file.
- `mdp.prod`
  Production MDP file used for the seeded run and downstream FFMD assets.

## Outputs

The main downstream output is `PROJECT/build-manifest.yaml`. It records the
prepared topology, index, MDP files, copied bias input, seeded production
coordinate file, seeded production trajectory, CP2K charge/multiplicity, and
box for each system.

After a successful build, run `bff prepare-assets` to write the `ffmd/` and
`reference/` asset trees.
