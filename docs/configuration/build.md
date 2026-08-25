# Build Configuration

Source code:

- `bff/workflows/build/config.py`
- `bff/workflows/build/main.py`
- `bff/topology.py`

## Purpose

`bff build` prepares equilibrated systems and runs one seeded production
trajectory for each system. It also writes a reference-compatible topology and
coordinate pair with virtual sites removed.

- equilibrated GROMACS systems under `systems/<system_id>/`
- seeded production outputs under each stable system-ID directory
- `systems/<system_id>/reference/{topology.top,coordinates.gro}` for reference
  trajectories and QoI construction
- metadata-only `system.yaml` files colocated with each system

## Minimal Example

```yaml
project:
  directory: ./
  log: ./build.log

gromacs:
  command: gmx

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
    nsteps:
      npt: 0
      prod: 100000
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
  Required per-system NpT equilibration length. Use `0` to skip NpT.
- `nsteps.prod`
  Required per-system seeded production run length. The seed trajectory is
  used later by `bff label-snapshots`.
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

The `reference/` pair is always generated from the final `production.gro`.
Atoms declared by `[ virtual_sites* ]` sections are removed exactly from both
files; systems without virtual sites still receive the same stable paths. Use
this topology and coordinate pair when producing an external MLIP trajectory,
so its atom order matches the inputs later supplied to `build-qoi-datasets`.

`bff sample-parameters` and `bff validate` consume this directory directly.
`bff label-snapshots` accepts its production GRO and trajectory files as
explicit inputs. `bff build-qoi-datasets` accepts the files under `reference/`
and the externally generated reference trajectory as separate explicit inputs.
