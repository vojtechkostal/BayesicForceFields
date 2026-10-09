# Build Configuration

Source code:

- `bff/workflows/build/config.py`
- `bff/workflows/build/main.py`
- `bff/workflows/build/box.py`
- `bff/gromacs.py`

## Purpose

`bff build` prepares equilibrated systems and runs one seeded production
trajectory for each system. It also writes a reference-compatible topology and
coordinate pair with virtual sites removed.

- equilibrated GROMACS systems under `systems/<system_id>/`
- seeded production outputs under each stable system-ID directory
- `systems/<system_id>/reference/{topology.top,coordinates.gro}` for reference
  trajectories and QoI construction

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
    topology: ../inputs/acetate.top
    templates:
      ACE: ../inputs/ace.gro
    mdp:
      em: ../inputs/mdp/em.mdp
      npt: ../inputs/mdp/npt.mdp
      prod: ../inputs/mdp/nvt.mdp
    nsteps:
      npt: 0
      prod: 100000
    box: [15.7107, 15.7107, 15.7107, 90, 90, 90]
```

## Options

General rules for all options are on the
[conventions page](index.md).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `project.directory` | path | *required* | Output directory for `equilibration/` and `systems/`; created if missing. |
| `project.log` | path | `<project.directory>/build.log` | Workflow log file. |
| `gromacs.command` | string | *required* | GROMACS executable, for example `gmx`. |
| `systems` | list | *required* | Systems to build; see below. |

### `systems[]`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `system_id` | ID | *required* | Stable system ID used by every later stage. |
| `system_name` | string | none | Display name shown in the log; never used for matching or paths. |
| `topology` | path | *required* | GROMACS topology; its molecule counts define the box contents. |
| `templates` | mapping | `{}` | Residue name to coordinate template file, for residues other than built-in water and monoatomic ions. |
| `box` | 3 positive numbers | guessed from the heavy-atom count | Box lengths in angstrom. Only rectangular boxes are supported; three angles of 90 may follow. |
| `bias.colvars_file` | path | none | Colvars input for the production run and for campaigns built on this system; equilibration is unbiased. |
| `bias.plumed_file` | path | none | PLUMED input; at most one of `colvars_file` and `plumed_file`. |
| `nsteps.npt` | integer >= 0 | *required* | NpT equilibration steps; `0` skips NpT. |
| `nsteps.prod` | integer >= 1 | *required* | Seeded production steps; the final frame becomes the `reference/` coordinates. |
| `mdp.em` | path | *required* | Energy-minimization MDP file. |
| `mdp.npt` | path | *required* | NpT equilibration MDP file. |
| `mdp.prod` | path | *required* | Production MDP file, also used by later campaigns. |

## Outputs

The stage writes `build.log`, `gromacs.log`, and `systems/<system_id>/`.
Each directory uses fixed filenames for the topology, index, MDPs, optional
bias, and seeded production outputs; the files themselves are the record of
the build.

The `reference/` pair is always generated from the final `production.gro`.
Atoms declared by `[ virtual_sites* ]` sections are removed exactly from both
files; systems without virtual sites still receive the same stable paths. Use
this topology and coordinate pair to start the external reference MD, so its
atom order matches the inputs later supplied to `build-qoi-datasets`; see
[Reference trajectories](../reference-trajectories.md).

`bff sample-parameters` and `bff validate` consume this directory directly.
`bff build-qoi-datasets` accepts the files under `reference/`
and the externally generated reference trajectory as separate explicit inputs.
