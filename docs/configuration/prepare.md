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

Those directories are consumed directly by both `simulate` and `validate`.

The reference tree is system-centered:

- `PROJECT/reference/system-XXX/system.gro`
- `PROJECT/reference/system-XXX/system.top`
- `PROJECT/reference/system-XXX/system.xyz`
- `PROJECT/reference/system-XXX/md/`
  CP2K direct-MD inputs and `run.sh`.
- `PROJECT/reference/system-XXX/single-atoms/`
  One isolated-atom CP2K `input.inp` and centered `pos.xyz` per
  unique element in the prepared system plus shared top-level `run.sh` and
  `submit.sh` helpers. These jobs use a
  neutral atom in a nonperiodic `10 Å` cubic box with the same revPBE-based
  CP2K setup as the staged MD inputs.
- `PROJECT/reference/system-XXX/snapshots/`
  Decorrelated XYZ snapshots, `single-point.inp`, a 100-step `md.inp`,
  `run.sh`, and `submit.sh`. Use `bff cp2k-collect` from this directory to
  read the final short-MD positions, forces, and potential energies from
  `runs/` and write an 80/20 deterministic shuffled `train.extxyz` /
  `valid.extxyz` split in `eV` / `eV/Å`.
  CP2K XYZ outputs are read directly. If you switch the CP2K output format to
  DCD, run `bff cp2k-collect --topology pos.xyz` or point `--topology` at another
  per-run topology file.

Generated CP2K Slurm scripts are intentionally machine-agnostic. They use
`CP2K_CMD=cp2k.psmp` by default and source an optional local `setup-env.sh`
file if you need cluster-specific modules, Spack loads, or environment exports.

[prepare-configs]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/workflows/configs.py
[prepare-workflow]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/workflows/prepare.py
[prepare-topology]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/bff/topology.py
