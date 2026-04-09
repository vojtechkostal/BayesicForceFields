# Prepare Stage

Stage `01-prepare` contains two equivalent setup variants for the acetate
example:

- [common](common/)
  contains the shared force-field files, topologies, template coordinates, and
  MDP files used by both bias variants.
- [colvars](colvars/)
  stages the example with Colvars bias files only.
- [plumed](plumed/)
  stages the same systems with PLUMED bias files only.

Both variants produce the same downstream kind of assets:

- equilibrated GROMACS systems
- `training/system-XXX/` prepared assets for `bff simulate`
- staged CP2K reference inputs under `reference/system-XXX/`

Each reference system contains a direct-MD `md/` folder and a `snapshots/`
folder plus a `single-atoms/` folder for isolated elemental reference
energies with shared `run.sh` and `submit.sh` helpers. The snapshot folder contains decorrelated XYZ snapshots, CP2K
`single-point.inp` and 100-step `md.inp` templates, and Slurm helpers. Run
`bff cp2k-collect` in that folder to assemble a deterministic shuffled
`train.extxyz` and `valid.extxyz` split in `eV` / `eV/Å`.

The full stage is self-contained in the exported example tree and does not
depend on repository-level `data/` paths.

The rest of the acetate walkthrough uses the Colvars variant by default.
