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
- `training/window-XXX/` prepared assets for `bff simulate`
- staged CP2K reference inputs under `reference/window-XXX/`

Each reference window contains a direct-MD `md/` folder and a `snapshots/`
folder. The snapshot folder contains decorrelated XYZ snapshots, CP2K
`single-point.inp` and 100-step `md.inp` templates, and Slurm helpers. Run
`bff cp2k-collect` in that folder to assemble a deterministic shuffled
`train.extxyz` and `valid.extxyz` split.

The full stage is self-contained in the exported example tree and does not
depend on repository-level `data/` paths.

The rest of the acetate walkthrough uses the Colvars variant by default.
