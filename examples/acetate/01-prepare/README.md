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
- `training/system-XXX/` prepared assets for `bff trainset`
- staged CP2K reference inputs under `reference/system-XXX/`

Each staged reference system now contains only reusable inputs:

- `md/`
  CP2K direct-MD input files
- `snapshots/`
  decorrelated `snapshot-*.xyz` files plus `md.inp` and `sp.inp`
- `single-atoms/`
  one `input.inp` and `pos.xyz` per isolated element

No per-snapshot `run.sh`, `submit.sh`, or `runs/` folders are generated during
prepare anymore. Those belong to the execution stage.

The preferred launcher is:

```bash
cd colvars
bff prepare config.yaml
cd ../../02-reference-data
bff reference config-local.yaml
```

That runs the staged snapshot MD/single-point jobs from the inputs in
`01-prepare`, optionally evaluates the isolated single-atom energies, and
writes the deterministic shuffled `train.extxyz` and `valid.extxyz` split in
`eV` / `eV/Å` under `02-reference-data/reference-assets/`.

The full stage is self-contained in the exported example tree and does not
depend on repository-level `data/` paths.

The rest of the acetate walkthrough uses the Colvars variant by default.
