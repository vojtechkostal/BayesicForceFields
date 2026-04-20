# Reference Stage

Stage `02-reference` runs the staged CP2K reference calculations written by
`bff prepare`.

The prepared CP2K inputs still live under:

- `../01-prepare/colvars/reference/system-XXX/`

This stage keeps the runnable output separate under:

- `./reference/system-XXX/`

The configs in this directory point back to the staged inputs in
`../01-prepare/colvars/reference/system-XXX/`.

An optional revPBE0 single-point template is included at:

- `inputs/revpbe0-sp.inp`

Use it by uncommenting `systems[].sp` in `config-local.yaml` or
`config-slurm.yaml`. The template keeps the BFF convention of reading the final
short-MD frame from `md-pos-1.xyz` and printing forces to standard output.

Matching isolated-atom templates are also included:

- `inputs/revpbe0-single-atom-carbon.inp`
- `inputs/revpbe0-single-atom-hydrogen.inp`
- `inputs/revpbe0-single-atom-oxygen.inp`

These are examples for replacing the generated
`../01-prepare/colvars/reference/system-XXX/single-atoms/*/input.inp` files
when single-atom energies should be evaluated at the same revPBE0-D3/ADMM
level as the molecular single points.

Use:

```bash
bff reference config-local.yaml
```

for local execution, or:

```bash
bff reference config-slurm.yaml
```

for Slurm-backed execution.

The workflow stages one run directory per snapshot, runs the short GFN1-xTB
MD plus final single-point calculation, optionally evaluates isolated
single-atom energies, and writes:

- `reference/system-XXX/train.extxyz`
- `reference/system-XXX/valid.extxyz`
- `reference/system-XXX/single-atoms/energies.yaml`
