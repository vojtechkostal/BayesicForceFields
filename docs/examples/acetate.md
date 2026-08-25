# Acetate Walkthrough

The acetate example calibrates partial charges against three reference systems.
It is a complete template for the stages BFF owns, with external MLIP training
and reference MD as an explicit handoff rather than a self-contained
reproduction. Its numbered config templates mirror the runtime stages:

| Stage | Command | Config |
| --- | --- | --- |
| `01-build` | `bff build` | `01-build-colvars.yaml` or `01-build-plumed.yaml` |
| `02-reference-snapshots` | `bff label-snapshots` | `02-reference-snapshots-local.yaml` or `02-reference-snapshots-slurm.yaml` |
| `02-reference-md` | external MLIP training and MD | external handoff |
| `03-sample` | `bff sample-parameters` | `03-sample-local.yaml` or `03-sample-slurm.yaml` |
| `04-qoi` | `bff build-qoi-datasets` | `04-build-qoi-datasets.yaml` |
| `05-lgp` | `bff fit-lgp` | `05-fit-lgp.yaml` |
| `06-learn` | `bff learn` | `06-learn.yaml` |
| `07-validate` | `bff validate` | `07-validate.yaml` |

For each BFF stage, create the directory, copy its config to `config.yaml`, and
run the command there. For example:

```bash
cd examples/acetate
mkdir -p 01-build
cp configs/01-build-colvars.yaml 01-build/config.yaml
(cd 01-build && bff build config.yaml)

mkdir -p 02-reference-snapshots
cp configs/02-reference-snapshots-local.yaml 02-reference-snapshots/config.yaml
(cd 02-reference-snapshots && bff label-snapshots config.yaml)
```

Continue with the same pattern using the table above. The repository
`examples/acetate/README.md` contains the complete command sequence.

GROMACS is required for build, sampling, and validation, and CP2K is required
for labeling. Use the Colvars or PLUMED build matching the selected template,
edit cluster-specific Slurm setup commands, and change `device: cuda` to
`device: cpu` in the fit and learn configs when no CUDA GPU is available.

`label-snapshots` produces `train.extxyz`, `test.extxyz`, optional isolated-atom
energies, and `label-results.yaml`. BFF deliberately does not train or run an
MLIP. Train from the per-system EXTXYZ files and simulate it externally, then
place one reference trajectory per system at
`02-reference-md/trajectories/<system_id>/trajectory.xtc`.

Every built system contains `reference/topology.top` and
`reference/coordinates.gro`. These files omit declared virtual sites while
preserving the atom order used by the external reference trajectory.
Every trajectory must have the same atom count and order as this pair and must
omit its declared virtual sites.

`inputs/reference-inputs/` contains per-system xTB short-MD inputs, revPBE-D3
MD, single-point, and isolated-atom inputs, and revPBE0-D3 single-point and
isolated-atom inputs.

To adapt the template, replace the molecular and MD inputs, preserve stable
system IDs across configs, supply charge- and multiplicity-correct CP2K inputs,
and update parameter bounds, charge constraints, QoI selections, run lengths,
scheduler commands, and the fitting device.
