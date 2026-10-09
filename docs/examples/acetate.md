# Acetate Walkthrough

The acetate example learns partial charges against three reference systems.
It is a complete template for the stages BFF owns, with the reference MD as an
explicit external handoff rather than a self-contained reproduction. Every
stage is a directory with its config; run the command from inside it:

| Directory | Command | Config |
| --- | --- | --- |
| `01-build` | `bff build` | `config.yaml` (Colvars) or `config-plumed.yaml` |
| `02-reference-md` | external reference MD | optional `label-structures.yaml` |
| `03-sample-parameters` | `bff sample-parameters` | `config.yaml` or `config-slurm.yaml` |
| `04-build-qoi-datasets` | `bff build-qoi-datasets` | `config.yaml` |
| `05-fit-lgp` | `bff fit-lgp` | `config.yaml` |
| `06-learn` | `bff learn` | `config.yaml`, plus `posterior.ipynb` |
| `07-validate` | `bff validate` | `config.yaml` |

```bash
cd examples/acetate
(cd 01-build && bff build config.yaml)
```

Continue the same way with the table above. The repository
`examples/acetate/README.md` has the complete command sequence.

GROMACS is required for build, sampling, and validation; CP2K and Slurm only
for the optional labeling script. Use the Colvars or PLUMED build matching the
config, and edit the `ADAPT` settings (executables, Slurm setup, lengths).
`fit-lgp` runs on the CPU; `bff learn` uses `device: auto` (CUDA when
available).

BFF deliberately does not run the reference MD. Simulate each system
externally, for example with a foundation MLIP, and place one trajectory per
system at `02-reference-md/trajectories/<system_id>/trajectory.xtc`; the
example ships these three. To fine-tune the foundation model first, label
frames of its trajectories under
`02-reference-md/foundation/<system_id>/trajectory.xtc` with
`scripts/label_structures.py` and `label-structures.yaml`; see
[Reference trajectories](../reference-trajectories.md).

Every built system contains `reference/topology.top` and
`reference/coordinates.gro`, which omit declared virtual sites while keeping
the atom order of the system. Every reference trajectory must have the same
atom count and order as this pair.

`inputs/` holds the molecular inputs and `routines.py`, a custom QoI routine;
`02-reference-md/cp2k/` the CP2K inputs of the labeling script.

To adapt the template, replace the inputs, keep the system IDs identical across
configs, set the charge and multiplicity in the CP2K inputs, and update the
parameter bounds, charge constraints, QoI selections, and run lengths.
