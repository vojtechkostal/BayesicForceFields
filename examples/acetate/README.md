# Acetate Workflow

This is a complete template for the stages BFF owns, not a one-command or
self-contained reproduction. Reference MLIP training and molecular dynamics
remain an explicit external handoff between snapshot labeling and QoI
construction.

Run each BFF command from its stage directory. The numeric config prefix shows
where the config belongs; copy it to `config.yaml` so all relative paths resolve
from the stage directory.

| Stage | Responsibility | Config or handoff | Main output |
| --- | --- | --- | --- |
| `01-build` | Build and equilibrate the classical systems. | `01-build-colvars.yaml` or `01-build-plumed.yaml` | Classical systems plus vsite-free reference topology/coordinates. |
| `02-reference-snapshots` | Extract and label CP2K reference snapshots. | `02-reference-snapshots-local.yaml` or `02-reference-snapshots-slurm.yaml` | Per-system `train.extxyz` and `test.extxyz`. |
| `02-reference-md` | Train and simulate an MLIP outside BFF. | External workflow | Reference trajectories under `trajectories/<system_id>/trajectory.xtc`. |
| `03-sample` | Design parameters and run classical training simulations. | `03-sample-local.yaml` or `03-sample-slurm.yaml` | `specs.yaml`, `samples.yaml`, and sampled trajectories. |
| `04-qoi` | Analyze reference and sampled systems. | `04-build-qoi-datasets.yaml` | One `qoi/<name>.pt` dataset per routine. |
| `05-lgp` | Fit local-GP surrogate models. | `05-fit-lgp.yaml` | `models/<name>.lgp`. |
| `06-learn` | Sample the Bayesian posterior. | `06-learn.yaml` | `outputs/`, `plots/`, and `learn.log`. |
| `07-validate` | Simulate selected posterior parameters. | `07-validate.yaml` | Validation campaign manifest and trajectories. |

## Prerequisites

- Install BFF and PyTorch as described in the main installation guide.
- Provide GROMACS for build, parameter sampling, and validation; provide CP2K
  for snapshot labeling.
- Use a GROMACS build with Colvars or PLUMED support for the matching build
  template.
- Supply an external MLIP trainer and MD engine for `02-reference-md`.
- Edit every Slurm `setup` and `teardown` block for the target cluster.
- The fit and learn templates use `device: cuda`; change both to `cpu` on a
  machine without an available CUDA GPU.

## Run the BFF stages

```bash
mkdir -p 01-build
cp configs/01-build-colvars.yaml 01-build/config.yaml
(cd 01-build && bff build config.yaml)

mkdir -p 02-reference-snapshots
cp configs/02-reference-snapshots-local.yaml 02-reference-snapshots/config.yaml
(cd 02-reference-snapshots && bff label-snapshots config.yaml)

mkdir -p 03-sample
cp configs/03-sample-local.yaml 03-sample/config.yaml
(cd 03-sample && bff sample-parameters config.yaml)

mkdir -p 04-qoi
cp configs/04-build-qoi-datasets.yaml 04-qoi/config.yaml
(cd 04-qoi && bff build-qoi-datasets config.yaml)

mkdir -p 05-lgp
cp configs/05-fit-lgp.yaml 05-lgp/config.yaml
(cd 05-lgp && bff fit-lgp config.yaml)

mkdir -p 06-learn
cp configs/06-learn.yaml 06-learn/config.yaml
(cd 06-learn && bff learn config.yaml)

# Validate directly from 06-learn/outputs/posterior.pt. The notebook also
# demonstrates exporting posterior-samples.yaml for the alternative explicit
# parameter-file mode.
mkdir -p 07-validate
cp configs/07-validate.yaml 07-validate/config.yaml
(cd 07-validate && bff validate config.yaml)
```

Between labeling and QoI analysis, train an MLIP from each system's
`train.extxyz` and `test.extxyz`. Run reference MD and place these three
trajectories:

```text
02-reference-md/trajectories/acetate/trajectory.xtc
02-reference-md/trajectories/acetate-contact/trajectory.xtc
02-reference-md/trajectories/acetate-separated/trajectory.xtc
```

Each trajectory must match the corresponding
`01-build/systems/<system_id>/reference/topology.top` and
`reference/coordinates.gro`: same atom count, same atom order, and no declared
virtual sites. BFF deliberately does not own this MLIP stage.

`inputs/common/` contains molecular inputs, `inputs/biases/` contains Colvars
and PLUMED restraints, and `inputs/reference-inputs/` contains xTB short-MD
inputs, revPBE-D3 MD, single-point, and isolated-atom inputs, and revPBE0-D3
single-point and isolated-atom inputs.

Each built system includes `reference/topology.top` and
`reference/coordinates.gro`. This atom-order-matched, vsite-free pair is the
starting topology for the external reference MD. The custom trajectory routine
used during QoI construction is defined in `inputs/restraint.py`.

## Adapt the Template

For a new system, update the build topologies, coordinate templates, MDP and
bias files; keep the same stable `system_id` through every stage; supply CP2K
inputs with matching charge and multiplicity; choose parameter bounds and
charge constraints; replace the QoI selections and custom routines; then set
simulation lengths, sample counts, scheduler commands, and the compute device
for the available hardware.
