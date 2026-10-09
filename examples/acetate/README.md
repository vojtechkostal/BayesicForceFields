# Acetate

A complete template for the stages BFF owns: it learns the partial charges of
aqueous acetate, alone and with a calcium ion at two biased distances, from
reference QoIs. The reference molecular dynamics is an explicit external
handoff; see the
[reference-trajectory guide](https://vojtechkostal.github.io/BayesicForceFields/reference-trajectories/).

Each stage has its own directory with its config; run `bff` from inside it, so
that all paths resolve relative to the config.

| Directory | Stage | What it does | Main output |
| --- | --- | --- | --- |
| `01-build` | `bff build` | Build and equilibrate the systems. | `systems/<id>/` with the reference topology and coordinates |
| `02-reference-md` | external | Reference MD; optional CP2K labeling for MLIP fine-tuning. | `trajectories/<id>/trajectory.xtc` |
| `03-sample-parameters` | `bff sample-parameters` | Sample charges, run classical MD. | `specs.yaml`, `samples.yaml` |
| `04-build-qoi-datasets` | `bff build-qoi-datasets` | Analyze reference and sampled trajectories. | `qoi/<name>.pt` |
| `05-fit-lgp` | `bff fit-lgp` | Fit surrogate models. | `models/<name>.lgp` |
| `06-learn` | `bff learn` | Sample the posterior. | `outputs/results.pt`, `plots/` |
| `07-validate` | `bff validate` | Simulate posterior parameters. | validation campaign |

`inputs/` holds the molecular inputs (coordinates, topologies, force-field
`ff/`, `mdp/`, Colvars and PLUMED `biases/`) and `routines.py`, a custom QoI
routine. Replace them to adapt the example to your own system.

## Prerequisites

- BFF and PyTorch, as in the installation guide.
- GROMACS for stages 01, 03, and 07, with Colvars or PLUMED support to match
  the bias in `01-build` (`config.yaml` uses Colvars, `config-plumed.yaml`
  PLUMED).
- An engine for the reference MD, for example a foundation MLIP.
- A Slurm cluster only for `03-sample-parameters/config-slurm.yaml` and the
  optional CP2K labeling.

Settings you must adapt are marked `ADAPT` in the configs: executables, Slurm
setup, simulation lengths, and sample counts.

## Run

```bash
(cd 01-build && bff build config.yaml)
# 02-reference-md: provide the reference trajectories, see below
(cd 03-sample-parameters && bff sample-parameters config.yaml)
(cd 04-build-qoi-datasets && bff build-qoi-datasets config.yaml)
(cd 05-fit-lgp && bff fit-lgp config.yaml)
(cd 06-learn && bff learn config.yaml)
(cd 07-validate && bff validate config.yaml)
```

`06-learn/posterior.ipynb` loads `outputs/results.pt` to inspect the posterior,
the MAP, and the plots.

## Reference MD

`02-reference-md/trajectories/<system_id>/trajectory.xtc` are the reference
trajectories of the three systems (`acetate`, `acetate-contact`,
`acetate-separated`) and are committed with this example. For your own system,
run the reference MD outside BFF from
`01-build/systems/<system_id>/reference/coordinates.gro`. Each trajectory must
match that `reference/topology.top`: same atom count and order, no virtual
sites.

To fine-tune a foundation MLIP instead of using it directly, run
foundation-model MD into `02-reference-md/foundation/<system_id>/trajectory.xtc`
and label frames with CP2K on Slurm
(`scripts/label_structures.py`, not a `bff` command; from an installed package,
download it from the release tag):

```bash
(cd 02-reference-md && python ../../../scripts/label_structures.py run label-structures.yaml)
```

Fine-tune on the resulting `labels/{train,valid,test}.extxyz` with the
isolated-atom energies in `labels/energies.yaml`, then run the reference MD
with the fine-tuned model. The CP2K inputs are in `02-reference-md/cp2k/`; set
the charge and multiplicity in them for your own system.
