# Acetate

The acetate example learns the partial charges of acetate (charge −0.8, the
scaled charge of ECC-type force fields) from two systems:

| System | What is simulated | QoI | Reference |
| --- | --- | --- | --- |
| `acetate` | 1 acetate + 128 water | acetate–water RDFs (built-in `rdf`) | MLIP trajectory, `02-reference-md/acetate.xtc` |
| `calcium-acetate` | 1 acetate + 1 Ca²⁺ + 128 water, Ca–C2 distance biased by Colvars metadynamics | the Ca–C2 PMF (custom routine `inputs/pmf.py`) | MLIP metadynamics PMF, `02-reference-md/calcium-acetate.pmf` |

## Layout

```text
acetate/
├── inputs/
│   ├── acetate.top, calcium-acetate.top, ace.gro, ff/, mdp/
│   ├── colvars.dat               Ca–C2 well-tempered metadynamics
│   └── pmf.py                    QoI routine reading a PMF file
├── 01-build/                     bff build
├── 02-reference-md/              reference data (outside BFF)
│   ├── acetate.xtc
│   ├── calcium-acetate.pmf
│   ├── label-structures.yaml     optional CP2K labeling for MLIP fine-tuning
│   └── cp2k/
├── 03-sample-parameters/         bff sample-parameters
├── 04-build-qoi-datasets/        bff build-qoi-datasets
├── 05-fit-lgp/                   bff fit-lgp
├── 06-learn/                     bff learn, posterior.ipynb
└── 07-validate/                  bff validate
```

## Run

You need GROMACS with Colvars (GROMACS 2024 or newer) and PyTorch. Edit the
settings marked `ADAPT` in the configs (executables, run lengths, sample
count), then run each stage from its directory:

```bash
(cd 01-build && bff build config.yaml)
(cd 03-sample-parameters && bff sample-parameters config.yaml)
(cd 04-build-qoi-datasets && bff build-qoi-datasets config.yaml)
(cd 05-fit-lgp && bff fit-lgp config.yaml)
(cd 06-learn && bff learn config.yaml)
(cd 07-validate && bff validate config.yaml)
```

`06-learn/posterior.ipynb` opens `outputs/results.pt` to inspect the
posterior.

## Run on Slurm

`03-sample-parameters/config.yaml` runs locally. To run the samples as Slurm
job arrays, set `job_scheduler: slurm`, uncomment the block at the end of the
file, and adapt it (resources, `module load` lines, and `gmx_cmd`, for example
`srun gmx_mpi`). With `max_restarts`, samples stopped by the time limit
continue from their checkpoints; see
[time limits and restarts](../configuration/sample-parameters.md#time-limits-and-restarts).
Run `bff sample-parameters` in `tmux` or `screen`: it submits the arrays and
waits for them. `07-validate` takes the same block.

## How the PMF becomes a QoI

`inputs/colvars.dat` biases the Ca–C2 distance with the same well-tempered
metadynamics as the reference, and GROMACS writes the resulting PMF to
`production.pmf`. `store: [xtc, pmf]` in `03-sample-parameters` keeps that
file for every sample, and `04-build-qoi-datasets` gives it to the custom
routine as the `pmf` input:

```yaml
- name: pmf
  callable: ../inputs/pmf.py:read_pmf
  systems: [calcium-acetate]
  inputs: [pmf]
  options: {range: [0.27, 0.65], points: 39}  # nm
```

For the reference, `reference.systems` names the file under the same role:

```yaml
- system_id: calcium-acetate
  inputs:
    pmf: ../02-reference-md/calcium-acetate.pmf
```

`read_pmf` interpolates the profile onto 39 distances between 0.27 and 0.65 nm
and shifts it to zero mean, because a PMF is defined only up to a constant.
The [routine interface](../configuration/build-qoi-datasets.md#routine-interface)
explains how to write such routines.

## The reference data

BFF does not generate the reference. The example ships both reference files:
`acetate.xtc` is a short trajectory in the atom order of
`01-build/systems/acetate/reference/`, and `calcium-acetate.pmf` comes from
10 ns of metadynamics with the MACE foundation model mace-mh-1 fine-tuned on
DFT labels (see the header of the file). For your own system, run the
reference MD from `01-build/systems/<system_id>/reference/coordinates.gro`;
to fine-tune an MLIP first, label frames with
`scripts/label_structures.py` and `02-reference-md/label-structures.yaml`.
See [Reference data](../reference-trajectories.md).

## Adapt it to your system

Replace `inputs/`, keep each `system_id` identical in all configs, and update
the parameter `bounds`, `charge_constraints`, the QoI routines and their
selections, and the run lengths. For CP2K labeling, set the charge and
multiplicity in `02-reference-md/cp2k/`.
