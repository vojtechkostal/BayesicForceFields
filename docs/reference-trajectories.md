# Reference Trajectories

`build-qoi-datasets` compares every sampled force field with a reference
trajectory of the same system. BFF does not produce that trajectory: you run
the reference simulation yourself, with whatever level of theory the
calibration targets.

```text
build -> [your reference MD] -> sample-parameters -> build-qoi-datasets
      -> fit-lgp -> learn -> validate
```

## What BFF Provides

`bff build` equilibrates each system and writes a virtual-site-free reference
pair:

```text
systems/<system_id>/reference/topology.top
systems/<system_id>/reference/coordinates.gro
```

The coordinates are the last frame of the seeded classical production run and
are a convenient starting structure for the reference simulation. The
reference trajectory must have the same atom count and atom order as this
pair. `build-qoi-datasets` reads the pair and your trajectory as explicit
inputs for each `system_id`.

## Choosing a Reference

Any trajectory that satisfies the atom-order contract works: ab initio MD, an
existing MLIP, or an external simulation package. Two machine-learned routes
are common:

1. **Foundation model.** Run MD directly with a pretrained foundation MLIP,
   such as a MACE foundation model, starting from `reference/coordinates.gro`.
   This is the fastest route. Check that the model is accurate enough for the
   quantities of interest you calibrate against.
2. **Fine-tuned foundation model.** Run foundation-model MD first, label
   frames from it with a reference electronic-structure method using
   [`scripts/label_structures.py`](#labeling-snapshots-with-cp2k), fine-tune
   the model on those labels, and run the production reference MD with the
   fine-tuned model. Sampling frames from foundation-model MD rather than from
   the classical build trajectory keeps the training data close to the
   configurations the fine-tuned model will visit.

Training, fine-tuning, and running the MLIP happen outside BFF.

## Labeling Snapshots with CP2K

`scripts/label_structures.py` is a standalone helper distributed in the BFF
repository. It is not part of the installed `bff` package or CLI and does not
import BFF. It needs PyYAML, NumPy, MDAnalysis (or ASE for `.traj` input), CP2K,
and Slurm. Use the copy from the release tag matching your BFF version:

```bash
curl -O https://raw.githubusercontent.com/vojtechkostal/BayesicForceFields/v<version>/scripts/label_structures.py
```

The script selects frames from each trajectory, runs one CP2K energy-and-force
calculation per frame and one per isolated element as Slurm job arrays, and
collects the results:

```bash
python label_structures.py run config.yaml      # stage, submit, wait, collect
python label_structures.py prepare config.yaml  # only stage the tasks
python label_structures.py collect config.yaml  # re-collect finished tasks
```

`run` can be repeated after failures or interruptions: finished tasks whose
structure, cell, and CP2K input are unchanged are not resubmitted. The
complete configuration is documented at the top of the script and in
`python label_structures.py --help`; the acetate example contains a full
configuration in `02-reference-md/label-structures.yaml`.

Each CP2K input must read `structure.xyz` and, for periodic systems,
`@INCLUDE cell.inc` inside `&CELL`; it must print forces to standard output and
set the system's charge and multiplicity. Isolated-atom inputs read
`structure.xyz` with their own cell.

Outputs in `output_dir`:

| File | Content |
| --- | --- |
| `train.extxyz`, `valid.extxyz`, `test.extxyz` | Labeled frames: energy (eV), forces (eV/angstrom), optional stress and virials, and `config_type=<system id>`. |
| `energies.yaml` | Isolated-atom energies (eV) keyed by atomic number. Written as `energies.partial.yaml` while any element is missing. |
| `label-results.yaml` | Input fingerprints, selected frames, split membership, and incomplete tasks. |

Element symbols come from the topology or are guessed from atom names by
MDAnalysis. Guessing fails for names such as `CAL` (calcium) or `IW` (a
virtual site); map them explicitly with the system's `elements` key, and use a
virtual-site-free topology such as `reference/coordinates.gro`.
