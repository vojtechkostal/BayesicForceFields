# Reference Data

**BFF does not generate the reference data.** `build-qoi-datasets` compares
every sampled force field with a reference simulation of the same system, and
you run that simulation yourself, at the level of theory the force field
should reproduce.

## What BFF provides

`bff build` writes, for every system, a virtual-site-free topology and
coordinates to start the reference from:

```text
systems/<system_id>/reference/topology.top
systems/<system_id>/reference/coordinates.gro
```

A reference trajectory must contain the same atoms in the same order as this
pair. A reference can also be a file instead of a trajectory, for example a
PMF that a [file-based QoI routine](configuration/build-qoi-datasets.md#routine-interface)
reads, as in the [acetate example](examples/acetate.md).

## Choosing a reference

- **Ab initio MD** is the most direct reference, if you can afford long enough
  trajectories for your QoIs.
- **A machine-learned interatomic potential (MLIP)** reaches longer
  trajectories and enhanced sampling (for example a PMF). A pretrained
  foundation model, such as a MACE foundation model, can be used directly;
  check that it is accurate enough for your QoIs.
- **A fine-tuned foundation model** is what we recommend: run MD with the
  foundation model, label frames of it with DFT using
  [`label_structures.py`](#labeling-snapshots-with-cp2k), fine-tune on those
  labels, and run the reference MD with the fine-tuned model. Frames from
  foundation-model MD stay close to the configurations the fine-tuned model
  will visit.

Training and running the MLIP happen outside BFF.

## Labeling Snapshots with CP2K

`scripts/label_structures.py` is a standalone helper in the BFF repository.
**It is not installed by `pip`** and is not a `bff` command; it does not
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
