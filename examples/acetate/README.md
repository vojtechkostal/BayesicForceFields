# Acetate Example

This example uses stage-local configs. For each stage, create the output
directory, copy the needed template files into that directory, edit them if
needed, then run BFF from inside the stage directory. The stage writes its own
outputs to `./`.

## Run Order

```bash
cd examples/acetate

mkdir -p 01-build
cp configs/build-colvars.yaml 01-build/config.yaml
cd 01-build
bff build config.yaml
cd ..

mkdir -p 02-assets
cp configs/prepare-assets.yaml 02-assets/config.yaml
cd 02-assets
bff prepare-assets config.yaml
cd ..

mkdir -p 03-reference
cp configs/evaluate-run-slurm.yaml 03-reference/config-snapshots.yaml
cd 03-reference
bff evaluate-snapshots config-snapshots.yaml
mkdir -p trajectories
# Generate or place reference trajectories under trajectories/system-*/.
cd ..

mkdir -p 03-sample
cp configs/sample-local.yaml 03-sample/config.yaml
cd 03-sample
bff sample config.yaml
cd ..

mkdir -p 04-analyze
cp configs/analyze.yaml 04-analyze/config.yaml
cd 04-analyze
bff analyze config.yaml
cd ..

mkdir -p 05-fit
cp configs/fit.yaml 05-fit/config.yaml
cd 05-fit
bff fit config.yaml
cd ..

mkdir -p 06-learn
cp configs/learn.yaml 06-learn/config.yaml
cd 06-learn
bff learn config.yaml
cd ..

mkdir -p 07-validate
cp configs/validate.yaml 07-validate/config.yaml
cd 07-validate
bff validate config.yaml
cd ..
```

The copied config is the record of what was run for that stage. Stage
directories are runtime outputs and are ignored by git.

## Source Layout

```text
examples/acetate/
  configs/      config templates copied into stage directories
  inputs/       committed molecular inputs and optional reference trajectories
  notebooks/    optional interactive analysis notebooks
```

## Configs

- `build-colvars.yaml`: equilibrates systems and runs seeded production
  trajectories recorded in `./build-manifest.yaml`.
- `prepare-assets.yaml`: reads `../01-build/build-manifest.yaml`, packages
  `./ffmd/`, and stages CP2K inputs under `./reference/`.
- `evaluate-run-local.yaml`: runs CP2K snapshot evaluation on
  `../02-assets/reference/` and writes `train.extxyz` and `valid.extxyz`
  into `./snapshots/`.
- `evaluate-run-slurm.yaml`: same snapshot-evaluation stage through Slurm.
- `sample-local.yaml`: samples force-field parameters and runs local FFMD into
  `./`.
- `analyze.yaml`: compares `../03-sample/` against reference trajectories in
  `../03-reference/trajectories/` and writes QoI datasets into `./`.
- `fit.yaml`: fits surrogate models into `./models/`.
- `learn.yaml`: assigns effective observations, learns the posterior, and
  writes `marginals.pdf`, `qoi-marginals.pdf`, and `corner.pdf` into `./`.
- `validate.yaml`: reruns selected posterior samples into `./`.

## Reference Trajectories

`bff evaluate-snapshots` evaluates short CP2K snapshot jobs and writes
`train.extxyz` and `valid.extxyz`; it does not generate the reference MD
trajectories used by `bff analyze`. After the snapshots are evaluated, generate
those trajectories yourself. You can run AIMD directly from the CP2K inputs in
`02-assets/reference/system-*/md/`, or train a machine-learning potential of
your choice from the evaluated snapshots and use that potential to run the
reference trajectories.

Keep evaluated snapshot datasets in `03-reference/snapshots/`. A good place for
the generated reference trajectories is
`03-reference/trajectories/system-*/trajectory.xtc`, alongside `system.top` and
`system.gro`. The `system.top` and `system.gro` files can be copied from the
matching `02-assets/reference/system-*/` directory.

```text
03-reference/
  snapshots/system-*/      evaluated snapshot datasets
  trajectories/system-*/   reference trajectories for analysis
```

## Variants

- `build-plumed.yaml` uses PLUMED restraint files instead of Colvars.
- `sample-slurm.yaml` runs the sampling campaign through Slurm.

The Slurm configs keep scheduler setup commands in the YAML. Edit those blocks
for your cluster before running them.

## Inputs

- `inputs/common/`: topologies, force-field includes, template coordinates, and
  GROMACS MDP files.
- `inputs/biases/`: Colvars and PLUMED restraint files.
- `inputs/reference-trajectories/`: optional committed AIMD trajectories that
  can be copied into `03-reference/trajectories/`.
- `inputs/reference-inputs/`: optional CP2K input overrides for customized
  reference runs.
- `inputs/restraint.py`: custom distance-distribution QoI for the calcium-bound
  systems.
