# Acetate Example

This example uses stage-local configs. For each stage, create the output
directory, copy the matching template to `config.yaml`, edit it if needed, then
run BFF from inside that directory. The stage writes its own outputs to `./`.

## Run Order

```bash
cd examples/acetate

mkdir -p 01-build-colvars
cp configs/build-colvars.yaml 01-build-colvars/config.yaml
cd 01-build-colvars
bff build config.yaml
cd ..

mkdir -p 02-reference-run-local
cp configs/reference-run-local.yaml 02-reference-run-local/config.yaml
cd 02-reference-run-local
bff reference config.yaml
cd ..

mkdir -p 02-reference-import
cp configs/reference-import.yaml 02-reference-import/config.yaml
cd 02-reference-import
bff reference config.yaml
cd ..

mkdir -p 03-sample-local
cp configs/sample-local.yaml 03-sample-local/config.yaml
cd 03-sample-local
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
  inputs/       committed molecular inputs and reference trajectories
  notebooks/    optional interactive analysis notebooks
```

## Configs

- `build-colvars.yaml`: prepares FFMD assets in `./ffmd/` and reference assets
  in `./reference/`.
- `reference-run-local.yaml`: runs CP2K on `../01-build-colvars/reference/` and
  writes `train.extxyz` and `valid.extxyz` into `./`.
- `reference-import.yaml`: imports committed AIMD trajectories into `./`.
- `sample-local.yaml`: samples force-field parameters and runs local FFMD into
  `./`.
- `analyze.yaml`: compares `../03-sample-local/` against
  `../02-reference-import/` and writes QoI datasets into `./`.
- `fit.yaml`: fits surrogate models into `./models/`.
- `learn.yaml`: learns the posterior and writes outputs into `./`.
- `validate.yaml`: reruns selected posterior samples into `./`.

## Variants

- `build-plumed.yaml` uses PLUMED restraint files instead of Colvars.
- `reference-run-slurm.yaml` runs the reference workflow through Slurm.
- `sample-slurm.yaml` runs the sampling campaign through Slurm.

The Slurm configs keep scheduler setup commands in the YAML. Edit those blocks
for your cluster before running them.

## Inputs

- `inputs/common/`: topologies, force-field includes, template coordinates, and
  GROMACS MDP files.
- `inputs/biases/`: Colvars and PLUMED restraint files.
- `inputs/reference-trajectories/`: committed AIMD trajectories imported by
  `reference-import.yaml`.
- `inputs/reference-inputs/`: optional CP2K input overrides for customized
  reference runs.
- `inputs/restraint.py`: custom distance-distribution QoI for the calcium-bound
  systems.
