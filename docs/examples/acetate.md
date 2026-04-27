# Acetate Walkthrough

Source files:

- [examples/acetate/][acetate-root]

## Goal

The acetate example optimizes acetate partial charges using three systems:

- aqueous acetate
- acetate with calcium at contact distance
- acetate with calcium in a solvent-shared configuration

## Stage-Local Workflow

The config files under `configs/` are templates. For each stage, make the stage
directory, copy the template to `config.yaml`, edit it there, and run BFF from
inside that directory. Each stage writes its own outputs to `./`.

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
directories are generated outputs and are ignored by git.

## Layout

```text
examples/acetate/
  configs/      config templates copied into stage directories
  inputs/       committed molecular inputs and reference trajectories
  notebooks/    optional interactive notebooks
```

## Main Configs

- Colvars build config:
  [configs/build-colvars.yaml][acetate-build-colvars]
- local CP2K reference-run config:
  [configs/reference-run-local.yaml][acetate-reference-run]
- imported AIMD reference config:
  [configs/reference-import.yaml][acetate-reference-import]
- local sampling config:
  [configs/sample-local.yaml][acetate-sample]
- analyze config:
  [configs/analyze.yaml][acetate-analyze]
- fit config:
  [configs/fit.yaml][acetate-fit]
- learn config:
  [configs/learn.yaml][acetate-learn]
- validate config:
  [configs/validate.yaml][acetate-validate]

## Variants

- PLUMED build config:
  [configs/build-plumed.yaml][acetate-build-plumed]
- Slurm reference-run config:
  [configs/reference-run-slurm.yaml][acetate-reference-slurm]
- Slurm sampling config:
  [configs/sample-slurm.yaml][acetate-sample-slurm]

The Slurm configs keep scheduler setup commands in the YAML. Edit those blocks
for your cluster before running them.

## Inputs

- shared force-field files, topologies, template coordinates, and MDP inputs:
  `inputs/common/`
- Colvars and PLUMED restraint files:
  `inputs/biases/`
- ab initio reference trajectories imported for QoI analysis:
  `inputs/reference-trajectories/`
- optional CP2K input overrides for customized reference runs:
  `inputs/reference-inputs/`
- custom QoI routine:
  [inputs/restraint.py][acetate-restraint]

[acetate-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/acetate
[acetate-build-colvars]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/build-colvars.yaml
[acetate-build-plumed]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/build-plumed.yaml
[acetate-reference-run]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/reference-run-local.yaml
[acetate-reference-slurm]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/reference-run-slurm.yaml
[acetate-reference-import]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/reference-import.yaml
[acetate-sample]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/sample-local.yaml
[acetate-sample-slurm]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/sample-slurm.yaml
[acetate-analyze]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/analyze.yaml
[acetate-fit]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/fit.yaml
[acetate-learn]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/learn.yaml
[acetate-validate]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/validate.yaml
[acetate-restraint]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/inputs/restraint.py
