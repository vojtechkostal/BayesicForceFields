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
directory, copy the needed template files into it, edit them there, and run BFF
from inside that directory. Each stage writes its own outputs to `./`.

```bash
cd examples/acetate

mkdir -p 01-build
cp configs/build-colvars.yaml 01-build/config.yaml
cd 01-build
bff build config.yaml
cd ..

mkdir -p 02-reference
cp configs/prepare-reference.yaml 02-reference/config.yaml
cd 02-reference
bff prepare-reference config.yaml
cd ..

mkdir -p 03-reference
cp configs/evaluate-run-local.yaml 03-reference/config-snapshots.yaml
cd 03-reference
bff evaluate-snapshots config-snapshots.yaml
mkdir -p trajectories
# Generate or place reference trajectories under trajectories/<system_id>/.
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

mkdir -p 05-lgpfit
cp configs/lgpfit.yaml 05-lgpfit/config.yaml
cd 05-lgpfit
bff lgpfit config.yaml
cd ..

mkdir -p 06-learn
cp configs/learn.yaml 06-learn/config.yaml
cd 06-learn
bff learn config.yaml
# Export selected explicit draws from output/posterior.pt to
# posterior-samples.yaml with PosteriorResults.sample_posterior.
cd ..

mkdir -p 07-validate
cp configs/validate.yaml 07-validate/config.yaml
cd 07-validate
bff validate config.yaml
cd ..
```

The copied config is the record of what was run for that stage. Stage
directories are generated outputs and are ignored by git.
The learn stage writes restartable Torch artifacts under `output/` and the
mandatory `marginals.pdf`, `qoi-marginals.pdf`, and `corner.pdf` under
`plots/`.

The `03-reference` stage keeps snapshot datasets and analysis trajectories as
sibling directories:

```text
03-reference/
  snapshots/systems/<system_id>/
  trajectories/<system_id>/
```

## Layout

```text
examples/acetate/
  configs/      config templates copied into stage directories
  inputs/       committed molecular inputs and optional reference trajectories
  notebooks/    optional interactive notebooks
```

## Main Configs

- Colvars build config:
  [configs/build-colvars.yaml][acetate-build-colvars]
- reference-preparation config:
  [configs/prepare-reference.yaml][acetate-prepare-reference]
- local CP2K snapshot evaluation config:
  [configs/evaluate-run-local.yaml][acetate-evaluate-run]
- local sampling config:
  [configs/sample-local.yaml][acetate-sample]
- analyze config:
  [configs/analyze.yaml][acetate-analyze]
- LGP-fit config:
  [configs/lgpfit.yaml][acetate-lgpfit]
- learn config:
  [configs/learn.yaml][acetate-learn]
- validate config:
  [configs/validate.yaml][acetate-validate]

## Variants

- PLUMED build config:
  [configs/build-plumed.yaml][acetate-build-plumed]
- Slurm snapshot evaluation config:
  [configs/evaluate-run-slurm.yaml][acetate-evaluate-slurm]
- Slurm sampling config:
  [configs/sample-slurm.yaml][acetate-sample-slurm]

The Slurm configs keep scheduler setup commands in the YAML. Edit those blocks
for your cluster before running them.

## Inputs

- shared force-field files, topologies, template coordinates, and MDP inputs:
  `inputs/common/`
- Colvars and PLUMED restraint files:
  `inputs/biases/`
- optional ab initio reference trajectories for QoI analysis:
  `inputs/reference-trajectories/`
- optional CP2K input overrides for customized reference runs:
  `inputs/reference-inputs/`
- custom QoI routine:
  [inputs/restraint.py][acetate-restraint]

[acetate-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/acetate
[acetate-build-colvars]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/build-colvars.yaml
[acetate-build-plumed]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/build-plumed.yaml
[acetate-prepare-reference]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/prepare-reference.yaml
[acetate-evaluate-run]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/evaluate-run-local.yaml
[acetate-evaluate-slurm]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/evaluate-run-slurm.yaml
[acetate-sample]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/sample-local.yaml
[acetate-sample-slurm]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/sample-slurm.yaml
[acetate-analyze]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/analyze.yaml
[acetate-lgpfit]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/lgpfit.yaml
[acetate-learn]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/learn.yaml
[acetate-validate]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/validate.yaml
[acetate-restraint]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/inputs/restraint.py
