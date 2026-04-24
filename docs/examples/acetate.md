# Acetate Walkthrough

Source files:

- [examples/acetate/][acetate-root]

## Goal

The acetate example optimizes acetate partial charges using three systems:

- aqueous acetate
- acetate with calcium at contact distance
- acetate with calcium in a solvent-shared configuration

## Stage Order

```bash
cd examples/acetate
bff build configs/build-colvars.yaml
bff reference configs/reference-run-local.yaml
bff reference configs/reference-import.yaml
bff sample configs/sample-local.yaml
bff analyze configs/analyze.yaml
bff fit configs/fit.yaml
bff learn configs/learn.yaml
bff validate configs/validate.yaml
```

## Layout

The example root keeps committed source assets under `inputs/`, runnable YAML
files under `configs/`, and generated outputs in numbered stage directories such
as `01-build-colvars/`, `02-reference-run-local/`, `03-sample-local/`, or
`06-learn/`. Those stage directories are intentionally ignored by git.

## Important Files

- Colvars build config:
  [configs/build-colvars.yaml][acetate-build-colvars]
- PLUMED build config:
  [configs/build-plumed.yaml][acetate-build-plumed]
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

## Inputs

- shared force-field files, topologies, template coordinates, and MDP inputs:
  `inputs/common/`
- Colvars and PLUMED restraint files:
  `inputs/biases/`
- optional CP2K input overrides for the Slurm reference run:
  `inputs/reference-inputs/`
- ab initio reference trajectories imported for QoI analysis:
  `inputs/reference-trajectories/`
- custom QoI routine:
  [inputs/restraint.py][acetate-restraint]

## Notes

- The build stage writes prepared FFMD assets under `01-build-*/ffmd/` and
  staged CP2K assets under `01-build-*/reference/`.
- `reference-run-*.yaml` produces canonical `train.extxyz` and `valid.extxyz`
  datasets from staged CP2K jobs.
- `reference-import.yaml` separately copies the AIMD trajectories into the
  canonical `system.top`, `system.gro`, and `trajectory.*` layout consumed by
  `bff analyze`.
- Interactive notebooks live under `notebooks/`.

[acetate-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/acetate
[acetate-build-colvars]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/build-colvars.yaml
[acetate-build-plumed]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/build-plumed.yaml
[acetate-reference-run]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/reference-run-local.yaml
[acetate-reference-import]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/reference-import.yaml
[acetate-sample]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/sample-local.yaml
[acetate-analyze]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/analyze.yaml
[acetate-fit]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/fit.yaml
[acetate-learn]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/learn.yaml
[acetate-validate]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/configs/validate.yaml
[acetate-restraint]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/inputs/restraint.py
