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
cd examples/acetate/01-prepare/colvars
bff prepare config.yaml

cd ../../02-reference-data
bff reference config-local.yaml

cd ../02-training-data
bff trainset config-local.yaml

cd ../03-qoi
bff qoi config.yaml

cd ../04-train-lgp
bff train config.yaml

cd ../05-learning
bff learn config.yaml

cd ../07-validate
bff validate config.yaml
```

## Important Files

- prepare stage overview:
  [01-prepare/README.md][acetate-prepare-readme]
- shared prepare assets:
  `01-prepare/common/`
- Colvars prepare config:
  [01-prepare/colvars/config.yaml][acetate-prepare-colvars]
- local reference config:
  [02-reference-data/config-local.yaml][acetate-reference]
- reference CP2K overrides and trajectories:
  `02-reference-data/inputs/`, `02-reference-data/trajectories/`
- PLUMED prepare config:
  [01-prepare/plumed/config.yaml][acetate-prepare-plumed]
- local force-field trajectory config:
  [02-training-data/config-local.yaml][acetate-trainset]
- QoI config:
  [03-qoi/config.yaml][acetate-qoi]
- train config:
  [04-train-lgp/config.yaml][acetate-train]
- learn config:
  [05-learning/config.yaml][acetate-learn]
- validate config:
  [07-validate/config.yaml][acetate-validate]

## Notebooks

- `05-learning/interactive.ipynb`
  walks through surrogate training, posterior sampling, posterior sample export,
  and basic posterior plots interactively.
- `06-visualize/visualize.ipynb`
  focuses on plotting and inspection only.

## Notes

- The walkthrough uses the Colvars preparation variant by default.
- Shared topologies, force-field files, template coordinates, and MDP inputs
  live in `01-prepare/common/`.
- `bff reference` consumes the staged CP2K reference inputs under
  `01-prepare/colvars/reference/system-XXX/` and writes runnable
  outputs plus collected extxyz files under
  `02-reference-data/reference-assets/system-XXX/`.
- `02-reference-data/inputs/` contains optional CP2K input overrides, and
  `02-reference-data/trajectories/` contains the ab initio reference
  trajectories used by QoI.
- `02-training-data/` contains force-field trajectory campaign configs and
  generated trainset outputs.
- Prepared training assets live under
  `01-prepare/colvars/ace-colvars/training/system-XXX/`.
- The QoI step demonstrates both builtin routines and a custom routine loaded
  from `./restraint.py:distance_distribution`.
- The learn stage writes posterior chains in `05-learning/` and the interactive
  notebook can also export `posterior-samples.yaml` for `07-validate`.

[acetate-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/acetate
[acetate-prepare-readme]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/01-prepare/README.md
[acetate-prepare-colvars]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/01-prepare/colvars/config.yaml
[acetate-reference]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/02-reference-data/config-local.yaml
[acetate-prepare-plumed]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/01-prepare/plumed/config.yaml
[acetate-trainset]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/02-training-data/config-local.yaml
[acetate-qoi]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/03-qoi/config.yaml
[acetate-train]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/04-train-lgp/config.yaml
[acetate-learn]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/05-learning/config.yaml
[acetate-validate]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/07-validate/config.yaml
