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

cd ../../03-training-trjs
bff simulate config-local.yaml

cd ../04-qoi
bff qoi config.yaml

cd ../05-train-lgp
bff train config.yaml

cd ../06-learn
bff learn config.yaml

cd ../08-validate
bff validate config.yaml
```

## Important Files

- prepare stage overview:
  [01-prepare/README.md][acetate-prepare-readme]
- shared prepare assets:
  `01-prepare/common/`
- Colvars prepare config:
  [01-prepare/colvars/config.yaml][acetate-prepare-colvars]
- PLUMED prepare config:
  [01-prepare/plumed/config.yaml][acetate-prepare-plumed]
- local simulate config:
  [03-training-trjs/config-local.yaml][acetate-simulate]
- QoI config:
  [04-qoi/config.yaml][acetate-qoi]
- train config:
  [05-train-lgp/config.yaml][acetate-train]
- learn config:
  [06-learn/config.yaml][acetate-learn]
- validate config:
  [08-validate/config.yaml][acetate-validate]

## Notebooks

- `06-learn/interactive.ipynb`
  walks through surrogate training, posterior sampling, posterior sample export,
  and basic posterior plots interactively.
- `07-visualize/visualize.ipynb`
  focuses on plotting and inspection only.

## Notes

- The walkthrough uses the Colvars preparation variant by default.
- Shared topologies, force-field files, template coordinates, and MDP inputs
  live in `01-prepare/common/`.
- Prepared training assets live under
  `01-prepare/colvars/ace-colvars/training/system-XXX/`.
- The QoI step demonstrates both builtin routines and a custom routine loaded
  from `./restraint.py:distance_distribution`.
- The learn stage writes posterior chains in `06-learn/` and the interactive
  notebook can also export `posterior-samples.yaml` for stage `08`.

[acetate-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/acetate
[acetate-prepare-readme]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/01-prepare/README.md
[acetate-prepare-colvars]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/01-prepare/colvars/config.yaml
[acetate-prepare-plumed]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/01-prepare/plumed/config.yaml
[acetate-simulate]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/03-training-trjs/config-local.yaml
[acetate-qoi]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/04-qoi/config.yaml
[acetate-train]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/05-train-lgp/config.yaml
[acetate-learn]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/06-learn/config.yaml
[acetate-validate]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/acetate/08-validate/config.yaml
