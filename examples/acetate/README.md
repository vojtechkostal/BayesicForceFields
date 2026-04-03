# Acetate Example

This example demonstrates the intended BFF workflow on acetate partial charges.
Each stage has a small, explicit configuration file and a predictable output
location.

## Stages

1. [01-prepare/](01-prepare/)
   contains two preparation variants:
   [common/](01-prepare/common/),
   [colvars/](01-prepare/colvars/)
   and
   [plumed/](01-prepare/plumed/).
2. [02-reference-trjs/](02-reference-trjs/)
   contains reference trajectories.
3. [03-training-trjs/](03-training-trjs/)
   runs sampled GROMACS campaigns.
4. [04-qoi/](04-qoi/)
   computes quantities of interest from the trainset and reference data.
5. [05-train-lgp/](05-train-lgp/)
   trains the surrogate models.
6. [06-learn/](06-learn/)
   runs posterior inference and contains an interactive notebook.
7. [07-visualize/](07-visualize/)
   contains a plotting-only notebook.
8. [08-validate/](08-validate/)
   reruns selected posterior samples for validation.

## Recommended Run Order

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

## Key Files

- [01-prepare/README.md](01-prepare/README.md)
  explains the two preparation variants.
- [03-training-trjs/config-local.yaml](03-training-trjs/config-local.yaml)
  runs the sampled training campaign locally.
- [04-qoi/config.yaml](04-qoi/config.yaml)
  computes QoIs from the trainset and reference set.
- [05-train-lgp/config.yaml](05-train-lgp/config.yaml)
  trains the surrogate models.
- [06-learn/config.yaml](06-learn/config.yaml)
  runs posterior inference from the selected surrogates.
- [06-learn/interactive.ipynb](06-learn/interactive.ipynb)
  demonstrates training, learning, sample export, and plotting interactively.
- [07-visualize/visualize.ipynb](07-visualize/visualize.ipynb)
  focuses on posterior plotting only.
- [08-validate/config.yaml](08-validate/config.yaml)
  reruns selected posterior samples for validation.

## Current API Shape

The example reflects the streamlined package APIs:

- `01-prepare/common/` holds shared topologies, force-field files, template
  coordinates, and MDP inputs used by both bias variants.
- prepare systems define residue `templates`, per-system `mdp` files, and
  optional bias files.
- prepared training assets are written into one directory per system under
  `training/window-XXX/`.
- simulate and validate configs point to those prepared asset directories with
  `systems[].assets` and system-specific `n_steps`.
- the QoI workflow uses `trainset`, `refset`, `run`, and `output`.
- train writes surrogate model files.
- learn reads only `specs`, `models`, and `mcmc`.
- routines always use a single `routine` key. Bare names are builtins and
  `path.py:function` strings load custom routines.
