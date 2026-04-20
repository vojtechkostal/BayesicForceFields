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
2. [02-reference-data/](02-reference-data/)
   contains reference configs, CP2K input overrides, ab initio reference
   trajectories, and generated `reference-assets/` outputs.
3. [02-training-data/](02-training-data/)
   contains force-field trajectory campaign configs and generated trainset
   outputs.
4. [03-qoi/](03-qoi/)
   computes quantities of interest from the trainset and reference data.
5. [04-train-lgp/](04-train-lgp/)
   trains the surrogate models.
6. [05-learning/](05-learning/)
   runs posterior inference and contains an interactive notebook.
7. [06-visualize/](06-visualize/)
   contains a plotting-only notebook.
8. [07-validate/](07-validate/)
   reruns selected posterior samples for validation.

## Recommended Run Order

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

## Key Files

- [01-prepare/README.md](01-prepare/README.md)
  explains the two preparation variants.
- [02-reference-data/config-local.yaml](02-reference-data/config-local.yaml)
  runs the staged CP2K reference jobs locally after `bff prepare`.
- [02-reference-data/inputs/](02-reference-data/inputs/)
  contains optional CP2K input overrides.
- [02-reference-data/trajectories/](02-reference-data/trajectories/)
  contains the reference trajectories consumed by the QoI stage.
- [02-training-data/config-local.yaml](02-training-data/config-local.yaml)
  runs the sampled force-field trajectory campaign locally.
- [03-qoi/config.yaml](03-qoi/config.yaml)
  computes QoIs from the trainset and reference set.
- [04-train-lgp/config.yaml](04-train-lgp/config.yaml)
  trains the surrogate models.
- [05-learning/config.yaml](05-learning/config.yaml)
  runs posterior inference from the selected surrogates.
- [05-learning/interactive.ipynb](05-learning/interactive.ipynb)
  demonstrates training, learning, sample export, and plotting interactively.
- [06-visualize/visualize.ipynb](06-visualize/visualize.ipynb)
  focuses on posterior plotting only.
- [07-validate/config.yaml](07-validate/config.yaml)
  reruns selected posterior samples for validation.

## Current API Shape

The example reflects the streamlined package APIs:

- `01-prepare/common/` holds shared topologies, force-field files, template
  coordinates, and MDP inputs used by both bias variants.
- prepare systems define residue `templates`, per-system `mdp` files, and
  optional bias files.
- `bff reference` consumes the staged `01-prepare/.../reference/system-XXX/`
  directories, creates runnable outputs in
  `02-reference-data/reference-assets/system-XXX/`, and runs CP2K snapshot plus
  optional single-atom calculations locally or through Slurm.
- prepared training assets are written into one directory per system under
  `training/system-XXX/`.
- trainset and validate configs point to those prepared asset directories with
  `systems[].assets` and system-specific `n_steps`.
- the QoI workflow uses `trainset`, `refset`, `run`, and `output`.
- train writes surrogate model files.
- learn reads only `specs`, `models`, and `mcmc`.
- routines always use a single `routine` key. Bare names are builtins and
  `path.py:function` strings load custom routines.
