# Acetate Example

This example keeps committed source assets under `inputs/`, runnable YAML files
under `configs/`, and generated workflow outputs in numbered stage directories
at the example root.

## Recommended Run Order

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

The default walkthrough uses the Colvars build plus local execution. Variant
configs for PLUMED and Slurm are also included.

## Layout

```text
examples/acetate/
  README.md
  configs/
    build-colvars.yaml
    build-plumed.yaml
    reference-run-local.yaml
    reference-run-slurm.yaml
    reference-import.yaml
    sample-local.yaml
    sample-slurm.yaml
    analyze.yaml
    fit.yaml
    learn.yaml
    validate.yaml
  inputs/
    common/
    biases/
    reference-inputs/
    reference-trajectories/
    restraint.py
  notebooks/
  01-build-colvars/
  01-build-plumed/
  02-reference-run-local/
  02-reference-run-slurm/
  02-reference-import/
  03-sample-local/
  03-sample-slurm/
  04-analyze/
  05-fit/
  06-learn/
  07-validate/
```

## Config Files

- [configs/build-colvars.yaml](configs/build-colvars.yaml)
  stages the default Colvars-biased systems into [01-build-colvars/](01-build-colvars/).
- [configs/build-plumed.yaml](configs/build-plumed.yaml)
  stages the same systems with PLUMED bias files into `01-build-plumed/`.
- [configs/reference-run-local.yaml](configs/reference-run-local.yaml)
  runs staged CP2K snapshot and single-atom jobs locally into
  [02-reference-run-local/](02-reference-run-local/).
- [configs/reference-run-slurm.yaml](configs/reference-run-slurm.yaml)
  is the Slurm-backed CP2K variant and can override staged MD inputs from
  `inputs/reference-inputs/`.
- [configs/reference-import.yaml](configs/reference-import.yaml)
  canonicalizes external AIMD trajectories into `02-reference-import/` for the
  analysis stage.
- [configs/sample-local.yaml](configs/sample-local.yaml)
  runs the default sampled FFMD campaign into `03-sample-local/`.
- [configs/sample-slurm.yaml](configs/sample-slurm.yaml)
  is the Slurm-backed sampling variant.
- [configs/analyze.yaml](configs/analyze.yaml)
  compares `03-sample-local/` against `02-reference-import/` and writes QoI
  datasets under `04-analyze/`.
- [configs/fit.yaml](configs/fit.yaml)
  fits surrogate models under `05-fit/`.
- [configs/learn.yaml](configs/learn.yaml)
  runs posterior learning under `06-learn/`.
- [configs/validate.yaml](configs/validate.yaml)
  reruns selected posterior samples under `07-validate/`.

## Inputs

- [inputs/common/](inputs/common/)
  contains the shared topologies, force-field includes, template coordinates,
  and MDP files used by the build stage.
- [inputs/biases/](inputs/biases/)
  contains the Colvars and PLUMED restraint files used by the two build
  variants.
- [inputs/reference-inputs/](inputs/reference-inputs/)
  contains optional CP2K input overrides for the Slurm reference run.
- [inputs/reference-trajectories/](inputs/reference-trajectories/)
  contains the external AIMD trajectories imported by `reference-import.yaml`.
- [inputs/restraint.py](inputs/restraint.py)
  defines the custom distance-distribution QoI used for the calcium-bound
  systems.

## Notes

- The build stage writes prepared FFMD assets under `01-build-*/ffmd/` and
  staged CP2K inputs under `01-build-*/reference/`.
- `reference-run-*.yaml` produces canonical `train.extxyz` and `valid.extxyz`
  labels from staged CP2K jobs, while `reference-import.yaml` copies the AIMD
  trajectories consumed by `bff analyze`.
- Numbered stage directories are runtime outputs and are ignored by git.
- Interactive notebooks live in [notebooks/](notebooks/).
