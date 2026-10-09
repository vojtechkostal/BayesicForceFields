# Bayesic Force Fields

<img src="assets/bff-logo.svg" alt="BFF logo" width="300">

Bayesic Force Fields (BFF) is a command-line workflow for learning
fixed-charge molecular force fields from trajectory observables.

Publication:
[Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)

Preprint:
[arXiv:2511.05398](https://arxiv.org/abs/2511.05398)

For exact reproduction of the published paper data, use the archived Git tag
`v0.0.1`. The current `bfflearn` package is the refactored workflow.
See the [changelog](changelog.md) for post-publication highlights.

## What BFF Does

BFF learns force-field parameters against a reference trajectory that you
simulate outside BFF:

```text
build -> [external reference MD] -> sample-parameters -> build-qoi-datasets
      -> fit-lgp -> learn -> validate
```

- `build`: equilibrate systems and run seeded production trajectories
- reference MD: your own simulation, for example with a foundation or
  fine-tuned MLIP; see [Reference trajectories](reference-trajectories.md)
- `sample-parameters`: run sampled force-field MD campaigns
- `build-qoi-datasets`: compute quantities of interest from sample and reference data
- `fit-lgp`: train fingerprinted surrogate models
- `learn`: infer posterior force-field parameters
- `validate`: rerun selected posterior samples

## Supported Learned Parameters

BFF currently learns GROMACS partial charges, Lennard-Jones sigma and epsilon,
and function-9 dihedral force constants. A single bound can tie multiple atom
names or atom types to one learned value. Charge parameters also support
hierarchical residue- or system-level constraints.

See the [sample configuration reference](configuration/sample-parameters.md#parameter-labels)
for the accepted labels, matching rules, and examples.

## Quick Start

Install BFF, copy the example tree, then run the acetate walkthrough:

```bash
mamba create -n bfflearn python=3.10 pip
mamba activate bfflearn
pip install bfflearn

bff examples
cd examples/acetate
```

!!! warning
    Install the PyTorch build that matches your machine separately before
    fitting or learning. Use the
    [official PyTorch selector](https://pytorch.org/get-started/locally/) for
    CPU or CUDA installation commands.

Each stage of the acetate example is a directory with its config. Edit the
config there and run BFF from that directory:

```bash
cd 01-build
bff build config.yaml
cd ..
```

Continue with the stages in the [acetate example](examples/acetate.md).

## Where To Go Next

- [Installation](installation.md)
- [Architecture](architecture.md)
- [Reference trajectories](reference-trajectories.md)
- [Command-line interface](cli.md)
- [Examples overview](examples/index.md)
- [Configuration reference](configuration/build.md)
- [Development](development.md)
- [Contributing](https://github.com/vojtechkostal/BayesicForceFields/blob/main/CONTRIBUTING.md)
- [Support](https://github.com/vojtechkostal/BayesicForceFields/blob/main/SUPPORT.md)
