# Bayesic Force Fields

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://vojtechkostal.github.io/BayesicForceFields/)
[![Paper](https://img.shields.io/badge/paper-JCTC%202026-blue)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)
[![Preprint](https://img.shields.io/badge/preprint-arXiv%202511.05398-b31b1b)](https://arxiv.org/abs/2511.05398)
[![Release](https://img.shields.io/github/v/tag/vojtechkostal/BayesicForceFields?label=release)](https://github.com/vojtechkostal/BayesicForceFields/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

Bayesic Force Fields (BFF) is a workflow-oriented Python package for learning
fixed-charge molecular force-field parameters from molecular dynamics
observables. It combines system building, reference-data handling, sampled MD
campaigns, QoI analysis, surrogate fitting, and posterior learning in one
toolchain.

The public CLI is centered around seven workflows:

- `bff build`
- `bff reference`
- `bff sample`
- `bff analyze`
- `bff fit`
- `bff learn`
- `bff validate`

Examples can be fetched on demand with:

```bash
bff examples
```

Documentation lives under [docs/](docs/) and is intended to be published with
MkDocs on GitHub Pages.

Published documentation:
[vojtechkostal.github.io/BayesicForceFields](https://vojtechkostal.github.io/BayesicForceFields/)

## How to Cite

If you use BFF, please cite:

```text
Kostal, V.; Shanks, B. L.; Jungwirth, P.; Martinez-Seara, H.
Bayesian Learning for Accurate and Robust Biomolecular Force Fields.
J. Chem. Theory Comput. 2026, 22 (5), 2652-2663.
https://doi.org/10.1021/acs.jctc.5c02051
```

Paper:
[Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)

Preprint:
[arXiv:2511.05398](https://arxiv.org/abs/2511.05398)

## Installation

Recommended user installation:

```bash
mamba create -n bfflearn python=3.10 pip
mamba activate bfflearn
```

Install a matching PyTorch build for your machine before fitting or learning.
The recommended way is to use the selector on the official PyTorch install
page:
https://pytorch.org/get-started/locally/

Example for Linux with CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install BFF from PyPI:

```bash
pip install bfflearn
```

If you want the exact code used for the paper, do not install `v0.0.1`
directly through a `pip` Git URL. That archived tag predates the packaging
cleanup. Instead, clone the repository, check out the archived tag, and follow
the `README.md` and `environment.yaml` shipped with that snapshot:

```bash
git clone https://github.com/vojtechkostal/BayesicForceFields.git
cd BayesicForceFields
git checkout v0.0.1
```

Use `v0.0.1` for exact reproduction of the published paper data. The current
`bfflearn` release line is the post-paper refactored workflow.

External tools are still required for full workflows:

- [Gromacs](https://www.gromacs.org) for `build`, `sample`, and `validate`
- [CP2K](https://www.cp2k.org) for `reference` and staged ab initio inputs
- [PLUMED](https://www.plumed.org) only for PLUMED-biased systems
- [PyTorch](https://pytorch.org) installed separately for `fit`, `learn`, and posterior notebooks

PyTorch is not installed by default because the appropriate CPU or CUDA build
depends on the target machine. Install the matching PyTorch build first, then
install BFF.

For development work on the repository itself, use:

```bash
mamba env create -f environment.yaml
mamba activate bfflearn
```

If you prefer to start from an existing environment instead:

```bash
pip install -e ".[dev,docs,notebook]"
```

## Quick Start

The acetate example in [examples/acetate/](examples/acetate/) keeps runnable
YAML files under `configs/` and writes generated results into numbered stage
directories:

```bash
cd examples/acetate
bff build configs/build-colvars.yaml
bff reference configs/reference-run-local.yaml
bff reference configs/reference-import.yaml
bff sample configs/sample-local.yaml
bff analyze configs/analyze.yaml
bff fit configs/fit.yaml
bff learn configs/learn.yaml
```

Validation is configured separately:

```bash
bff validate configs/validate.yaml
```

Two notebooks are included in the example:

- [notebooks/interactive.ipynb](examples/acetate/notebooks/interactive.ipynb)
  shows interactive surrogate fitting, posterior sampling, and posterior
  sample export.
- [notebooks/visualize.ipynb](examples/acetate/notebooks/visualize.ipynb)
  focuses on plotting and inspection only.

If you installed BFF from PyPI and want the example tree locally, run:

```bash
bff examples
cd examples/acetate
```

The reference workflow itself writes the final `train.extxyz` and
`valid.extxyz` files, so the normal path is simply:

```bash
bff reference CONFIG.yaml
```

## Repository Layout

- [bff/](bff/) contains the package code.
- [examples/acetate/](examples/acetate/) contains the worked example.
- [data/](data/) contains repository example inputs.
- [docs/](docs/) contains the documentation source.

## Documentation Locally

Preview the docs locally with MkDocs:

```bash
mkdocs serve
```

Build the static site with:

```bash
mkdocs build --strict
```

Shortcuts are also available:

```bash
make docs
make docs-build
```

## Shell Completion

When `bff` runs inside an activated conda environment, it writes a small
completion hook for bash and zsh into that environment. After the first `bff`
run, reactivate the environment once:

```bash
conda deactivate
conda activate bfflearn
```

After that, `bff <TAB>` should offer the public workflow commands.

## Development and Release

Packaging, docs, and deployment configuration live in:

- [pyproject.toml](pyproject.toml)
- [environment.yaml](environment.yaml)
- [.github/workflows/](.github/workflows/)

The release and publication strategy is documented in
[docs/development.md](docs/development.md).

## License

BFF is distributed under the GNU GPL v3. See [LICENSE](LICENSE).
