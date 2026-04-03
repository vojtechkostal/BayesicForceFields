# Bayesic Force Fields

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://vojtechkostal.github.io/BayesicForceFields/)
[![Paper](https://img.shields.io/badge/paper-JCTC%202026-blue)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)
[![Preprint](https://img.shields.io/badge/preprint-arXiv%202511.05398-b31b1b)](https://arxiv.org/abs/2511.05398)
[![Release](https://img.shields.io/github/v/tag/vojtechkostal/BayesicForceFields?label=release)](https://github.com/vojtechkostal/BayesicForceFields/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

Bayesic Force Fields (BFF) is a workflow-oriented Python package for learning
fixed-charge molecular force-field parameters from molecular dynamics
observables. It combines system preparation, sampled MD campaigns, QoI
analysis, surrogate training and posterior inference in one
toolchain.

The public CLI is centered around six workflows:

- `bff prepare`
- `bff simulate`
- `bff qoi`
- `bff train`
- `bff learn`
- `bff validate`

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

Install a matching PyTorch build for your machine before training or learning.
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

- [Gromacs](https://www.gromacs.org) for `prepare`, `simulate`, and `validate`
- [CP2K](https://www.cp2k.org) for staged reference calculations
- [PLUMED](https://www.plumed.org) only for PLUMED-biased systems
- [PyTorch](https://pytorch.org) installed separately for `train`, `learn`, and posterior notebooks

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

The acetate example in [examples/acetate/](examples/acetate/) shows the
intended stage order:

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
```

Validation is configured separately in stage `08`:

```bash
cd ../08-validate
bff validate config.yaml
```

Two notebooks are included in the example:

- [06-learn/interactive.ipynb](examples/acetate/06-learn/interactive.ipynb)
  shows interactive surrogate training, posterior sampling, and posterior
  sample export.
- [07-visualize/visualize.ipynb](examples/acetate/07-visualize/visualize.ipynb)
  focuses on plotting and inspection only.

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
