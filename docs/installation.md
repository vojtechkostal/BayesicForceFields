# Installation

## Requirements

- Python 3.10 or newer
- GROMACS available as `gmx` for `prepare`, `simulate`, and `validate`
- CP2K only if you want to run the staged reference inputs
- PLUMED only for PLUMED-biased systems

## Recommended User Install

Create a small conda environment first:

```bash
mamba create -n bbflearn python=3.10 pip
mamba activate bbflearn
```

Install a matching PyTorch build for your machine before training or learning.
Use the official PyTorch selector for the exact command:

https://pytorch.org/get-started/locally/

Example for Linux with CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then install BFF from PyPI:

```bash
pip install bbflearn
```

PyTorch is intentionally not part of the default package dependencies because
the correct CPU or CUDA build depends on the target hardware and driver stack.

## Repository Environment

For work on the repository itself, create the shared project environment from
the repository root:

```bash
mamba env create -f environment.yaml
mamba activate bbflearn
```

That environment installs BFF in editable mode together with the `dev`, `docs`,
and `notebook` extras, but still leaves PyTorch to you so you can choose the
correct CPU or CUDA build.

If you prefer to start from an existing environment:

```bash
pip install -e ".[dev,docs,notebook]"
```

## Direct Pip Installation

If you already have a Python environment and a working PyTorch install:

```bash
pip install bbflearn
```

## Local Docs

Preview the docs locally:

```bash
mkdocs serve
```

Build the static site:

```bash
mkdocs build --strict
```

## Packaging Metadata

Package metadata lives in `pyproject.toml`. That file defines:

- runtime dependencies
- optional `dev`, `docs`, and `notebook` extras
- the `bff` console script
- Ruff configuration
- but not PyTorch, which is installed separately to let you choose the
  appropriate CPU or CUDA build
