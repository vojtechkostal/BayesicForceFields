# Installation

## Requirements

- Python 3.9 or newer
- GROMACS available as `gmx` for `prepare`, `simulate`, and `validate`
- CP2K only if you want to run the staged reference inputs
- PLUMED only for PLUMED-biased systems

## Conda Environment

The repository ships an environment file for development, notebooks, docs, and
packaging:

```bash
conda env create -f environment.yaml
conda activate bff
```

Install a matching PyTorch build for your machine before training or learning.
Then install BFF itself:

```bash
pip install -e . --no-deps
```

The environment includes the core runtime dependencies and common developer
tools such as `pytest`, `ruff`, `mkdocs`, and Jupyter.

PyTorch is intentionally not part of the default environment file because the
correct CPU or CUDA build depends on the target hardware and driver stack.

## Pip Installation

Editable installation with extras:

```bash
pip install -e ".[dev,docs,notebook]"
```

Install PyTorch separately before using `bff train`, `bff learn`, or the
posterior notebooks.

Core runtime installation:

```bash
pip install .
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
