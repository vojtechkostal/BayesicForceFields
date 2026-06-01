# Contributing

Thank you for improving Bayesic Force Fields.

## Development Setup

Create the repository environment and install the PyTorch build appropriate
for your machine:

```bash
mamba env create -f environment.yaml
mamba activate bfflearn
```

See the [development guide](https://vojtechkostal.github.io/BayesicForceFields/development/)
for the branch workflow and repository layout.

## Before Opening A Pull Request

Run the release-facing checks:

```bash
python -m compileall -q bff
ruff check .
python -m pytest -q
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Keep pull requests focused, describe observable behavior changes, and add
regression tests when fixing a bug.
