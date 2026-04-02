# Bayesic Force Fields

Bayesic Force Fields (BFF) is a workflow-oriented Python package for learning
fixed-charge molecular force-field parameters from molecular dynamics
observables. It combines system preparation, sampled MD campaigns, QoI
analysis, surrogate training, posterior inference, and validation in one
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

Associated publication:
[Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)

## Installation

Create the development environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate bff
pip install -e . --no-deps
```

If you prefer `pip`, the package also exposes optional extras:

```bash
pip install -e ".[dev,docs,notebook]"
```

External tools are still required for full workflows:

- `gmx` for `prepare`, `simulate`, and `validate`
- CP2K for staged reference calculations
- PLUMED only for PLUMED-biased systems
- PyTorch installed separately for `train`, `learn`, and posterior notebooks

PyTorch is not installed by default because the appropriate CPU or CUDA build
depends on the target machine. Install the matching PyTorch build first, then
install BFF.

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
conda activate bff
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
