# Development

## Setup

```bash
git clone https://github.com/vojtechkostal/BayesicForceFields.git
cd BayesicForceFields
mamba env create -f environment.yaml   # editable install with dev, docs, notebook extras
mamba activate bfflearn
pip install torch                      # the build for your machine
```

## Check a change

```bash
make check      # compileall, ruff, pytest, mkdocs build --strict
```

Run the narrowest tests while you work (`pytest tests/qoi -q`), and `make
check` before you push. Tests use temporary files and mocked GROMACS, Slurm,
and CP2K; nothing external has to be installed.

## Code layout

```text
bff/
├── cli.py                  the `bff` command; thin, calls workflows
├── __init__.py             public Python API: stage functions, Project, QoI, Results
├── workflows/
│   ├── config.py           ConfigSection: the typed YAML reader of every stage
│   ├── build/              bff build
│   ├── campaign/           MD campaigns of sample-parameters and validate,
│   │                       including the per-sample job `bff md`
│   ├── sample_parameters/
│   ├── build_qoi_datasets/
│   ├── fit_lgp/
│   ├── learn/
│   └── validate/
├── domain/                 specs, charge constraints, systems, samples.yaml, biases
├── qoi/                    routines (rdf, hydrogen_bonds, custom), analysis, QoIDataset
├── bayes/                  Gaussian-process surrogates, likelihoods, priors, results
├── mcmc/                   Metropolis-Hastings sampler, proposals, checkpoints
├── io/                     MDP, Colvars, PLUMED, YAML, logs
├── gromacs.py              grompp/mdrun with an optional bias
├── slurm.py                job arrays and queue polling
├── topology.py             force-field parameter edits on GROMACS topologies
└── plotting.py
scripts/label_structures.py standalone CP2K labeling; must not import bff
```

## Design rules

- **Stage files are interfaces.** Directory layouts, file names, and
  `system_id`/`sample_id` values are what stages and users rely on; change them
  only deliberately, with a changelog and migration note.
- **IDs pair systems**, never list order or display names.
- **Configs go through `ConfigSection`.** It reports unknown keys, wrong types,
  and missing files with the full key path. A new option needs its parser,
  a test, the option table in `docs/configuration/`, and the example YAML if
  it is user-facing; `tests/test_docs_config.py` fails when an accepted key is
  missing from its table.
- **External tools stay at the edges**: GROMACS in `gromacs.py`, Slurm in
  `slurm.py`, file formats in `io/`. The reference MD is outside BFF.
- **Plain code first.** Prefer linear functions over small wrappers, and raise
  errors that name the stage, system or sample, and the path or key.

## Branches and pull requests

Work on a branch (`fix/slurm-time`, `docs/acetate`), keep pull requests
focused, describe the behavior change, and add a regression test for every
bug fix. After `main` moves on: `git rebase origin/main` and `git push
--force-with-lease`.

## Release

1. Set the version in `pyproject.toml`, `bff/__init__.py`, and `CITATION.cff`;
   update `CHANGELOG.md` and `docs/changelog.md`.
2. Run `make check`, `python -m build`, and `python -m twine check dist/*`,
   and execute the notebooks in a temporary copy.
3. Check that new files are tracked (`git status`).
4. Merge into `main`, tag `vX.Y.Z`, and publish the GitHub release; the release
   workflow publishes to PyPI.
5. Install the wheel and run `bff examples`, which fetches the examples of
   the tag.
6. Archive the release on Zenodo and record its DOI in `CITATION.cff`.

AI coding agents follow [`AGENTS.md`](https://github.com/vojtechkostal/BayesicForceFields/blob/main/AGENTS.md).
