# Guidance for AI Coding Agents

Repository context for making safe, focused changes. The user's instructions
and the documentation come first; this file does not widen a task.

## Before editing

1. Read `README.md`, `docs/workflow.md`, `docs/development.md`, and the page
   of the stage you change (`docs/configuration/<stage>.md`).
2. Run `git status --short`. The worktree may hold the user's work, including
   untracked run outputs inside `examples/`: keep it, never discard it for a
   clean tree.
3. Find the nearby tests and follow existing names, file layouts, and config
   conventions.
4. Make the smallest cohesive change; update tests and docs with it.

## Map

| Path | Content |
| --- | --- |
| `bff/cli.py` | the `bff` command; thin, calls `bff/workflows/<stage>/main.py` |
| `bff/__init__.py` | public Python API and `__version__` |
| `bff/workflows/config.py` | `ConfigSection`: read every option through it |
| `bff/workflows/<stage>/` | one package per stage: `config.py` (parser) and `main.py` |
| `bff/workflows/campaign/` | MD campaigns of `sample-parameters` and `validate`; `job.py` is the per-sample `bff md` job |
| `bff/gromacs.py`, `bff/slurm.py` | GROMACS runs, Slurm job arrays |
| `bff/qoi/` | QoI routines (built-in and custom share one interface), analysis, `QoIDataset` |
| `bff/bayes/`, `bff/mcmc/` | surrogates, likelihoods, posterior sampling |
| `bff/domain/`, `bff/io/` | data models (`specs.yaml`, `samples.yaml`), file formats |
| `scripts/label_structures.py` | standalone CP2K labeling; not installed by pip; must not import `bff` |
| `examples/` | acetate MD workflow and two notebooks |
| `docs/` | MkDocs site; `docs/configuration/` is the YAML contract |

## Pipeline

```text
build -> [reference MD, outside BFF] -> sample-parameters -> build-qoi-datasets
      -> fit-lgp -> learn -> validate
```

Use these stage names everywhere and add no aliases for retired names. `bff md`
is internal. BFF never runs the reference MD; do not add substitute reference
trajectories or an MLIP implementation.

## Contracts

- Stage directories, file names, `system_id` and `sample_id` values, and
  `samples.yaml`/`specs.yaml` are interfaces. Pair systems by ID, never by
  order or display name.
- A new or changed option needs: its parser (through `ConfigSection`), a test,
  its row in `docs/configuration/` (`tests/test_docs_config.py` checks this),
  a link in `docs/configuration/index.md` for a new option group, the example
  YAML if users set it, and a changelog entry if it is user-visible.
- Campaign files of unfinished samples (`production.cpt`, `.tpr`, `.log`) are
  what a restart needs: never prune them. Only systems with a
  `production.done` marker are cleaned up.
- A reference trajectory has the atoms, in order, of
  `systems/<system_id>/reference/` written by `bff build`.

## External software

GROMACS, Colvars/PLUMED, Slurm, and CP2K stay at the edges (`gromacs.py`,
`slurm.py`, `io/`, the labeling script). Tests use temporary files and mocked
processes; never claim an external run succeeded when only staging was
tested. On a shared cluster, follow the site's rules for running test suites
and builds (for example inside a scheduler job).

## Style

- Linear, readable code; no wrappers that hide control flow and no
  speculative abstractions. Reuse the existing domain models.
- `pathlib.Path`; paths in configs resolve relative to the config file.
- Errors name the stage, system or sample, and the key or path.
- Python 3.10 compatible; Ruff settings in `pyproject.toml`.
- Every bug fix gets a regression test of public behavior.

## Examples and notebooks

- `examples/acetate` is deliberately simple: two systems (`acetate` with an
  RDF QoI, `calcium-acetate` with a Colvars PMF read by `inputs/pmf.py`), one
  `config.yaml` per stage. The sampling config runs locally; its Slurm block
  is commented out and `tests/test_examples.py` uncomments and loads it, so
  keep it valid.
- Example YAML must load with the CLI's parser; mark settings users must
  adapt with `ADAPT`.
- Committed notebooks are output-free. The data notebooks let BFF choose the
  device (no `cuda` or `DEVICE` in them). Execute notebooks in a temporary
  copy so generated files stay out of the repository.

## Checks

```bash
make check                          # compileall, ruff, pytest, mkdocs --strict
python -m build && python -m twine check dist/*   # packaging changes
git diff --check
```

Report exactly what was and was not validated when a check cannot run.

## Releases

Keep the version identical in `pyproject.toml`, `bff/__init__.py`, and
`CITATION.cff`; update `CHANGELOG.md` and `docs/changelog.md`. Before a release
commit, check untracked files so new packages, configs, docs, and tests are
included.
