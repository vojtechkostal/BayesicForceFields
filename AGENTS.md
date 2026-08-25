# Guidance for AI Coding Agents

This file gives coding agents the repository-specific context needed to make
safe, focused changes. User instructions and the documentation remain
authoritative; do not treat this file as permission to broaden a task.

## Start Here

1. Read `README.md`, `docs/architecture.md`, and the documentation for the
   workflow or module being changed.
2. Inspect `git status --short` and the relevant diff before editing. The
   worktree may contain intentional user changes; preserve unrelated work and
   never discard it to obtain a clean tree.
3. Find nearby tests and follow existing public names, artifact layouts, and
   configuration conventions.
4. Make the smallest cohesive change, update tests and user-facing
   documentation together, and run checks proportional to the change.

## Project Map

- `bff/cli.py`: Typer CLI and public command names.
- `bff/__init__.py`: supported public Python imports and package version.
- `bff/workflows/`: one package per user-facing workflow stage.
- `bff/workflows/_shared/`: configuration, campaign, preparation, and
  scheduler infrastructure shared by stages.
- `bff/domain/`: stable domain models and serialized workflow records.
- `bff/io/`: external process and file-format boundaries.
- `bff/qoi/`: trajectory analysis and `QoIDataset` construction.
- `bff/bayes/` and `bff/mcmc/`: surrogate fitting and posterior learning.
- `examples/`: user templates and self-contained notebook examples.
- `docs/`: MkDocs site; configuration pages describe the YAML contract.
- `tests/`: unit, integration, configuration, and example-contract tests.

## Canonical Pipeline

The public pipeline is:

```text
build -> label-snapshots -> external MLIP workflow
      -> sample-parameters -> build-qoi-datasets -> fit-lgp -> learn -> validate
```

Use these names in code, tests, examples, and documentation. Do not add aliases
for retired stage names unless the user explicitly requests compatibility.
Scheduled-job commands such as `md` and `label-snapshot-job` are internal, even
though workflow code invokes them.

Stage directories and serialized files are interfaces, not incidental output.
Preserve stable `system_id` and `sample_id` values, fixed artifact names,
explicit handoffs, and compatibility metadata. Pair systems by ID, never by
YAML order or display name. When changing a configuration model or artifact,
update its parser, tests, example YAML, configuration reference, and migration
notes when the change is user-visible.

## External Software Boundaries

BFF orchestrates GROMACS, CP2K, PLUMED or Colvars, and optionally Slurm. Keep
external commands at the established topology, I/O, scheduler, and workflow
boundaries. Tests should use temporary files and mocked process execution
unless an integration test explicitly requires installed scientific software.
Do not claim an external simulation succeeded when only staging was tested.

The acetate example is a complete template for BFF-owned stages but has an
intentional external MLIP boundary. Do not add substitute trajectories or an
MLIP implementation. Its reference trajectories must retain the documented
atom-order-matched, virtual-site-free handoff.

## Implementation Style

- Prefer linear, readable code over small wrappers that obscure control flow.
- Reuse established domain models and shared helpers when they represent the
  same concept; avoid speculative abstractions.
- Keep CLI entry points thin and put reusable behavior in the appropriate
  workflow or lower-level module.
- Use `pathlib.Path` and preserve the repository's existing path-resolution
  semantics. Never assume the caller's home directory or a cluster layout.
- Raise errors with enough context to identify the stage, system or sample,
  and offending path or configuration field.
- Preserve Python 3.10 compatibility and the Ruff configuration in
  `pyproject.toml`.
- Add a regression test for a bug fix and test public behavior rather than
  private implementation details where practical.

## Examples and Notebooks

Example YAML must load through the same configuration parser as the CLI. Keep
paths relative to the documented stage directory and clearly mark settings
that users must adapt, especially Slurm setup, executable paths, CPU/GPU
selection, atom selections, and simulation lengths.

Committed notebooks must be output-free. The arbitrary-data and Neon examples
select `cuda` only when `torch.cuda.is_available()` and otherwise use `cpu`;
keep the chosen device visible and consistent. Execute notebook changes in a
temporary copy so generated datasets, models, plots, and checkpoints do not
pollute the repository.

## Validation

Use the narrowest relevant tests while iterating, then run the applicable
repository checks before handoff:

```bash
python -m compileall -q bff
ruff check .
python -m pytest -q
mkdocs build --strict
```

For packaging or release changes, also run:

```bash
python -m build
python -m twine check dist/*
```

Run `git diff --check` after documentation or patch-heavy edits. If an external
dependency or network restriction prevents a gate, report exactly what was and
was not validated rather than weakening or silently skipping the check.

## Release-Coordinated Files

Keep the version synchronized in `pyproject.toml`, `bff/__init__.py`, and
`CITATION.cff`. A release normally also updates `CHANGELOG.md`,
`docs/changelog.md`, examples, and any migration documentation. Before a
release commit, inspect untracked files so new workflow packages, configs,
inputs, documentation, and tests are not omitted.

