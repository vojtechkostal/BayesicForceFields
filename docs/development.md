# Development

## Get The Code

```bash
git clone https://github.com/vojtechkostal/BayesicForceFields.git
cd BayesicForceFields
mamba env create -f environment.yaml
mamba activate bfflearn
```

The environment installs BFF in editable mode with the developer, docs, and
notebook extras. Install the PyTorch build that matches your machine before
running `bff fit`, `bff learn`, or posterior notebooks.

## Start A Feature Branch

Keep `main` clean and do work on a branch:

```bash
git switch main
git pull
git switch -c feature/my-change
```

Use short branch names that describe the change, for example:

```text
feature/new-qoi
fix/slurm-submit
docs/acetate-example
```

## Make The Change

Before committing, run the checks that match your change:

```bash
python -m py_compile $(find bff -name '*.py')
mkdocs build --strict
```

If tests are available in your environment:

```bash
python -m pytest -q tests
```

For docs-only changes, `mkdocs build --strict` is usually enough.

## Commit And Push

Review what changed:

```bash
git status
git diff
```

Commit focused changes:

```bash
git add PATHS_YOU_CHANGED
git commit -m "Short imperative summary"
```

Push the branch:

```bash
git push -u origin feature/my-change
```

Open a pull request from your branch into `main`.

## Updating A Branch

If `main` changed while you were working:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

Use `--force-with-lease`, not plain `--force`, so you do not overwrite someone
else's pushed work by accident.

## Repository Layout

- `bff/`: package code
- `docs/`: documentation site
- `examples/acetate/`: worked example
- `tests/`: tests
- `.github/workflows/`: CI and publishing workflows
