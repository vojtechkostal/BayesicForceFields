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
python -m compileall -q bff
ruff check .
python -m pytest -q
mkdocs build --strict
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

The [architecture guide](architecture.md) describes the package modules,
workflow stages, and persisted artifacts in more detail.

## Publishing Research Software

For a citable BFF release:

1. Update the changelog, version, examples, and documentation together.
2. Reserve a version-specific Zenodo DOI if the release needs an archival DOI.
3. Run the test suite, lint checks, package build, and strict documentation
   build in CI.
4. Merge the release branch into `main` and create a signed `vX.Y.Z` tag.
5. Publish the GitHub release from that tag. The release workflow builds,
   verifies, and publishes the distributions through PyPI trusted publishing.
6. Archive the tagged release in Zenodo and record its version-specific DOI
   and release date in `CITATION.cff`.

The repository already includes package metadata, an OSI-approved license,
`CITATION.cff`, a changelog, documentation, examples, tests, and GitHub Actions
workflows. Contribution, support, conduct, and security policies live in the
repository root. Expand API reference documentation as the public Python API
matures.

Useful references:

- [FAIR Principles for Research Software](https://doi.org/10.15497/RDA00068)
- [GitHub citation-file documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- [Zenodo DOI documentation](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/)
- [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
- [Diátaxis documentation framework](https://diataxis.fr/)
