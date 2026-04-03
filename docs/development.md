# Development and Release

## Source Tree

- package code: `bff/`
- documentation: `docs/`
- worked example: `examples/acetate/`
- GitHub Actions: `.github/workflows/`

## Repository Environment

From the repository root:

```bash
mamba env create -f environment.yaml
mamba activate bbflearn
```

That shared environment installs the package in editable mode together with the
developer, notebook, and docs extras. Install PyTorch separately afterward if
you want to run `bff train`, `bff learn`, or the posterior notebooks.

## Local Quality Checks

Suggested checks from the repository root:

```bash
python -m compileall bff
python -m py_compile $(find bff -name '*.py')
python -m build
mkdocs build --strict
```

If `ruff` is installed in your environment, also run:

```bash
ruff check bff
```

## GitHub Actions

Recommended repository workflows:

- `docs.yml`
  builds and deploys the MkDocs site to GitHub Pages.
- `checks.yml`
  runs fast quality checks such as Python compilation, Ruff, package build, and
  docs build on pushes and pull requests.
- `publish.yml`
  builds and publishes wheels and sdists to PyPI through trusted publishing.

## Trusted PyPI Publishing

The clean GitHub-to-PyPI setup is:

1. Create a PyPI project for `bbflearn`.
2. In PyPI, add a trusted publisher for this repository and the
   `.github/workflows/publish.yml` workflow.
3. In GitHub, keep the publishing workflow protected by tags or release
   publication instead of every push to `main`.

The publishing workflow should:

- build the sdist and wheel with `python -m build`
- upload them as workflow artifacts
- publish them with `pypa/gh-action-pypi-publish`
- use `permissions: id-token: write`

## Versioning Strategy

Associated publication:
[Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)

For the paper snapshot you want to archive as `0.0.1`, the clean approach is:

1. Identify the exact commit that produced the paper data.
2. Set the package version on that commit to `0.0.1` if it is not already.
3. Create an annotated tag such as `v0.0.1` on that exact commit.
4. Create a GitHub release from the tag and include the paper DOI and URL in
   the release notes.
5. Switch the repository default branch to `develop`.
6. Protect `main` against direct pushes so it becomes the frozen paper branch.

That keeps `main` aligned with the publication snapshot while active
development continues on `develop`.

## Notes On Main Branch Releases

If you want `main` itself to be the archived paper branch, freeze it after the
paper tag and move day-to-day development to `develop`. The important GitHub
steps are:

- set `develop` as the default branch
- protect `main`
- create a release from the paper tag
- keep the paper DOI in the release notes and repository citation metadata
