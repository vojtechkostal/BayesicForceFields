# Development and Release

## Source Tree

- package code: `bff/`
- documentation: `docs/`
- worked example: `examples/acetate/`
- GitHub Actions: `.github/workflows/`

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

1. Create a PyPI project for `BayesicForceFields`.
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

For the paper snapshot you want to archive as `0.0.1`, the clean approach is:

1. Identify the exact commit that produced the paper data.
2. Set the package version on that commit to `0.0.1` if it is not already.
3. Create an annotated tag such as `v0.0.1` on that exact commit.
4. Create a GitHub release from the tag.
5. Optionally create a long-lived branch such as `paper/v0.0.1` if you want a
   stable branch name in addition to the tag.

After that, bump `main` immediately to the next development version, for
example `0.1.0.dev0`, and continue development there.

That keeps the paper version immutable and citable while allowing active
development to move forward cleanly.

## Notes On Main Branch Releases

Do not keep `main` pretending to be the archived paper release once active
development continues. Treat the paper release as a tagged historical snapshot,
then bump `main` to a development version right away.
