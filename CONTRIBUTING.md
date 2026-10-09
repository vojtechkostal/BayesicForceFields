# Contributing

Thank you for improving Bayesic Force Fields.

```bash
mamba env create -f environment.yaml
mamba activate bfflearn
pip install torch          # the build for your machine
make check                 # compileall, ruff, pytest, mkdocs build --strict
```

Keep pull requests focused, describe the behavior change, and add a
regression test for every bug fix. The
[development guide](https://vojtechkostal.github.io/BayesicForceFields/development/)
covers the code layout, design rules, and releases.
