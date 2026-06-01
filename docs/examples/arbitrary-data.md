# Arbitrary Data Walkthrough

Source files:

- [examples/arbitrary-data/][arbitrary-data-root]

## Goal

This notebook-first example shows how to use BFF when simulation results and
reference targets already exist outside the built-in trajectory-analysis
workflow. It calibrates two water-like Lennard-Jones parameters against
realistic synthetic density, enthalpy-of-vaporization, and diffusion data.

No GROMACS installation is required.
The fitting and learning cells use CUDA by default.

## Run

```bash
cd examples/arbitrary-data
jupyter lab
```

Open `arbitrary-data.ipynb` and execute it from top to bottom. The notebook
demonstrates the complete data-facing workflow:

1. load user-provided whitespace-delimited tables;
2. construct and write one `QoIDataset` per observable;
3. fit local Gaussian-process surrogate models;
4. build a constrained `LearningProblem`;
5. sample the posterior and write summary plots.

The generated `qoi-*.pt` datasets can also be passed to `bff fit`. All
notebook-generated files are written under `generated/`.

## Input Tables

- [raw-data/simulation-results.dat][arbitrary-data-simulations] contains one
  externally evaluated parameter set per row.
- [raw-data/experimental-targets.dat][arbitrary-data-targets] contains the
  target value and uncertainty for each observable.
- The inline `Specs` dictionary in the notebook defines the parameter bounds
  used during posterior learning.

The committed `.dat` values are illustrative synthetic data. Replace them with
your own tables and update the notebook mappings for a new application.

[arbitrary-data-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/arbitrary-data
[arbitrary-data-simulations]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/arbitrary-data/raw-data/simulation-results.dat
[arbitrary-data-targets]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/arbitrary-data/raw-data/experimental-targets.dat
