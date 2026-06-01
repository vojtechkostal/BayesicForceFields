# Arbitrary Data Example

This notebook-first example shows how to train and use BFF surrogate models
from user-provided tabular data. It does not run molecular dynamics or require
GROMACS.

The example calibrates two water-like Lennard-Jones oxygen parameters against
three observables:

- liquid density
- enthalpy of vaporization
- self-diffusion coefficient

The values are realistic synthetic data, intended to stand in for simulation
results and experimental targets produced outside BFF.
The fitting and learning cells use CUDA by default.

## Run

Start Jupyter from this directory and run `arbitrary-data.ipynb` from top to
bottom:

```bash
cd examples/arbitrary-data
jupyter lab
```

The notebook:

1. loads `raw-data/simulation-results.dat` and
   `raw-data/experimental-targets.dat`;
2. converts each observable into a BFF `QoIDataset`;
3. fits one local Gaussian-process surrogate per observable;
4. learns the posterior distribution of `epsilon O` and `sigma O`;
5. writes the posterior summary and plots.

Generated datasets, models, posterior files, logs, and plots are written under
`generated/` and ignored by git.

## Adapt It

Replace the two `.dat` files with your own data, update the `observable_columns`
mapping in the notebook, and edit the inline `Specs` dictionary to describe your
parameter bounds.

The rows in the simulation table are arbitrary training samples. Each output
column can be a scalar property or can be replaced with a vector-valued
observable when constructing `QoIDataset`.
