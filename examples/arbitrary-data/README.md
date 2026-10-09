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

## Run

Install the optional notebook tools once (`pip install "bfflearn[notebook]"`),
then run `arbitrary-data.ipynb` from top to bottom in this directory. The
notebook loads the two tables in `raw-data/`, builds one `QoIDataset` per
observable, fits a local-GP surrogate for each (on the CPU), and learns the
posterior of `epsilon O` and `sigma O`; learning uses CUDA when available.
Results are written to `generated/` (ignored by git).

## Adapt It

Replace the two `.dat` files with your own data, change the inline `Specs`
bounds, and adapt the `QoIDataset` construction: the output of an observable
can be a scalar or a vector (a curve).
