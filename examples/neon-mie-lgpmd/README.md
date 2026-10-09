# Neon Mie Inference Example

This notebook-first example uses liquid-neon radial distribution function
(RDF) data from the
[LGPMD `tutorial_v2.0`](https://github.com/hoepfnergroup/LGPMD/tree/main/tutorial_v2.0)
to infer the parameters of a lambda-6 Mie potential with BFF.

The upstream tutorial accompanies:

> Brennon L. Shanks, Harry Sullivan, Benjamin Shazed, and Michael P. Hoepfner,
> "Accelerated Bayesian Inference for Molecular Simulations using Local
> Gaussian Process Surrogate Models",
> *Journal of Chemical Theory and Computation* (2024).
> [DOI: 10.1021/acs.jctc.3c01358](https://doi.org/10.1021/acs.jctc.3c01358)

The committed files under `upstream/` were copied verbatim from LGPMD commit
`e2787cf0d830758f65f133fd1d2f7258a2ad3dee`. See `SOURCE.md` for the precise
file list and license information.

## Run

Install the optional notebook tools once:

```bash
pip install "bfflearn[notebook]"
```

Start Jupyter from this directory and run `neon-mie-inference.ipynb` from top
to bottom:

```bash
cd examples/neon-mie-lgpmd
jupyter lab
```

The notebook loads the LGPMD data, fits a local-GP surrogate whose mean is the
RDF implied by the Mie potential, checks it on LGPMD's held-out simulations,
infers `epsilon`, `lambda`, and `sigma` together with the RDF discrepancy, and
plots the posterior and the surrogate RDF at the posterior mean and the MAP.
Fitting runs on the CPU; learning uses CUDA when available. Results are written
to `generated/` (ignored by git).
