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
The fitting and learning cells use CUDA by default.

## Run

Start Jupyter from this directory and run `neon-mie-inference.ipynb` from top
to bottom:

```bash
cd examples/neon-mie-lgpmd
jupyter lab
```

The notebook:

1. loads the LGPMD simulation and experimental RDF files;
2. interpolates them onto the 73-bin grid used by the upstream tutorial;
3. retains rows inside the declared physical inference domain;
4. converts the data into a BFF `QoIDataset`;
5. fits and validates a BFF local Gaussian-process surrogate with a
   notebook-local Mie PMF mean;
6. infers the Mie `epsilon`, `lambda`, and `sigma` parameters together with
   the RDF discrepancy;
7. reports physical-unit posterior samples and plots the result.

Generated datasets, models, posterior files, logs, and plots are written under
`generated/` and ignored by git.
