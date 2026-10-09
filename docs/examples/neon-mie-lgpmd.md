# Neon Mie Inference

Source files:

- [examples/neon-mie-lgpmd/][neon-root]
- [Upstream LGPMD tutorial][lgpmd-tutorial]
- [Associated article][lgpmd-paper]

## Goal

This notebook-first example uses real liquid-neon radial distribution function
(RDF) data from the LGPMD `tutorial_v2.0` directory. It infers the `epsilon`,
`lambda`, and `sigma` parameters of a lambda-6 Mie potential from the
experimental RDF.

No molecular dynamics run is required. The committed upstream files include
the simulation training set, held-out validation set, and experimental RDF.
Fitting runs on the CPU; learning uses CUDA when available.

## Run

Install the optional notebook tools once:

```bash
pip install "bfflearn[notebook]"
```

```bash
cd examples/neon-mie-lgpmd
jupyter lab
```

Open `neon-mie-inference.ipynb` and execute it from top to bottom. The notebook
demonstrates the complete data-facing workflow:

1. load and interpolate the upstream RDF data and keep the rows inside the
   physical inference domain;
2. build a BFF `QoIDataset`;
3. fit a local Gaussian-process surrogate with a notebook-local Mie PMF mean
   and check it on LGPMD's held-out simulations;
4. infer the Mie parameters and the RDF discrepancy;
5. plot the posterior, the QoI-attributed marginals, and the surrogate RDF at
   the posterior mean and the MAP.

The RDF discrepancy is learned because the source data do not provide an
experimental uncertainty. The callable PMF mean stays inside the notebook
because it is specific to this Mie-potential example. The effective number of
independent RDF observations is inferred from the RDF's correlation length.

The copied-file inventory, source commit, article citation, and upstream
license are recorded in [SOURCE.md][neon-source]. Generated files are written
under `generated/`.

[neon-root]: https://github.com/vojtechkostal/BayesicForceFields/tree/main/examples/neon-mie-lgpmd
[neon-source]: https://github.com/vojtechkostal/BayesicForceFields/blob/main/examples/neon-mie-lgpmd/SOURCE.md
[lgpmd-tutorial]: https://github.com/hoepfnergroup/LGPMD/tree/main/tutorial_v2.0
[lgpmd-paper]: https://doi.org/10.1021/acs.jctc.3c01358
