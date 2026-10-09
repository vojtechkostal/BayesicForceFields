# Acetate

Learn the partial charges of acetate from two systems:

- `acetate`: aqueous acetate; QoI: acetate–water RDFs.
- `calcium-acetate`: aqueous calcium acetate with the Ca–C2 distance biased by
  Colvars metadynamics; QoI: the Ca–C2 PMF, read by the custom routine
  `inputs/pmf.py`.

```text
inputs/                  topologies, force field, MDP files, colvars.dat, pmf.py
01-build/                bff build
02-reference-md/         reference data (made outside BFF): acetate.xtc, calcium-acetate.pmf
03-sample-parameters/    bff sample-parameters
04-build-qoi-datasets/   bff build-qoi-datasets
05-fit-lgp/              bff fit-lgp
06-learn/                bff learn, posterior.ipynb
07-validate/             bff validate
```

Needs GROMACS with Colvars (2024 or newer) and PyTorch. Edit the settings
marked `ADAPT`, then:

```bash
(cd 01-build && bff build config.yaml)
(cd 03-sample-parameters && bff sample-parameters config.yaml)
(cd 04-build-qoi-datasets && bff build-qoi-datasets config.yaml)
(cd 05-fit-lgp && bff fit-lgp config.yaml)
(cd 06-learn && bff learn config.yaml)
(cd 07-validate && bff validate config.yaml)
```

Sampling runs locally; for Slurm, set `job_scheduler: slurm` and uncomment the
block at the end of `03-sample-parameters/config.yaml`.

The walkthrough explains every step, the PMF routine, and the reference data:
<https://vojtechkostal.github.io/BayesicForceFields/examples/acetate/>
