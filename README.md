# Bayesic Force Fields

<p align="center">
  <img src="https://raw.githubusercontent.com/vojtechkostal/BayesicForceFields/main/docs/assets/bff-logo.svg" alt="BFF logo" width="260">
</p>

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://vojtechkostal.github.io/BayesicForceFields/)
[![Paper](https://img.shields.io/badge/paper-JCTC%202026-blue)](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)
[![Release](https://img.shields.io/github/v/tag/vojtechkostal/BayesicForceFields?label=release)](https://github.com/vojtechkostal/BayesicForceFields/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/vojtechkostal/BayesicForceFields/blob/main/LICENSE)

Bayesic Force Fields (BFF) learns fixed-charge force-field parameters for
GROMACS by Bayesian inference. It runs classical MD for many sampled
parameter sets, compares quantities of interest (QoIs) such as RDFs or free
energy profiles with a reference simulation, fits Gaussian-process surrogates
to them, and samples the posterior distribution of the parameters.

## Workflow

Each step is one command with one YAML config:

```text
  bff build                 build and equilibrate the GROMACS systems
      │
      ▼
  reference MD              outside BFF: AIMD or an MLIP (see below)
      │
      ▼
  bff sample-parameters     draw parameter sets, run classical MD for each
      │
      ▼
  bff build-qoi-datasets    compute the QoIs of every sample and of the reference
      │
      ▼
  bff fit-lgp               fit a Gaussian-process surrogate per QoI
      │
      ▼
  bff learn                 sample the posterior of the parameters (MCMC)
      │
      ▼
  bff validate              rerun MD with posterior parameters
```

**BFF does not generate the reference data.** You run the reference simulation
yourself; we recommend ab initio MD or a machine-learned interatomic potential
(MLIP), ideally a foundation model fine-tuned for your system. For fine-tuning,
the repository provides
[`scripts/label_structures.py`](https://github.com/vojtechkostal/BayesicForceFields/blob/main/scripts/label_structures.py), which labels MD
frames with CP2K on Slurm. It is not installed by `pip`; download it from the
repository. See
[Reference data](https://vojtechkostal.github.io/BayesicForceFields/reference-trajectories/).

## Install

```bash
mamba create -n bfflearn python=3.10 pip
mamba activate bfflearn
pip install torch          # pick the build for your CPU/GPU: https://pytorch.org/get-started/locally/
pip install bfflearn
```

The MD stages need GROMACS (with Colvars or PLUMED for biased systems).
Notebook examples need `pip install "bfflearn[notebook]"`.

## Quick start

```bash
bff examples           # copy the examples matching your BFF version
cd examples/acetate    # full workflow: aqueous acetate and calcium acetate
```

Every stage of the example is a directory with its config; run `bff <stage>
config.yaml` inside it. The
[acetate walkthrough](https://vojtechkostal.github.io/BayesicForceFields/examples/acetate/)
explains each step.

## Learned parameters

| Parameter | Label in `bounds` |
| --- | --- |
| Partial charge | `charge O1 O2` |
| Lennard-Jones sigma | `sigma OW` |
| Lennard-Jones epsilon | `epsilon OW` |
| Function-9 dihedral force constant | `dihedraltype9_3_180` |

Names in one label share one value. Charges can be tied by residue- or
system-level charge constraints.

## Repository

```text
BayesicForceFields/
├── bff/                     the Python package and the `bff` command
│   ├── workflows/           one package per stage
│   ├── qoi/                 QoI routines (rdf, hydrogen_bonds, custom) and datasets
│   ├── bayes/, mcmc/        surrogates, likelihoods, posterior sampling
│   └── domain/, io/         data models and file formats
├── examples/
│   ├── acetate/             full MD workflow with an RDF and a PMF
│   ├── arbitrary-data/      notebook: learn from your own tabular data
│   └── neon-mie-lgpmd/      notebook: learn from published RDFs
├── scripts/
│   └── label_structures.py  CP2K labeling for MLIP fine-tuning (not installed by pip)
├── docs/                    documentation site
└── tests/
```

## Documentation

[vojtechkostal.github.io/BayesicForceFields](https://vojtechkostal.github.io/BayesicForceFields/):
[how BFF works](https://vojtechkostal.github.io/BayesicForceFields/workflow/),
[all settings](https://vojtechkostal.github.io/BayesicForceFields/configuration/),
[examples](https://vojtechkostal.github.io/BayesicForceFields/examples/), and
[development](https://vojtechkostal.github.io/BayesicForceFields/development/).
See also the [changelog](https://github.com/vojtechkostal/BayesicForceFields/blob/main/CHANGELOG.md), [contributing](https://github.com/vojtechkostal/BayesicForceFields/blob/main/CONTRIBUTING.md),
[support](https://github.com/vojtechkostal/BayesicForceFields/blob/main/SUPPORT.md), and [security](https://github.com/vojtechkostal/BayesicForceFields/blob/main/SECURITY.md).

## Citation

> Kostal, V.; Shanks, B. L.; Jungwirth, P.; Martinez-Seara, H.
> Bayesian Learning for Accurate and Robust Biomolecular Force Fields.
> *J. Chem. Theory Comput.* **2026**, *22* (5), 2652-2663.
> [doi:10.1021/acs.jctc.5c02051](https://doi.org/10.1021/acs.jctc.5c02051)

The code used for the paper is archived as
[`v0.0.1`](https://github.com/vojtechkostal/BayesicForceFields/tree/v0.0.1).

## License

[GNU GPL v3](https://github.com/vojtechkostal/BayesicForceFields/blob/main/LICENSE)
