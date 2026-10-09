# Bayesic Force Fields

<img src="assets/bff-logo.svg" alt="BFF logo" width="260">

Bayesic Force Fields (BFF) learns fixed-charge force-field parameters for
GROMACS by Bayesian inference. It runs classical MD for many sampled
parameter sets, compares quantities of interest (QoIs) such as RDFs or free
energy profiles with a reference simulation, fits Gaussian-process surrogates
to them, and samples the posterior distribution of the parameters.

Paper: [Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051),
*J. Chem. Theory Comput.* 2026 ([arXiv:2511.05398](https://arxiv.org/abs/2511.05398)).
The code used for the paper is archived as the Git tag `v0.0.1`.

## Workflow

```text
  bff build                 build and equilibrate the GROMACS systems
      │
      ▼
  reference MD              outside BFF: AIMD or an MLIP
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

!!! note "BFF does not generate the reference data"
    You run the reference simulation yourself. We recommend ab initio MD or a
    machine-learned interatomic potential (MLIP), ideally a foundation model
    fine-tuned for your system. For fine-tuning, the repository provides a
    CP2K labeling script that is not installed by `pip`; see
    [Reference data](reference-trajectories.md).

[How BFF works](workflow.md) describes what each stage reads and writes.

## Quick start

```bash
mamba create -n bfflearn python=3.10 pip
mamba activate bfflearn
pip install torch     # the build for your machine: https://pytorch.org/get-started/locally/
pip install bfflearn

bff examples
cd examples/acetate
```

Each stage of the [acetate example](examples/acetate.md) is a directory with
its config; run `bff <stage> config.yaml` inside it.

## Learned parameters

| Parameter | Label in `bounds` |
| --- | --- |
| Partial charge | `charge O1 O2` |
| Lennard-Jones sigma | `sigma OW` |
| Lennard-Jones epsilon | `epsilon OW` |
| Function-9 dihedral force constant | `dihedraltype9_3_180` |

Names in one label share one value; charges can be tied by
[charge constraints](configuration/sample-parameters.md#charge-constraints).
See [parameter labels](configuration/sample-parameters.md#parameter-labels).

## Where to go next

- [Installation](installation.md)
- [How BFF works](workflow.md)
- [All settings](configuration/index.md), stage by stage
- [Examples](examples/index.md)
- [Development](development.md)
