# Bayesic Force Fields

Bayesic Force Fields (BFF) is a workflow-oriented toolkit for learning
fixed-charge molecular force fields from trajectory-derived observables.

Associated publication:
[Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)

The public interface is intentionally small:

- `bff prepare` stages equilibrated systems and reusable training assets.
- `bff simulate` generates a sampled trainset from prepared assets.
- `bff qoi` computes quantities of interest from trainset and reference data.
- `bff train` fits surrogate models from analyzed QoI datasets.
- `bff learn` performs posterior inference from trained surrogate models.
- `bff validate` reruns selected posterior samples with the same campaign
  machinery used for training.

## Design Goals

- readable YAML configs with one clear job per file
- minimal hidden behavior between workflow stages
- reusable prepared training assets
- custom QoI routines that are easy to write
- packaging and release metadata clean enough for public deployment

## Where To Start

- [Installation](installation.md)
- [CLI](cli.md)
- [Configuration reference](configuration/prepare.md)
- [Acetate example](examples/acetate.md)
- [Development and release](development.md)
