# Bayesic Force Fields

Bayesic Force Fields (BFF) is a workflow-oriented toolkit for learning
fixed-charge molecular force fields from trajectory-derived observables.

Associated publication:
[Bayesian Learning for Accurate and Robust Biomolecular Force Fields](https://pubs.acs.org/doi/10.1021/acs.jctc.5c02051)

Preprint:
[arXiv:2511.05398](https://arxiv.org/abs/2511.05398)

For exact reproduction of the published paper data, use the archived Git tag
`v0.0.1`. The current `bfflearn` release line documents and ships the
post-paper refactored workflow.

The public interface is intentionally small:

- `bff build` stages reusable FFMD and reference assets.
- `bff reference` runs or imports canonical reference data.
- `bff sample` generates sampled FFMD campaigns from prepared assets.
- `bff analyze` computes quantities of interest from sampled and reference trajectories.
- `bff fit` fits surrogate models from analyzed QoI datasets.
- `bff learn` performs posterior learning from trained surrogate models.
- `bff validate` reruns selected posterior samples with the same campaign machinery used for sampling.

## Design Goals

- readable YAML configs with one clear job per file
- minimal hidden behavior between workflow stages
- reusable prepared assets across multiple downstream runs
- custom QoI routines that are easy to write
- packaging and release metadata clean enough for public deployment

## Where To Start

- [Installation](installation.md)
- [CLI](cli.md)
- [Configuration reference](configuration/build.md)
- [Acetate example](examples/acetate.md)
- [Development and release](development.md)
