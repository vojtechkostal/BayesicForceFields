# Architecture

BFF is a staged research-software workflow. Each stage consumes explicit files
and writes artifacts that can be inspected, archived, or reused independently.
This keeps simulation orchestration separate from trajectory analysis and
Bayesian inference.

## Runtime Flow

```mermaid
flowchart LR
    User["YAML config or Python API"] --> Entry["bff.cli / bff.__init__"]
    Entry --> Workflow["bff.workflows"]
    Workflow --> Domain["bff.domain"]
    Workflow --> Topology["bff.topology"]
    Workflow --> IO["bff.io"]
    Topology --> External["GROMACS / CP2K / PLUMED / Slurm"]
    IO --> External
    External --> Trajectories["Topologies, trajectories, energies"]
    Trajectories --> QoI["bff.qoi"]
    Domain --> QoI
    QoI --> Dataset["QoIDataset (.pt)"]
    Dataset --> Bayes["bff.bayes"]
    Bayes <--> MCMC["bff.mcmc"]
    Bayes --> Results["PosteriorResults and plots"]
```

The CLI and Python API are thin entry points. Workflow modules own the
application-level sequence, while lower layers own reusable domain logic.

## Repository Layout

The abbreviated tree below is a navigation aid rather than an exhaustive file
listing:

```text
BayesicForceFields/
|-- bff/
|   |-- bayes/          # Gaussian processes, likelihoods, and learning
|   |-- domain/         # Parameter, campaign, and constraint models
|   |-- io/             # GROMACS, CP2K, PLUMED, YAML, and scheduler I/O
|   |-- mcmc/           # Torch-native posterior sampling
|   |-- qoi/            # Trajectory analysis and QoI datasets
|   |-- workflows/      # User-facing build-to-validation stages
|   |-- cli.py          # Command-line interface
|   |-- plotting.py     # Visualization helpers
|   `-- topology.py     # Force-field topology modification
|-- docs/               # MkDocs documentation
|-- examples/           # MD and external-data tutorials
|-- tests/              # Unit and integration tests
|-- mkdocs.yml
`-- pyproject.toml
```

## Package Map

| Module | Responsibility |
| --- | --- |
| `bff.__init__` | Public Python API: workflow functions, `Project`, `QoI`, `QoIDataset`, and `PosteriorResults`. |
| `bff.cli` | Typer command-line entry points and shell-completion setup. |
| `bff.workflows` | One package per user-facing stage. Each stage loads configuration, coordinates lower-level modules, and writes explicit artifacts. |
| `bff.workflows._shared` | Shared simulation-campaign staging, configuration parsing, preparation helpers, and scheduler integration. |
| `bff.domain` | Stable data models for parameter specifications, charge constraints, sampling campaigns, trajectories, and simulation biases. |
| `bff.topology` | GROMACS topology handling, system construction, MDAnalysis selections, force-field parameter updates, and charge reconstruction support. |
| `bff.io` | File-format and process boundaries: CP2K, EXTXYZ, MDP, PLUMED, Colvars, logging, schedulers, and YAML/PT helpers. |
| `bff.qoi` | Trajectory analysis, built-in RDF and hydrogen-bond routines, custom routine loading, and serialized `QoIDataset` objects. |
| `bff.bayes` | Local Gaussian-process surrogates, kernels, means, likelihoods, priors, posterior learning, and result handling. |
| `bff.mcmc` | Torch-native Metropolis-Hastings sampling, adaptive proposals, checkpoints, restart support, and convergence diagnostics. |
| `bff.plotting` | Posterior and surrogate visualization. |
| `bff.tools` | Small shared numerical helpers. |

## Workflow Stages

| Command | Main Input | Responsibility | Main Output |
| --- | --- | --- | --- |
| `bff build` | GROMACS topologies, coordinate templates, and MDP files | Build, equilibrate, seed, and remove virtual sites for reference use. | Self-contained `systems/<system_id>/` directories with a stable `reference/` pair |
| `bff label-snapshots` | Trajectory, topology, and user CP2K inputs | Extract frames, run CP2K labels, and collect MLIP datasets. | `train.extxyz`, `test.extxyz`, and optional isolated-atom energies |
| `bff sample-parameters` | FFMD assets, parameter bounds, and charge constraints | Draw parameter vectors and run sampled GROMACS campaigns. | `specs.yaml`, `samples.yaml`, and sampled trajectories |
| `bff build-qoi-datasets` | Sampled and reference trajectories | Compute matching quantities of interest. | One serialized `QoIDataset` per quantity of interest |
| `bff fit-lgp` | QoI datasets | Train fingerprinted local Gaussian-process surrogate committees. | `fit-lgp.log` and `models/<routine>.lgp` |
| `bff learn` | Surrogate models and `specs.yaml` | Assign effective observations and run validated posterior learning. | Fixed `outputs/` artifacts and mandatory `plots/` |
| `bff validate` | A learned posterior or explicit parameter samples, plus build systems | Draw or load parameters and rerun them as an independent campaign. | Campaign-local specs, realized samples, trajectories, and energies |

## Core Artifacts

| Artifact | Meaning |
| --- | --- |
| Build system directory | Fixed-name GROMACS files, metadata-only `system.yaml`, and virtual-site-free `reference/` topology and coordinates. |
| Label system directory | CP2K run directories, train/test EXTXYZ datasets, and optional isolated-atom energies. |
| `specs.yaml` | Named parameter bounds and reconstructable hierarchical charge constraints. |
| `samples.yaml` | Explicit sampled force-field parameter vectors and trajectory records. |
| `samples/<sample_id>/` | Per-sample job config, submission script, scheduler output, and one scientific-output directory per system. |
| `outputs/<sample_id>/` | Live job logs and auxiliary runtime files; deleted after collection when cleanup is enabled. |
| `qoi/<name>.pt` | Training-ready `QoIDataset` with ID-paired outputs and reference targets. |
| `<name>.lgp` | Trained local Gaussian-process committee for one quantity of interest. |
| `outputs/specs.yaml` | Unchanged, portable copy of the learned parameter specification. |
| `outputs/posterior.pt` | Learned posterior chain and compatibility metadata. |
| `outputs/mcmc.ckpt` | Restartable MCMC state and compatibility fingerprints. |
| `qoi-marginals.pdf` | Posterior parameter marginals colored by local QoI responsibility. |

## Design Choices

- **Stage directories are interfaces.** Fixed layouts and local metadata make
  workflow handoffs visible without duplicating deterministic paths. Expensive
  stages remain independently runnable and archivable.
- **Files are inspectable.** Workflow boundaries use inspectable artifacts so
  expensive simulation stages can be resumed, archived, or replaced.
- **IDs are identities.** System matching is keyed by explicit file-safe IDs;
  display names and YAML order never determine pairing.
- **Domain models are separate from orchestration.** Constraint reconstruction,
  trajectory records, and QoI datasets remain usable from notebooks and Python
  code without invoking the CLI.
- **External tools stay at the edges.** GROMACS, CP2K, PLUMED, and Slurm
  integration lives in topology, I/O, and workflow modules.
- **Inference is reusable.** `QoIDataset`, `bff.bayes`, and `bff.mcmc` can learn
  from user-provided data without running molecular dynamics inside BFF.

## Where To Start

- To run BFF, start with the [examples](examples/index.md).
- To configure a stage, use the [configuration reference](configuration/build.md).
- To extend analysis, inspect `bff/qoi/routines.py`.
- To contribute code, read the [development guide](development.md).
