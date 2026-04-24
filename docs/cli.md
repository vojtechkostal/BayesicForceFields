# Command-Line Interface

The CLI entry point is implemented in `bff/cli.py`.

## Public Commands

- `bff build CONFIG.yaml`
  Build reusable FFMD and reference starting assets.
- `bff reference CONFIG.yaml`
  Run or import canonical reference datasets.
- `bff sample CONFIG.yaml`
  Sample force-field parameters and run FFMD campaigns.
- `bff analyze CONFIG.yaml`
  Analyze sampled and reference trajectories into matched QoI datasets.
- `bff fit CONFIG.yaml`
  Fit surrogate models from analyzed QoI datasets.
- `bff learn CONFIG.yaml`
  Run Bayesian posterior learning over force-field parameters.
- `bff validate CONFIG.yaml`
  Rerun selected parameter samples for validation.
- `bff examples`
  Copy or download the repository example tree.
- `bff version`
  Print the installed package version.

Hidden internal commands also exist for scheduled jobs:

- `bff md CONFIG.yaml`
- `bff reference-job CONFIG.yaml`

## Config Philosophy

Each top-level workflow uses one focused config file:

- build config: how to stage reusable FFMD and reference assets
- reference config: how to run or import canonical reference data
- sample config: how to turn prepared assets into a sampled FFMD campaign
- analyze config: how to compute observables from trajectories
- fit config: how to train surrogates
- learn config: which models and MCMC settings to use for posterior learning
- validate config: how to rerun chosen parameter samples

Detailed key-by-key documentation is in the configuration reference.

## Shell Completion

When `bff` runs inside an activated conda environment, it writes a small
completion hook for bash and zsh into that environment. The hook is deliberately
minimal: `bff <TAB>` shows the public workflows and completes config-file
arguments for commands that expect a path.

After the first `bff` run, reactivate the environment once:

```bash
conda deactivate
conda activate bfflearn
```

After that, `bff <TAB>` should offer:

- `build`
- `reference`
- `sample`
- `analyze`
- `fit`
- `learn`
- `validate`
- `examples`
