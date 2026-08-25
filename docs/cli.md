# Command-Line Interface

The CLI entry point is implemented in `bff/cli.py`.

## Public Commands

- `bff build CONFIG.yaml`
  Build equilibrated systems and seeded production trajectories.
- `bff label-snapshots CONFIG.yaml`
  Extract trajectory snapshots and label them with CP2K.
- `bff sample-parameters CONFIG.yaml`
  Sample force-field parameters and run FFMD campaigns.
- `bff build-qoi-datasets CONFIG.yaml`
  Analyze sampled and reference trajectories into matched QoI datasets.
- `bff fit-lgp CONFIG.yaml`
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
- `bff label-snapshot-job CONFIG.yaml`

## Config Philosophy

Each top-level workflow uses one focused config file:

- build config: how to equilibrate and seed production trajectories
- label-snapshots config: how to extract and label trajectory snapshots
- sample-parameters config: how to turn build systems into a sampled FFMD campaign
- build-qoi-datasets config: how to compute observables from trajectories
- fit-lgp config: how to train surrogates
- learn config: which models and MCMC settings to use for posterior learning
- validate config: how to rerun chosen parameter samples

Detailed key-by-key documentation is in the configuration reference.

## Shell Completion

`bff` uses Typer's native shell-completion support. To enable completion in the
current bash session, run:

```bash
eval "$(bff --show-completion bash)"
```

For zsh, run:

```zsh
eval "$(bff --show-completion zsh)"
```

Add the matching line to `~/.bashrc` or `~/.zshrc` if you want completion in
future shells. After completion is loaded, `bff <TAB>` should offer:

- `build`
- `label-snapshots`
- `sample-parameters`
- `build-qoi-datasets`
- `fit-lgp`
- `learn`
- `validate`
- `examples`
