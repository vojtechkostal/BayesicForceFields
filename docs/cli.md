# Command-Line Interface

The CLI entry point is implemented in `bff/cli.py`.

## Public Commands

- `bff build CONFIG.yaml`
  Build equilibrated systems and seeded production trajectories.
- `bff prepare-reference CONFIG.yaml`
  Prepare CP2K reference inputs from a completed build stage.
- `bff evaluate-snapshots CONFIG.yaml`
  Evaluate staged CP2K snapshots.
- `bff sample CONFIG.yaml`
  Sample force-field parameters and run FFMD campaigns.
- `bff analyze CONFIG.yaml`
  Analyze sampled and reference trajectories into matched QoI datasets.
- `bff lgpfit CONFIG.yaml`
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
- `bff evaluate-snapshot-job CONFIG.yaml`

## Config Philosophy

Each top-level workflow uses one focused config file:

- build config: how to equilibrate and seed production trajectories
- prepare-reference config: how to stage CP2K inputs from build systems
- evaluate-snapshots config: how to run CP2K snapshot jobs
- sample config: how to turn build systems into a sampled FFMD campaign
- analyze config: how to compute observables from trajectories
- lgpfit config: how to train surrogates
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
- `prepare-reference`
- `evaluate-snapshots`
- `sample`
- `analyze`
- `lgpfit`
- `learn`
- `validate`
- `examples`
