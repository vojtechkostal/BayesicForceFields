# Command-Line Interface

The CLI entry point is implemented in
`bff/cli.py`.

## Public Commands

- `bff prepare CONFIG.yaml`
  Stage equilibrated systems, reference inputs, and training assets.
- `bff simulate CONFIG.yaml`
  Run a sampled GROMACS campaign from prepared assets.
- `bff qoi CONFIG.yaml`
  Compute quantities of interest from trainset and reference trajectories.
- `bff train CONFIG.yaml`
  Fit surrogate models from analyzed QoI datasets.
- `bff learn CONFIG.yaml`
  Run posterior inference from selected trained surrogate models.
- `bff validate CONFIG.yaml`
  Rerun selected parameter samples for validation.
- `bff examples`
  Copy or download the repository example tree.
- `bff cp2k-collect`
  Collect CP2K snapshot runs into `train.extxyz` and `valid.extxyz`.
- `bff version`
  Print the installed package version.

`bff md` exists as an internal low-level command used by scheduled campaign
jobs and is intentionally hidden from normal workflow navigation.

## Config Philosophy

Each top-level workflow uses one focused config file:

- prepare config: how to stage systems and reusable assets
- simulate config: how to turn prepared assets into a sampled trainset
- QoI config: how to compute observables from trajectories
- train config: how to fit surrogates
- learn config: which models to use for posterior inference
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

- `prepare`
- `simulate`
- `qoi`
- `train`
- `learn`
- `validate`
- `examples`
- `cp2k-collect`
