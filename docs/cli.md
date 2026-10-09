# Command-Line Interface

| Command | Does |
| --- | --- |
| `bff build CONFIG` | Build and equilibrate the systems. |
| `bff sample-parameters CONFIG` | Draw parameter sets and run their MD. |
| `bff build-qoi-datasets CONFIG` | Compute the QoIs of the samples and the reference. |
| `bff fit-lgp CONFIG` | Fit a surrogate per QoI. |
| `bff learn CONFIG` | Sample the posterior. |
| `bff validate CONFIG` | Run MD for posterior or explicit parameters. |
| `bff examples` | Copy the examples matching the installed version. |
| `bff version` | Print the version. |

`bff md` is internal: every campaign sample runs as `bff md campaign.yaml
<sample_id>`. The options of each config are on the
[settings overview](configuration/index.md). Errors are reported as one line
that names the key or file at fault.

The same stages are available from Python:

```python
import bff

bff.learn("06-learn/config.yaml")
bff.Project("examples/acetate").fit_lgp("05-fit-lgp/config.yaml")
```

## Shell completion

```bash
eval "$(bff --show-completion bash)"   # zsh: bff --show-completion zsh
```

Add the line to `~/.bashrc` (or `~/.zshrc`) to keep it; `bff <TAB>` then
offers the commands. Completion works only where the `bff` command of the
active environment is on `PATH`.
