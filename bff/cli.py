"""Command-line entry points for Bayesic Force Fields."""

import os
from pathlib import Path
from typing import Callable

import typer


app = typer.Typer(
    help="BayesicForceFields: Bayesian optimization of molecular force fields.",
    no_args_is_help=True,
    add_completion=False,
)

WorkflowMain = Callable[[Path], None]
WORKFLOW_COMMANDS = (
    "prepare",
    "simulate",
    "qoi",
    "train",
    "learn",
    "validate",
)

_COMPLETION_ACTIVATE = f"""# Managed automatically by Bayesic Force Fields.
if [ -n "${{BASH_VERSION-}}" ]; then
    _bff_completion() {{
        local cur prev words cword
        if declare -F _init_completion >/dev/null 2>&1; then
            _init_completion || return
        else
            COMPREPLY=()
            cur="${{COMP_WORDS[COMP_CWORD]}}"
            prev="${{COMP_WORDS[COMP_CWORD-1]}}"
            words=("${{COMP_WORDS[@]}}")
            cword=$COMP_CWORD
        fi

        local commands="{" ".join(WORKFLOW_COMMANDS)}"
        if [[ $COMP_CWORD -eq 1 ]]; then
            COMPREPLY=($(compgen -W "$commands" -- "$cur"))
            return 0
        fi

        case "${{COMP_WORDS[1]}}" in
            prepare|simulate|qoi|train|learn|validate)
                COMPREPLY=($(compgen -f -- "$cur"))
                ;;
            *)
                COMPREPLY=()
                ;;
        esac
    }}

    complete -o nosort -F _bff_completion bff
elif [ -n "${{ZSH_VERSION-}}" ]; then
    autoload -Uz compinit 2>/dev/null || true
    compinit -i 2>/dev/null || true

    _bff_completion() {{
        local -a commands
        commands=({" ".join(WORKFLOW_COMMANDS)})

        if (( CURRENT == 2 )); then
            compadd -- "${{commands[@]}}"
            return 0
        fi

        case "${{words[2]}}" in
            prepare|simulate|qoi|train|learn|validate)
                _files
                ;;
        esac
    }}

    compdef _bff_completion bff
fi
"""

_COMPLETION_DEACTIVATE = """# Managed automatically by Bayesic Force Fields.
if [ -n "${BASH_VERSION-}" ]; then
    complete -r bff 2>/dev/null || true
elif [ -n "${ZSH_VERSION-}" ]; then
    unfunction _bff_completion 2>/dev/null || true
    compdef -d bff 2>/dev/null || true
fi
"""


def _conda_completion_hook_paths() -> tuple[Path, Path] | None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    conda_dir = Path(conda_prefix) / "etc" / "conda"
    return (
        conda_dir / "activate.d" / "bff-completion.sh",
        conda_dir / "deactivate.d" / "bff-completion.sh",
    )


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _ensure_completion_hooks() -> None:
    hook_paths = _conda_completion_hook_paths()
    if hook_paths is None:
        return

    activate_hook, deactivate_hook = hook_paths
    try:
        _write_if_changed(activate_hook, _COMPLETION_ACTIVATE)
        _write_if_changed(deactivate_hook, _COMPLETION_DEACTIVATE)
    except OSError:
        return


def run_workflow(
    fn_config: Path,
    workflow_main: WorkflowMain,
    workflow_name: str,
) -> None:
    """
    Run a workflow main function with a config path.
    """
    try:
        workflow_main(fn_config)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="fn_config") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=workflow_name) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


def config_argument() -> Path:
    return typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the configuration file.",
    )


@app.callback()
def init_cli() -> None:
    """
    Initialize the CLI environment.
    """
    _ensure_completion_hooks()


@app.command()
def version() -> None:
    """
    Print package version.
    """
    from importlib import metadata

    try:
        version_str = metadata.version("bbflearn")
    except metadata.PackageNotFoundError:
        from . import __version__

        version_str = __version__

    typer.echo(f"BayesicForceFields version: {version_str}")


@app.command()
def prepare(fn_config: Path = config_argument()) -> None:
    """
    Prepare equilibration, training, and reference assets.
    """
    from bff.workflows.prepare import main as prepare_main

    run_workflow(fn_config, prepare_main, "prepare")


@app.command(name="simulate")
def simulate(fn_config: Path = config_argument()) -> None:
    """
    Run a sampled molecular-dynamics campaign for training-set generation.
    """
    from bff.workflows.simulate import main as simulate_main

    run_workflow(fn_config, simulate_main, "simulate")


@app.command()
def validate(fn_config: Path = config_argument()) -> None:
    """
    Run a validation campaign for provided parameter samples.
    """
    from bff.workflows.validate import main as validate_main

    run_workflow(fn_config, validate_main, "validate")


@app.command(hidden=True)
def md(fn_config: Path = config_argument()) -> None:
    """
    Run molecular dynamics from a configuration file.
    """
    from bff.workflows.md import main as md_main

    run_workflow(fn_config, md_main, "md")


@app.command(name="qoi")
def qoi(fn_config: Path = config_argument()) -> None:
    """
    Analyze quantities of interest from simulation data.
    """
    from bff.workflows.qoi import main as qoi_main

    run_workflow(fn_config, qoi_main, "qoi")


@app.command(name="analyze", hidden=True)
def analyze_alias(fn_config: Path = config_argument()) -> None:
    """
    Hidden compatibility alias for the QoI workflow.
    """
    qoi(fn_config)


@app.command()
def train(fn_config: Path = config_argument()) -> None:
    """
    Train surrogate models from analyzed QoI datasets.
    """
    from bff.workflows.train import main as train_main

    run_workflow(fn_config, train_main, "train")


@app.command()
def learn(fn_config: Path = config_argument()) -> None:
    """
    Run posterior learning from trained surrogate models.
    """
    from bff.workflows.learn import main as learn_main

    run_workflow(fn_config, learn_main, "learn")


if __name__ == "__main__":
    app()
