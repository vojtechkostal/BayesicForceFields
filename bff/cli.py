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
    "examples",
    "cp2k-collect",
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
        version_str = metadata.version("bfflearn")
    except metadata.PackageNotFoundError:
        from . import __version__

        version_str = __version__

    typer.echo(f"BayesicForceFields version: {version_str}")


@app.command()
def examples(
    output_dir: Path = typer.Option(
        Path("examples"),
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=False,
        help="Directory to write the example tree into.",
    ),
    ref: str | None = typer.Option(
        None,
        "--ref",
        help="Git ref to download examples from. Defaults to the installed tag.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Replace the output directory if it already exists.",
    ),
) -> None:
    """
    Download or copy all repository examples.
    """
    from bff.workflows.examples import fetch_examples

    try:
        examples_dir, source = fetch_examples(output_dir, ref=ref, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc), param_hint="output_dir") from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Examples written to: {examples_dir}")
    typer.echo(f"Source: {source}")


@app.command(name="cp2k-collect")
def cp2k_collect(
    runs: Path = typer.Option(
        Path("runs"),
        "--runs",
        file_okay=False,
        dir_okay=True,
        resolve_path=False,
        help="Directory containing per-snapshot CP2K run directories.",
    ),
    train: Path = typer.Option(
        Path("train.extxyz"),
        "--train",
        dir_okay=False,
        resolve_path=False,
        help="Output training extxyz file.",
    ),
    valid: Path = typer.Option(
        Path("valid.extxyz"),
        "--valid",
        dir_okay=False,
        resolve_path=False,
        help="Output validation extxyz file.",
    ),
    train_fraction: float = typer.Option(
        0.8,
        "--train-fraction",
        min=0.0,
        max=1.0,
        help="Fraction of collected frames written to the training split.",
    ),
    seed: int = typer.Option(
        2026,
        "--seed",
        help="Seed for the deterministic shuffled train/validation split.",
    ),
    topology: str = typer.Option(
        "pos.xyz",
        "--topology",
        help="Per-run topology file used when CP2K outputs DCD files.",
    ),
) -> None:
    """
    Collect CP2K snapshot runs into train/validation extxyz files.
    """
    from bff.io.cp2k_collect import collect_outputs

    try:
        n_train, n_valid = collect_outputs(
            runs=runs,
            train=train,
            valid=valid,
            train_fraction=train_fraction,
            seed=seed,
            topology_name=topology,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="cp2k-collect") from exc

    typer.echo(f"Wrote {n_train} training frames to {train}")
    typer.echo(f"Wrote {n_valid} validation frames to {valid}")


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
