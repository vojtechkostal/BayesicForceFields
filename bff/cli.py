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

WorkflowMain = Callable[[Path], object]
WORKFLOW_COMMANDS = (
    "build",
    "reference",
    "sample",
    "analyze",
    "fit",
    "learn",
    "validate",
    "examples",
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
            build|reference|sample|analyze|fit|learn|validate)
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
            build|reference|sample|analyze|fit|learn|validate)
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
) -> object:
    try:
        return workflow_main(fn_config)
    except FileNotFoundError as exc:
        missing = getattr(exc, 'filename', None)
        if missing is not None and Path(missing).resolve() != fn_config.resolve():
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
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
    _ensure_completion_hooks()


@app.command()
def version() -> None:
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


@app.command()
def build(fn_config: Path = config_argument()) -> None:
    """Build reusable FFMD and reference starting assets."""
    from bff.workflows.build import main as build_main

    run_workflow(fn_config, build_main, "build")


@app.command()
def reference(fn_config: Path = config_argument()) -> None:
    """Run or import reference data into canonical reference assets."""
    from bff.workflows.reference import main as reference_main

    run_workflow(fn_config, reference_main, "reference")


@app.command()
def sample(fn_config: Path = config_argument()) -> None:
    """Sample force-field parameters and run FFMD training simulations."""
    from bff.workflows.sample import main as sample_main

    run_workflow(fn_config, sample_main, "sample")


@app.command()
def analyze(fn_config: Path = config_argument()) -> None:
    """Analyze reference and FFMD trajectories into matched QoI datasets."""
    from bff.workflows.analyze import main as analyze_main

    run_workflow(fn_config, analyze_main, "analyze")


@app.command()
def fit(fn_config: Path = config_argument()) -> None:
    """Fit surrogate models from analyzed QoI datasets."""
    from bff.workflows.fit import main as fit_main

    run_workflow(fn_config, fit_main, "fit")


@app.command()
def learn(fn_config: Path = config_argument()) -> None:
    """Run Bayesian posterior learning over force-field parameters."""
    from bff.workflows.learn import main as learn_main

    run_workflow(fn_config, learn_main, "learn")


@app.command()
def validate(fn_config: Path = config_argument()) -> None:
    """Run explicit validation simulations for selected parameter samples."""
    from bff.workflows.validate import main as validate_main

    run_workflow(fn_config, validate_main, "validate")


@app.command(hidden=True)
def md(fn_config: Path = config_argument()) -> None:
    """Run molecular dynamics from a configuration file."""
    from bff.workflows.md import main as md_main

    run_workflow(fn_config, md_main, "md")


@app.command(name="reference-job", hidden=True)
def reference_job(fn_config: Path = config_argument()) -> None:
    """Run one staged CP2K reference job from a configuration file."""
    from bff.workflows.reference import run_job

    run_workflow(fn_config, run_job, "reference-job")


if __name__ == "__main__":
    app()
