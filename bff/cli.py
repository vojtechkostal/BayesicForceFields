from pathlib import Path
from typing import Callable

import typer


app = typer.Typer(
    help="BayesicForceFields: Bayesian optimization of molecular force fields.",
    no_args_is_help=True,
)

WorkflowMain = Callable[[Path], None]


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


def config_argument() -> Path:
    return typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the configuration file.",
    )


@app.command()
def version() -> None:
    """
    Print package version.
    """
    from importlib import metadata

    try:
        version_str = metadata.version("BayesicForceFields")
    except metadata.PackageNotFoundError:
        from . import __version__

        version_str = __version__

    typer.echo(f"BayesicForceFields version: {version_str}")


@app.command()
def initialize(fn_config: Path = config_argument()) -> None:
    """
    Initialize the project from a configuration file.
    """
    from bff.workflows.initialize import main as initialize_main

    run_workflow(fn_config, initialize_main, "initialize")


@app.command()
def runsims(fn_config: Path = config_argument()) -> None:
    """
    Run training or validation molecular simulations.
    """
    from bff.workflows.runsims import main as runsims_main

    run_workflow(fn_config, runsims_main, "runsims")


@app.command()
def md(fn_config: Path = config_argument()) -> None:
    """
    Run molecular dynamics from a configuration file.
    """
    from bff.workflows.md import main as md_main

    run_workflow(fn_config, md_main, "md")


@app.command()
def analyze(fn_config: Path = config_argument()) -> None:
    """
    Analyze quantities of interest from simulation data.
    """
    from bff.workflows.analyze_qoi import main as analyze_qoi_main

    run_workflow(fn_config, analyze_qoi_main, "analyze")


@app.command()
def learn(fn_config: Path = config_argument()) -> None:
    """
    Run surrogate training and posterior learning.
    """
    from bff.workflows.learn import main as learn_main

    run_workflow(fn_config, learn_main, "learn")


if __name__ == "__main__":
    app()
