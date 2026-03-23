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
