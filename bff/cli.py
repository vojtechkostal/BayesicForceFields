import typer
from pathlib import Path
from typing import Optional, Callable


app = typer.Typer(
    help="BayesicForceFields: Bayesian optimization of molecular force fields.")


@app.command()
def version():
    """
    Print package version.
    """
    import bff
    typer.echo(f"BayesicForceFields version: {bff.__version__}")


@app.command()
def help():
    """
    Print help message.
    """
    typer.echo("Use as 'bff <command> fn_config'")
    typer.echo("\nAvailable commands:")
    typer.echo("  initialize   Creates the system")
    typer.echo("  runsims      Run MDs (training or validation).")
    typer.echo("  optimize     Runs the Bayesian inference of the force field parameters.")
    typer.echo("  version      Show package version.")


def run_workflow(
    fn_config: Optional[Path],
    workflow_main: Callable[[Path], None],
    workflow_name: str
):
    """
    Helper to run a workflow main function with config path,
    or print a friendly message if no config provided.
    """
    if fn_config:
        workflow_main(fn_config)
    else:
        typer.echo(
            f"No configuration file provided for {workflow_name} workflow. "
            "Please specify a config file.")


@app.command()
def initialize(
    fn_config: Optional[Path] = typer.Argument(
        None, help="Path to the configuration file (required)."
    )
):
    """
    Initialize the BayesicForceFields package with the given configuration file.
    """
    from bff.workflows.initialize import main as initialize_main
    run_workflow(fn_config, initialize_main, "initialize")


@app.command()
def runsims(
    fn_config: Optional[Path] = typer.Argument(
        None, help="Path to the configuration file (required)."
    )
):
    """
    Run molecular simulations as configured in the given config file.
    """

    from bff.workflows.runsims import main as runsims_main
    run_workflow(fn_config, runsims_main, "runsims")


@app.command()
def optimize(
    fn_config: Optional[Path] = typer.Argument(
        None, help="Path to the configuration file (required)."
    )
):
    """
    Run molecular simulations as configured in the given config file.
    """
    from bff.workflows.optimize import main as optimize_main
    run_workflow(fn_config, optimize_main, "optimize")


if __name__ == "__main__":
    app()
