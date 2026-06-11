"""Command-line entry points for Bayesic Force Fields."""

from pathlib import Path
from typing import Callable

import typer

app = typer.Typer(
    help="BayesicForceFields: Bayesian optimization of molecular force fields.",
    no_args_is_help=True,
    add_completion=True,
)

WorkflowMain = Callable[[Path], object]


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
    """Build equilibrated systems and seeded production trajectories."""
    from bff.workflows.build import main as build_main

    run_workflow(fn_config, build_main, "build")


@app.command(name="prepare-assets")
def prepare_assets(fn_config: Path = config_argument()) -> None:
    """Prepare FFMD and snapshot evaluation assets from a build manifest."""
    from bff.workflows.prepare_assets import main as prepare_main

    run_workflow(fn_config, prepare_main, "prepare-assets")


@app.command(name="evaluate-snapshots")
def evaluate_snapshots(fn_config: Path = config_argument()) -> None:
    """Evaluate staged CP2K snapshots."""
    from bff.workflows.evaluate_snapshots import main as evaluate_main

    run_workflow(fn_config, evaluate_main, "evaluate-snapshots")


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


@app.command(name="evaluate-snapshot-job", hidden=True)
def evaluate_snapshot_job(fn_config: Path = config_argument()) -> None:
    """Run one staged CP2K snapshot job from a configuration file."""
    from bff.workflows.evaluate_snapshots import run_job

    run_workflow(fn_config, run_job, "evaluate-snapshot-job")


if __name__ == "__main__":
    app()
