from pathlib import Path

from typer.testing import CliRunner

import bff
from bff.cli import app


def test_cli_exposes_refactored_workflow_names() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    for command in (
        "build",
        "sample-parameters",
        "build-qoi-datasets",
        "fit-lgp",
    ):
        assert command in help_result.stdout
    for removed in (
        "label-snapshots",
        "label-snapshot-job",
        "prepare-reference",
        "evaluate-snapshots",
        "sample",
        "analyze",
        "lgpfit",
    ):
        result = runner.invoke(app, [removed, "missing.yaml"])
        assert "No such command" in result.stdout + result.stderr


def test_python_api_exposes_only_refactored_workflow_names(tmp_path: Path) -> None:
    names = (
        "build",
        "sample_parameters",
        "build_qoi_datasets",
        "fit_lgp",
    )
    project = bff.Project(tmp_path)
    assert all(callable(getattr(bff, name)) for name in names)
    assert all(callable(getattr(project, name)) for name in names)
    assert not any(
        hasattr(bff, name)
        for name in (
            "label_snapshots",
            "prepare_reference",
            "evaluate_snapshots",
            "sample",
            "analyze",
            "lgpfit",
        )
    )
