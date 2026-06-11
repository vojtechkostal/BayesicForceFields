import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bff.workflows import examples as examples_workflow

EXAMPLES = Path(__file__).parents[1] / "examples"
NOTEBOOKS = (
    EXAMPLES / "arbitrary-data" / "arbitrary-data.ipynb",
    EXAMPLES / "neon-mie-lgpmd" / "neon-mie-inference.ipynb",
)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_notebook_examples_are_clean(notebook: Path) -> None:
    cells = json.loads(notebook.read_text())["cells"]

    assert all(not cell.get("outputs") for cell in cells)
    assert all(cell.get("execution_count") is None for cell in cells)
    assert all(
        "".join(cell.get("source", [])).strip()
        for cell in cells
        if cell["cell_type"] == "code"
    )


def test_notebook_examples_use_cuda_by_default() -> None:
    for notebook in NOTEBOOKS:
        source = notebook.read_text()
        assert "device='cuda'" in source or 'device=\\"cuda\\"' in source


def test_notebook_examples_write_qoi_marginals() -> None:
    for notebook in NOTEBOOKS:
        source = notebook.read_text()
        assert "gaussian_log_likelihood_by_qoi" in source
        assert "plot_qoi_marginals" in source
        assert "qoi-marginals.pdf" in source


def test_acetate_configs_use_current_effective_observation_schema() -> None:
    fit = yaml.safe_load((EXAMPLES / "acetate/configs/fit.yaml").read_text())
    learn = yaml.safe_load((EXAMPLES / "acetate/configs/learn.yaml").read_text())

    assert all(
        "observation_scale" not in dataset
        for dataset in fit["datasets"].values()
    )
    assert learn["models"]["rdf"]["tolerance"] > 0
    assert learn["models"]["hb"]["independent_observations"] is True
    assert learn["models"]["dist"]["tolerance"] > 0
    assert all(
        "model_path" in model
        for model in learn["models"].values()
    )


def test_neon_notebook_uses_local_pmf_mean() -> None:
    source = NOTEBOOKS[1].read_text()

    assert "class MieRDFPMFMean:" in source
    assert 'y_means={\\"rdf\\": pmf_mean}' in source
    assert "# y_means" not in source


def test_default_example_ref_uses_installed_version(monkeypatch) -> None:
    monkeypatch.setattr(examples_workflow, "_installed_version", lambda: "0.2.0")

    assert examples_workflow._default_ref() == "v0.2.0"


def test_local_example_copy_only_includes_listed_files(tmp_path, monkeypatch) -> None:
    source_dir = tmp_path / "examples"
    output_dir = tmp_path / "copied"
    tracked = source_dir / "demo" / "README.md"
    generated = source_dir / "demo" / "generated" / "results.pt"
    tracked.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    tracked.write_text("tracked")
    generated.write_text("generated")

    monkeypatch.setattr(
        examples_workflow.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="examples/demo/README.md\0"),
    )

    examples_workflow._copy_local_examples(source_dir, output_dir)

    assert (output_dir / "demo" / "README.md").read_text() == "tracked"
    assert not (output_dir / "demo" / "generated").exists()


def test_archive_extraction_rejects_parent_paths(tmp_path) -> None:
    archive_path = tmp_path / "examples.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"outside"
        member = tarfile.TarInfo("repository/examples/../../outside.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(RuntimeError, match="unsafe path"):
        examples_workflow._extract_examples_archive(
            archive_path,
            tmp_path / "examples",
        )
