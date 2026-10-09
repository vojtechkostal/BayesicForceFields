import importlib.util
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bff.workflows import examples as examples_workflow
from bff.workflows.build.config import BuildConfig
from bff.workflows.build_qoi_datasets.config import BuildQoIDatasetsConfig
from bff.workflows.fit_lgp.config import FitLGPConfig
from bff.workflows.learn.config import LearnConfig
from bff.workflows.sample_parameters.config import SampleParametersConfig
from bff.workflows.validate.config import ValidateConfig

EXAMPLES = Path(__file__).parents[1] / "examples"
ACETATE = EXAMPLES / "acetate"
_spec = importlib.util.spec_from_file_location(
    "label_structures",
    Path(__file__).parents[1] / "scripts" / "label_structures.py",
)
label_structures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(label_structures)
NOTEBOOKS = (
    EXAMPLES / "acetate" / "06-learn" / "posterior.ipynb",
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
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook}:{cell.get('id')}", "exec")


def test_data_notebooks_let_bff_choose_the_device() -> None:
    for notebook in NOTEBOOKS[1:]:
        source = notebook.read_text()
        assert "cuda" not in source.lower()
        assert "DEVICE" not in source


def test_notebook_examples_write_qoi_marginals() -> None:
    for notebook in NOTEBOOKS[1:]:
        source = notebook.read_text()
        assert "plot_qoi_marginals" in source


def test_acetate_learn_config_sets_tolerances_per_qoi() -> None:
    learn = yaml.safe_load((ACETATE / "06-learn/config.yaml").read_text())

    assert learn["models"]["rdf"]["tolerance"] > 0
    assert set(learn["models"]["hb"]) == {"model_path"}
    assert learn["models"]["contact-distance"]["tolerance"] > 0
    assert all("model_path" in model for model in learn["models"].values())


def test_acetate_includes_cp2k_inputs_for_labeling() -> None:
    inputs = ACETATE / "02-reference-md/cp2k"
    names = {path.name for path in inputs.glob("*.inp")}
    assert names == {
        "sp-0.inp",
        "sp-1.inp",
        "sp-2.inp",
        "single-atom-h.inp",
        "single-atom-c.inp",
        "single-atom-o.inp",
        "single-atom-ca.inp",
    }
    assert "&HF" in (inputs / "sp-0.inp").read_text()
    for path in inputs.glob("*.inp"):
        text = path.read_text()
        assert "COORD_FILE_NAME structure.xyz" in text
        assert ("@INCLUDE cell.inc" in text) is path.name.startswith("sp-")
    for index, charge in enumerate((-1, 1, 1)):
        assert f"CHARGE {charge}" in (inputs / f"sp-{index}.inp").read_text()


def test_neon_notebook_uses_local_pmf_mean() -> None:
    source = NOTEBOOKS[2].read_text()

    assert "def pmf_mean(X):" in source
    assert "means={'rdf': pmf_mean}" in source
    assert "y_means" not in source


def test_acetate_uses_numbered_stage_contract() -> None:
    stages = {
        "01-build": {"config.yaml", "config-plumed.yaml"},
        "03-sample-parameters": {"config.yaml", "config-slurm.yaml"},
        "04-build-qoi-datasets": {"config.yaml"},
        "05-fit-lgp": {"config.yaml"},
        "06-learn": {"config.yaml"},
        "07-validate": {"config.yaml"},
    }
    for stage, names in stages.items():
        assert {p.name for p in (ACETATE / stage).glob("config*.yaml")} == names

    qoi = yaml.safe_load((ACETATE / "04-build-qoi-datasets/config.yaml").read_text())
    assert qoi["training_samples"]["manifest"] == (
        "../03-sample-parameters/samples.yaml"
    )
    for system in qoi["reference"]["systems"]:
        trajectory = ACETATE / "04-build-qoi-datasets" / system["inputs"]["trajectory"]
        assert trajectory.is_file()  # the committed reference trajectory

    source = NOTEBOOKS[0].read_text()
    assert "Results.load('outputs/results.pt')" in source
    assert "results.draw(" in source


def test_acetate_configs_load_against_staged_contract(tmp_path: Path) -> None:
    """Copy configs and inputs, stage the files earlier stages would write, and
    load every config with the parser the CLI uses."""
    shutil.copytree(ACETATE / "inputs", tmp_path / "inputs")
    for stage in (
        "01-build",
        "02-reference-md",
        "03-sample-parameters",
        "04-build-qoi-datasets",
        "05-fit-lgp",
        "06-learn",
        "07-validate",
    ):
        (tmp_path / stage).mkdir()
        for config in (ACETATE / stage).glob("*.yaml"):
            shutil.copy2(config, tmp_path / stage / config.name)
    for name in ("trajectories", "cp2k"):
        shutil.copytree(
            ACETATE / "02-reference-md" / name, tmp_path / "02-reference-md" / name
        )

    system_ids = ("acetate", "acetate-contact", "acetate-separated")
    for system_id in system_ids:
        system_dir = tmp_path / "01-build/systems" / system_id
        (system_dir / "reference").mkdir(parents=True)
        for filename in (
            "topology.top",
            "coordinates.gro",
            "em.mdp",
            "npt.mdp",
            "production.mdp",
            "index.ndx",
            "production.gro",
            "production.xtc",
            "reference/topology.top",
            "reference/coordinates.gro",
        ):
            (system_dir / filename).write_text("staged fixture\n")
    for name in ("samples.yaml", "specs.yaml"):
        (tmp_path / "03-sample-parameters" / name).write_text("staged fixture\n")
    for directory, suffix, names in (
        ("04-build-qoi-datasets/qoi", "pt", ("rdf", "hb")),
        ("05-fit-lgp/models", "lgp", ("rdf", "hb")),
        ("06-learn/outputs", "pt", ("results",)),
    ):
        (tmp_path / directory).mkdir()
        for name in names:
            (tmp_path / directory / f"{name}.{suffix}").write_text("staged fixture\n")
    for name in ("contact-distance", "separated-distance"):
        (tmp_path / "04-build-qoi-datasets/qoi" / f"{name}.pt").write_text("fixture\n")
        (tmp_path / "05-fit-lgp/models" / f"{name}.lgp").write_text("fixture\n")

    for name in ("config.yaml", "config-plumed.yaml"):
        BuildConfig.load(tmp_path / "01-build" / name)
    for name in ("config.yaml", "config-slurm.yaml"):
        SampleParametersConfig.load(tmp_path / "03-sample-parameters" / name)
    BuildQoIDatasetsConfig.load(tmp_path / "04-build-qoi-datasets/config.yaml")
    FitLGPConfig.load(tmp_path / "05-fit-lgp/config.yaml")
    LearnConfig.load(tmp_path / "06-learn/config.yaml")
    ValidateConfig.load(tmp_path / "07-validate/config.yaml")

    label_root = tmp_path / "02-reference-md"
    label_config = label_structures.load_config(label_root / "label-structures.yaml")
    for system in label_config["systems"]:
        assert (label_root / system["cp2k_input"]).is_file()
    for cp2k_input in label_config["single_atom_inputs"].values():
        assert (label_root / cp2k_input).is_file()


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
