import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bff.domain.systems import BuildSystemMetadata, write_build_system_metadata
from bff.workflows import examples as examples_workflow
from bff.workflows.build.config import BuildConfig
from bff.workflows.build_qoi_datasets.config import BuildQoIDatasetsConfig
from bff.workflows.fit_lgp.config import FitLGPConfig
from bff.workflows.label_snapshots.config import LabelSnapshotsConfig
from bff.workflows.learn.config import LearnConfig
from bff.workflows.sample_parameters.config import SampleParametersConfig
from bff.workflows.validate.config import ValidateConfig

EXAMPLES = Path(__file__).parents[1] / "examples"
NOTEBOOKS = (
    EXAMPLES / "acetate" / "notebooks" / "interactive.ipynb",
    EXAMPLES / "acetate" / "notebooks" / "visualize.ipynb",
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


def test_notebook_examples_select_an_available_device() -> None:
    for notebook in NOTEBOOKS[2:]:
        source = notebook.read_text()
        assert "torch.cuda.is_available()" in source
        assert source.count("device=DEVICE") == 2
        assert "device='cuda'" not in source
        assert 'device=\\"cuda\\"' not in source


def test_notebook_examples_write_qoi_marginals() -> None:
    for notebook in NOTEBOOKS[2:]:
        source = notebook.read_text()
        assert "gaussian_log_likelihood_by_qoi" in source
        assert "plot_qoi_marginals" in source
        assert "qoi-marginals.pdf" in source


def test_acetate_configs_use_current_effective_observation_schema() -> None:
    fit_lgp = yaml.safe_load(
        (EXAMPLES / "acetate/configs/05-fit-lgp.yaml").read_text()
    )
    learn = yaml.safe_load(
        (EXAMPLES / "acetate/configs/06-learn.yaml").read_text()
    )

    assert all(
        "observation_scale" not in dataset
        for dataset in fit_lgp["datasets"].values()
    )
    assert learn["models"]["rdf"]["tolerance"] > 0
    assert learn["models"]["hb"]["independent_observations"] is True
    assert learn["models"]["contact-distance"]["tolerance"] > 0
    assert all(
        "model_path" in model
        for model in learn["models"].values()
    )


def test_acetate_includes_explicit_cp2k_functional_families() -> None:
    inputs = EXAMPLES / "acetate/inputs/reference-inputs"
    required = {
        "sp-0.inp",
        "sp-1.inp",
        "sp-2.inp",
        "single-atom-h.inp",
        "single-atom-c.inp",
        "single-atom-o.inp",
        "single-atom-ca.inp",
    }
    for family, uses_hybrid_exchange in (
        ("revpbe-d3", False),
        ("revpbe0-d3", True),
    ):
        family_dir = inputs / family
        assert required <= {path.name for path in family_dir.glob("*.inp")}
        assert ("&HF" in (family_dir / "sp-0.inp").read_text()) is (
            uses_hybrid_exchange
        )

    label_config = yaml.safe_load(
        (
            EXAMPLES
            / "acetate/configs/02-reference-snapshots-local.yaml"
        ).read_text()
    )
    xtb_dir = inputs / "xtb"
    for index, charge in enumerate((-1, 1, 1)):
        md_input = (xtb_dir / f"md-{index}.inp").read_text()
        assert "METHOD XTB" in md_input
        assert f"CHARGE {charge}" in md_input
    assert all(
        system["single_atom_inputs"]
        for system in label_config["systems"]
    )
    assert all(
        "/xtb/md-" in system["md_input"]
        for system in label_config["systems"]
    )


def test_neon_notebook_uses_local_pmf_mean() -> None:
    source = NOTEBOOKS[3].read_text()

    assert "class MieRDFPMFMean:" in source
    assert 'y_means={\\"rdf\\": pmf_mean}' in source
    assert "# y_means" not in source


def test_acetate_uses_numbered_stage_contract() -> None:
    config_names = {
        path.name for path in (EXAMPLES / "acetate/configs").glob("*.yaml")
    }
    assert config_names == {
        "01-build-colvars.yaml",
        "01-build-plumed.yaml",
        "02-reference-snapshots-local.yaml",
        "02-reference-snapshots-slurm.yaml",
        "03-sample-local.yaml",
        "03-sample-slurm.yaml",
        "04-build-qoi-datasets.yaml",
        "05-fit-lgp.yaml",
        "06-learn.yaml",
        "07-validate.yaml",
    }

    qoi = yaml.safe_load(
        (EXAMPLES / "acetate/configs/04-build-qoi-datasets.yaml").read_text()
    )
    assert qoi["training_samples"]["manifest"] == "../03-sample/samples.yaml"
    assert all(
        system["inputs"]["trajectory"].startswith("../02-reference-md/")
        for system in qoi["reference"]["systems"]
    )

    notebook_source = "\n".join(
        path.read_text() for path in NOTEBOOKS[:2]
    )
    assert "PosteriorResults.load(LEARN_OUTPUTS / 'posterior.pt')" in notebook_source
    assert "sample_posterior(" in notebook_source
    assert "sample_parameters(" not in notebook_source


def test_acetate_configs_load_against_staged_contract(tmp_path: Path) -> None:
    acetate = EXAMPLES / "acetate"
    shutil.copytree(acetate / "inputs", tmp_path / "inputs")

    system_ids = ("acetate", "acetate-contact", "acetate-separated")
    build_root = tmp_path / "01-build"
    for system_id in system_ids:
        system_dir = build_root / "systems" / system_id
        system_dir.mkdir(parents=True)
        for filename in (
            "topology.top",
            "coordinates.gro",
            "em.mdp",
            "npt.mdp",
            "production.mdp",
            "index.ndx",
            "production.gro",
            "production.xtc",
        ):
            (system_dir / filename).write_text("staged fixture\n")
        reference_dir = system_dir / "reference"
        reference_dir.mkdir()
        (reference_dir / "topology.top").write_text("staged fixture\n")
        (reference_dir / "coordinates.gro").write_text("staged fixture\n")
        write_build_system_metadata(
            build_root,
            system_id,
            BuildSystemMetadata(
                system_name=system_id,
                charge=0,
                multiplicity=1,
                box=(2.0, 2.0, 2.0, 90.0, 90.0, 90.0),
                maxwarn=0,
                production_steps=100,
            ),
        )

        trajectory_dir = (
            tmp_path / "02-reference-md" / "trajectories" / system_id
        )
        trajectory_dir.mkdir(parents=True)
        (trajectory_dir / "trajectory.xtc").write_text("staged fixture\n")

    for filename in ("01-build-colvars.yaml", "01-build-plumed.yaml"):
        config_path = build_root / "config.yaml"
        shutil.copy2(acetate / "configs" / filename, config_path)
        BuildConfig.load(config_path)

    label_root = tmp_path / "02-reference-snapshots"
    label_root.mkdir()
    for filename in (
        "02-reference-snapshots-local.yaml",
        "02-reference-snapshots-slurm.yaml",
    ):
        config_path = label_root / "config.yaml"
        shutil.copy2(acetate / "configs" / filename, config_path)
        LabelSnapshotsConfig.load(config_path)

    sample_root = tmp_path / "03-sample"
    sample_root.mkdir()
    for filename in ("03-sample-local.yaml", "03-sample-slurm.yaml"):
        config_path = sample_root / "config.yaml"
        shutil.copy2(acetate / "configs" / filename, config_path)
        SampleParametersConfig.load(config_path)
    (sample_root / "samples.yaml").write_text("staged fixture\n")
    (sample_root / "specs.yaml").write_text("staged fixture\n")

    qoi_root = tmp_path / "04-qoi"
    qoi_root.mkdir()
    qoi_config = qoi_root / "config.yaml"
    shutil.copy2(
        acetate / "configs" / "04-build-qoi-datasets.yaml",
        qoi_config,
    )
    BuildQoIDatasetsConfig.load(qoi_config)
    qoi_data = qoi_root / "qoi"
    qoi_data.mkdir()
    for name in ("rdf", "hb", "contact-distance", "separated-distance"):
        (qoi_data / f"{name}.pt").write_text("staged fixture\n")

    fit_root = tmp_path / "05-lgp"
    fit_root.mkdir()
    fit_config = fit_root / "config.yaml"
    shutil.copy2(acetate / "configs" / "05-fit-lgp.yaml", fit_config)
    FitLGPConfig.load(fit_config)
    models = fit_root / "models"
    models.mkdir()
    for name in ("rdf", "hb", "contact-distance", "separated-distance"):
        (models / f"{name}.lgp").write_text("staged fixture\n")

    learn_root = tmp_path / "06-learn"
    learn_root.mkdir()
    learn_config = learn_root / "config.yaml"
    shutil.copy2(acetate / "configs" / "06-learn.yaml", learn_config)
    LearnConfig.load(learn_config)
    learn_outputs = learn_root / "outputs"
    learn_outputs.mkdir()
    (learn_outputs / "posterior.pt").write_text("staged fixture\n")

    validate_root = tmp_path / "07-validate"
    validate_root.mkdir()
    validate_config = validate_root / "config.yaml"
    shutil.copy2(acetate / "configs" / "07-validate.yaml", validate_config)
    ValidateConfig.load(validate_config)


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
