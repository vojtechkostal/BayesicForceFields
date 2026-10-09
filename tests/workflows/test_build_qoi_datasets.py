import json
from pathlib import Path

import numpy as np
import yaml

from bff.domain.samples import write_sample_manifest
from bff.qoi import QoI
from bff.qoi.dataset import QoIDataset
from bff.workflows.build_qoi_datasets.main import (
    _mismatch,
    _shared_block_metadata,
    main,
)


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_shared_qoi_metadata_does_not_contain_recursive_references() -> None:
    blocks = [
        QoI("rdf", [1.0], settings={"bins": 200}),
        QoI("rdf", [2.0], settings={"bins": 200}),
    ]

    settings, metadata = _shared_block_metadata(blocks)

    assert settings == {"bins": 200}
    assert metadata == {}
    json.dumps(metadata)


def _campaign(campaign: Path) -> Path:
    """Three samples whose topology files hold their 'charge'; sample 1's
    topology is corrupted and sample 2 lacks the stored PMF."""
    _write(campaign / "systems/a/topology.top")
    _write(campaign / "systems/a/coordinates.gro")
    samples = {}
    for sample_id, charge in (("0", "0.1"), ("1", "broken"), ("2", "0.3")):
        top = _write(campaign / f"samples/{sample_id}/a/topology.top", charge)
        outputs = {"topology": str(top.relative_to(campaign))}
        if sample_id != "2":
            pmf = _write(campaign / f"samples/{sample_id}/a/production.pmf", "1.0")
            outputs["pmf"] = str(pmf.relative_to(campaign))
        samples[sample_id] = {
            "params": [float(sample_id)],
            "status": "completed",
            "outputs": {"a": outputs},
        }
    return write_sample_manifest(
        campaign / "samples.yaml",
        parameter_names=["charge X"],
        systems={"a": 10},
        samples=samples,
    )


def test_samples_use_their_own_topology_and_bad_samples_are_skipped(
    tmp_path: Path,
) -> None:
    manifest = _campaign(tmp_path / "campaign")
    routine = _write(
        tmp_path / "charge.py",
        "from bff.qoi import QoI\n"
        "def total(*, inputs, options):\n"
        "    charge = float(inputs['topology'].read_text())\n"
        "    return QoI('charge', [charge * float(inputs['pmf'].read_text())])\n",
    )
    reference = {
        "topology": str(_write(tmp_path / "ref.top", "0.2")),
        "pmf": str(_write(tmp_path / "ref.pmf", "1.0")),
    }
    config = tmp_path / "qoi.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "training_samples": {
                    "manifest": str(manifest),
                    "systems": [{"system_id": "a"}],
                    "workers": 1,
                },
                "reference": {"systems": [{"system_id": "a", "inputs": reference}]},
                "routines": [
                    {
                        "name": "charge",
                        "callable": f"{routine}:total",
                        "systems": ["a"],
                        "inputs": ["topology", "pmf"],
                    }
                ],
            }
        )
    )

    main(config)

    dataset = QoIDataset.load(tmp_path / "qoi" / "charge.pt")
    assert dataset.sample_ids == ("0",)
    assert dataset.parameter_names == ("charge X",)
    np.testing.assert_allclose(dataset.y.ravel(), [0.1])
    log = (tmp_path / "build-qoi-datasets.log").read_text()
    assert "Sample 1 skipped: ValueError" in log
    assert "Sample 2 skipped: no {'a': ['pmf']} output" in log
    assert not (tmp_path / "qoi" / "raw.json").exists()


def test_sample_qoi_on_a_different_grid_is_a_mismatch() -> None:
    reference = QoI("pmf", [0.0, 1.0], settings={"distance_nm": [0.3, 0.4]})
    shifted = QoI("pmf", [0.0, 1.0], settings={"distance_nm": [0.35, 0.45]})
    same_grid = QoI("pmf", [2.0, 3.0], settings=reference.settings)

    assert _mismatch(reference, same_grid) is None
    assert "settings" in _mismatch(reference, shifted)
