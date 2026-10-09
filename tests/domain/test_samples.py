from pathlib import Path

import numpy as np
import pytest

from bff.domain.samples import (
    SampleSet,
    load_sample_manifest,
    write_sample_manifest,
)


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _campaign(tmp_path: Path, samples: dict) -> Path:
    for system_id in ("a", "b"):
        _write(tmp_path / "systems" / system_id / "topology.top")
        _write(tmp_path / "systems" / system_id / "coordinates.gro")
    for record in samples.values():
        for roles in record.get("outputs", {}).values():
            for path in roles.values():
                _write(tmp_path / path)
    return write_sample_manifest(
        tmp_path / "samples.yaml",
        parameter_names=["sigma A"],
        systems={"a": 10, "b": 10},
        samples=samples,
        provenance={"source": "test"},
    )


def test_completed_samples_get_staged_and_own_inputs(tmp_path: Path) -> None:
    fn_manifest = _campaign(
        tmp_path,
        {
            "0": {
                "params": [0.4],
                "status": "completed",
                "outputs": {
                    "a": {
                        "topology": "s/0/a/topology.top",
                        "trajectory": "s/0/a/x.xtc",
                    },
                    "b": {"topology": "s/0/b/topology.top", "pmf": "s/0/b/x.pmf"},
                },
            },
            "1": {"params": [0.6], "status": "failed"},
        },
    )

    sample_set = SampleSet.from_manifest(fn_manifest)

    assert sample_set.parameter_names == ("sigma A",)
    assert sample_set.sample_ids == ["0"]
    np.testing.assert_allclose(sample_set.inputs, [[0.4]])
    (sample,) = sample_set.samples
    assert sample.inputs["a"]["topology"] == tmp_path / "s/0/a/topology.top"
    assert sample.inputs["a"]["coordinates"] == tmp_path / "systems/a/coordinates.gro"
    assert sample.inputs["b"]["pmf"] == tmp_path / "s/0/b/x.pmf"
    assert "trajectory" not in sample.inputs["b"]


def test_completed_sample_with_missing_file_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    outputs = {"a": {"trajectory": "s/0/a/x.xtc"}, "b": {}}
    fn_manifest = _campaign(
        tmp_path,
        {
            "0": {"params": [0.4], "status": "completed", "outputs": outputs},
            "1": {"params": [0.5], "status": "completed", "outputs": outputs},
        },
    )
    (tmp_path / "s/0/a/x.xtc").unlink()

    sample_set = SampleSet.from_manifest(fn_manifest)

    assert sample_set.sample_ids == []
    assert set(sample_set.unusable) == {"0", "1"}
    assert "x.xtc" in sample_set.unusable["0"]


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"params": [0.4, 0.5], "status": "completed"}, "one value per"),
        ({"params": [0.4], "status": "done"}, "status must be one of"),
        ({"params": [0.4], "status": "failed", "outputs": {"c": {}}}, "unknown"),
    ],
)
def test_manifest_records_are_checked(tmp_path: Path, record, message) -> None:
    fn_manifest = write_sample_manifest(
        tmp_path / "samples.yaml",
        parameter_names=["sigma A"],
        systems={"a": 10},
        samples={"0": record},
    )
    with pytest.raises(ValueError, match=message):
        load_sample_manifest(fn_manifest)
