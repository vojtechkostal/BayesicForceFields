from pathlib import Path

import pytest
import yaml

from bff.domain.samples import SampleSet


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _sample_campaign(tmp_path: Path, outputs: list[dict]) -> Path:
    for system_id in ("a", "b"):
        _write(tmp_path / "systems" / system_id / "topology.top")
        _write(tmp_path / "systems" / system_id / "coordinates.gro")
    for output in outputs:
        _write(tmp_path / output["trajectory"])
    (tmp_path / "specs.yaml").write_text(
        yaml.safe_dump({"bounds": {"sigma A": [0.1, 1.0]}, "charge_constraints": []})
    )
    (tmp_path / "samples.yaml").write_text(
        yaml.safe_dump(
            {
                "systems": [
                    {
                        "system_id": system_id,
                        "topology": f"systems/{system_id}/topology.top",
                        "coordinates": f"systems/{system_id}/coordinates.gro",
                    }
                    for system_id in ("a", "b")
                ],
                "samples": {
                    "sample-0": {
                        "params": [0.4],
                        "status": "completed",
                        "outputs": outputs,
                    }
                },
            }
        )
    )
    return tmp_path


def test_sample_outputs_are_paired_by_system_id_not_position(tmp_path: Path) -> None:
    campaign = _sample_campaign(
        tmp_path,
        [
            {"system_id": "b", "trajectory": "samples/sample-0/b/traj.xtc"},
            {"system_id": "a", "trajectory": "samples/sample-0/a/traj.xtc"},
        ],
    )
    samples = SampleSet.from_dir(campaign)
    sample = samples.samples[0]
    assert sample.system_ids == ("a", "b")
    assert [path.parts[-2] for path in sample.trajectory_paths] == ["a", "b"]


@pytest.mark.parametrize(
    "outputs, expected",
    [
        (
            [
                {"system_id": "a", "trajectory": "samples/s/a/one.xtc"},
                {"system_id": "a", "trajectory": "samples/s/a/two.xtc"},
            ],
            "duplicate a",
        ),
        (
            [{"system_id": "a", "trajectory": "samples/s/a/one.xtc"}],
            "missing b",
        ),
    ],
)
def test_sample_manifest_rejects_duplicate_or_missing_system_outputs(
    tmp_path: Path,
    outputs: list[dict],
    expected: str,
) -> None:
    campaign = _sample_campaign(tmp_path, outputs)
    with pytest.raises(ValueError, match=expected):
        SampleSet.from_dir(campaign)
