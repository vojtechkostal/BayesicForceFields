import json

from bff.qoi import QoI
from bff.workflows.build_qoi_datasets.main import _shared_block_metadata


def test_shared_qoi_metadata_does_not_contain_recursive_references() -> None:
    blocks = [
        QoI("rdf", [1.0], settings={"bins": 200}),
        QoI("rdf", [2.0], settings={"bins": 200}),
    ]

    settings, metadata = _shared_block_metadata(blocks)

    assert settings == {"bins": 200}
    assert metadata == {}
    json.dumps(metadata)
