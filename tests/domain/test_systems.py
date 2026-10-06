from pathlib import Path

import yaml

from bff.domain.systems import (
    BuildSystemMetadata,
    load_build_system_metadata,
    write_build_system_metadata,
)


def test_system_metadata_round_trips_without_paths_or_versions(tmp_path: Path) -> None:
    build_path = write_build_system_metadata(
        tmp_path,
        "acetate",
        BuildSystemMetadata(
            system_name="Aqueous acetate",
            charge=-1,
            multiplicity=1,
            box=(10.0, 11.0, 12.0, 90.0, 90.0, 90.0),
            maxwarn=0,
            production_steps=1000,
        ),
    )
    raw = yaml.safe_load(build_path.read_text())
    assert not ({"schema_version", "version", "paths", "inputs"} & set(raw))
    loaded = load_build_system_metadata(tmp_path, "acetate")
    assert loaded.system_name == "Aqueous acetate"
    assert loaded.box[:3] == (10.0, 11.0, 12.0)
