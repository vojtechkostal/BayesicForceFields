import re
from pathlib import Path

import yaml

import bff

ROOT = Path(__file__).parents[1]


def test_release_metadata_is_synchronized() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    version = match.group(1)

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    release_date = citation["date-released"]

    assert version == "0.4.1"
    assert bff.__version__ == version
    assert citation["version"] == version
    assert release_date == "2026-08-25"
    assert (
        f"## `{version}` - {release_date}"
        in (ROOT / "CHANGELOG.md").read_text()
    )
