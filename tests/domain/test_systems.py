from pathlib import Path

import pytest

from bff.domain.systems import resolve_explicit_inputs


def test_explicit_inputs_resolve_relative_paths_and_lists(tmp_path: Path) -> None:
    (tmp_path / "a.pmf").write_text("1\n")
    (tmp_path / "b.pmf").write_text("2\n")
    inputs = resolve_explicit_inputs(
        {"pmf": ["a.pmf", "b.pmf"], "unused": None},
        base_dir=tmp_path,
        system_id="acetate",
        field="inputs",
    )
    assert inputs.inputs["pmf"] == (tmp_path / "a.pmf", tmp_path / "b.pmf")
    assert inputs.inputs["unused"] is None


def test_explicit_inputs_name_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"inputs\.trajectory"):
        resolve_explicit_inputs(
            {"trajectory": "missing.xtc"},
            base_dir=tmp_path,
            system_id="acetate",
            field="inputs",
        )
