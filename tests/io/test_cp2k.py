from pathlib import Path

import pytest

from _fake_mdanalysis import install as install_fake_mdanalysis

from bff.io.cp2k import (
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    format_cp2k_input,
    get_cp2k_kind_defaults,
    get_cp2k_single_atom_directory_name,
    get_cp2k_single_atom_multiplicity,
    make_cp2k_isolated_atom_input,
    read_cp2k_energy,
    read_cp2k_forces,
)


def test_format_cp2k_input_formats_nested_sections() -> None:
    lines = format_cp2k_input(
        {
            "global": {"project_name": "job", "run_type": "ENERGY"},
            "force_eval": {"dft": {"basis_set_file_name": ["A", "B"]}},
            "motion": {"print": False},
        }
    )

    assert lines == [
        "&GLOBAL",
        "  PROJECT_NAME job",
        "  RUN_TYPE ENERGY",
        "&END GLOBAL",
        "&FORCE_EVAL",
        "  &DFT",
        "    BASIS_SET_FILE_NAME A",
        "    BASIS_SET_FILE_NAME B",
        "  &END DFT",
        "&END FORCE_EVAL",
        "&MOTION",
        "  PRINT .FALSE.",
        "&END MOTION",
    ]


def test_cp2k_lookup_helpers_report_known_and_unknown_elements() -> None:
    assert get_cp2k_kind_defaults("Ca")["potential"] == "GTH-PBE-q10"
    assert get_cp2k_single_atom_directory_name("Cl") == "chlorine"
    assert get_cp2k_single_atom_multiplicity("O") == 3

    with pytest.raises(ValueError, match="defaults"):
        get_cp2k_kind_defaults("Xe")
    with pytest.raises(ValueError, match="directory"):
        get_cp2k_single_atom_directory_name("Xe")
    with pytest.raises(ValueError, match="multiplicity"):
        get_cp2k_single_atom_multiplicity("Xe")


@pytest.mark.parametrize(
    ("element", "symbol", "multiplicity", "basis_set"),
    [
        ("o", "O", 3, "TZV2P-GTH-q6"),
        ("k", "K", 2, "TZV2P-MOLOPT-SR-GTH-q9"),
    ],
)
def test_make_cp2k_isolated_atom_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    element: str,
    symbol: str,
    multiplicity: int,
    basis_set: str,
) -> None:
    fn_input = tmp_path / "input.inp"
    install_fake_mdanalysis(monkeypatch)

    make_cp2k_isolated_atom_input(element, fn_input)

    text = fn_input.read_text()
    xyz = (tmp_path / "pos.xyz").read_text().splitlines()

    assert xyz == [
        "1",
        f"{symbol} isolated in vacuum",
        f"{symbol} 5.000000 5.000000 5.000000",
    ]
    assert f"PROJECT_NAME {symbol}_vacuum" in text
    assert "RUN_TYPE ENERGY" in text
    assert "PREFERRED_FFT_LIBRARY FFTW" in text
    assert "SCF_GUESS ATOMIC" in text
    assert f"MULTIPLICITY {multiplicity}" in text
    assert "UKS .TRUE." in text
    assert "COORD_FILE_NAME pos.xyz" in text
    assert "ABC [angstrom] 10.0 10.0 10.0" in text
    assert "PERIODIC NONE" in text
    assert "PSOLVER MT" in text
    assert f"BASIS_SET {basis_set}" in text
    assert "&MOTION" not in text
    assert "VDW_POTENTIAL" not in text


def test_read_cp2k_energy_converts_hartree_to_ev(tmp_path: Path) -> None:
    output = tmp_path / "cp2k.out"
    output.write_text(" ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -0.500000\n")

    assert read_cp2k_energy(output) == pytest.approx(-0.5 * HARTREE_TO_EV)

    output.write_text("no energy\n")
    with pytest.raises(ValueError, match="Could not extract"):
        read_cp2k_energy(output)


def test_read_cp2k_forces_parses_pipe_format(tmp_path: Path) -> None:
    output = tmp_path / "cp2k.out"
    output.write_text(
        " FORCES| 1 0.100000 0.200000 -0.300000 0.000000\n"
    )

    forces = read_cp2k_forces(output)

    assert forces == [
        [
            pytest.approx(0.1 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM),
            pytest.approx(0.2 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM),
            pytest.approx(-0.3 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM),
        ]
    ]


def test_read_cp2k_forces_parses_last_legacy_table(tmp_path: Path) -> None:
    output = tmp_path / "cp2k.out"
    output.write_text(
        " ATOMIC FORCES in [a.u.]\n"
        " 1 1 H 0.0 0.0 0.1\n"
        " SUM OF ATOMIC FORCES\n"
        " ATOMIC FORCES in [a.u.]\n"
        " 1 1 H 0.0 0.0 0.2\n"
        " SUM OF ATOMIC FORCES\n"
    )

    forces = read_cp2k_forces(output)

    assert forces[0][2] == pytest.approx(
        0.2 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
    )
