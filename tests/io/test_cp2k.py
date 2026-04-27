from pathlib import Path

import pytest

from bff.io.cp2k import (
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    format_cp2k_input,
    get_cp2k_kind_defaults,
    get_cp2k_single_atom_directory_name,
    get_cp2k_single_atom_multiplicity,
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
