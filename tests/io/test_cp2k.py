from pathlib import Path

import pytest

from bff.io.cp2k import (
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    read_cp2k_energy,
    read_cp2k_forces,
)


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
