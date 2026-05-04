from pathlib import Path

import pytest

from bff.io.cp2k import (
    HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
    HARTREE_TO_EV,
    write_cp2k_snapshot_extxyz,
)
from bff.io.extxyz import read_extxyz_frame


def _write_xyz(path: Path, atom: str, values: tuple[float, float, float]) -> None:
    path.write_text(
        "1\n"
        "comment\n"
        f"{atom} {values[0]} {values[1]} {values[2]}\n"
    )


def test_write_cp2k_snapshot_extxyz_converts_cp2k_units(tmp_path: Path) -> None:
    run_dir = tmp_path / "snapshot-0001"
    run_dir.mkdir()

    _write_xyz(run_dir / "pos.xyz", "H", (0.0, 0.0, 0.0))
    _write_xyz(run_dir / "md-pos-1.xyz", "H", (0.1, 0.2, 0.3))
    (run_dir / "sp.out").write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -1.500000\n"
        " FORCES| 1 1.000000 2.000000 3.000000 0.000000\n"
    )

    fn_extxyz = write_cp2k_snapshot_extxyz(run_dir)
    frame = read_extxyz_frame(fn_extxyz)

    assert frame["source"] == "sp"
    assert frame["atoms"] == ["H"]
    assert frame["positions"][0] == pytest.approx([0.1, 0.2, 0.3])
    assert frame["energy"] == pytest.approx(-1.5 * HARTREE_TO_EV)
    assert frame["forces"][0] == pytest.approx(
        [
            1.0 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            2.0 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            3.0 * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
        ]
    )
