from pathlib import Path

import MDAnalysis as mda
from gmxtopology import Topology

from bff.workflows._shared.preparation import write_reference_system


def _write_two_waters(tmp_path: Path, *, virtual_site: bool) -> tuple[Path, Path]:
    topology = tmp_path / "system.top"
    virtual_atom = "4 MW 1 SOL MW 1 0.0 0.0\n" if virtual_site else ""
    virtual_section = (
        "[ virtual_sites3 ]\n4 1 2 3 1 0.128 0.128\n" if virtual_site else ""
    )
    n_atoms_per_water = 4 if virtual_site else 3
    topology.write_text(
        "[ defaults ]\n1 2 yes 1.0 1.0\n"
        "[ atomtypes ]\n"
        "OW 8 15.999 0.0 A 0.3 0.5\n"
        "HW 1 1.008 0.0 A 0.0 0.0\n"
        "MW 0 0.0 0.0 D 0.0 0.0\n"
        "[ moleculetype ]\nSOL 2\n"
        "[ atoms ]\n"
        "1 OW 1 SOL OW 1 0.0 15.999\n"
        "2 HW 1 SOL HW1 1 0.5 1.008\n"
        "3 HW 1 SOL HW2 1 0.5 1.008\n"
        + virtual_atom
        + "[ settles ]\n1 1 0.09572 0.15139\n"
        + virtual_section
        + "[ system ]\nwater\n"
        "[ molecules ]\nSOL 2\n"
    )

    coordinates = tmp_path / "system.gro"
    names = ["OW", "HW1", "HW2"] + (["MW"] if virtual_site else [])
    lines = ["two waters", str(2 * n_atoms_per_water)]
    atom_index = 1
    for residue_index in (1, 2):
        for name in names:
            lines.append(
                f"{residue_index:5d}{'SOL':<5}{name:>5}{atom_index:5d}"
                f"{atom_index / 100:8.3f}{0.0:8.3f}{0.0:8.3f}"
            )
            atom_index += 1
    lines.append("   2.00000   2.00000   2.00000")
    coordinates.write_text("\n".join(lines) + "\n")
    return topology, coordinates


def test_write_reference_system_removes_declared_virtual_sites(tmp_path: Path) -> None:
    topology, coordinates = _write_two_waters(tmp_path, virtual_site=True)
    output_dir = tmp_path / "reference"

    removed = write_reference_system(
        topology,
        coordinates,
        output_dir / "topology.top",
        output_dir / "coordinates.gro",
    )

    assert removed == 2
    reference_topology = Topology(output_dir / "topology.top")
    assert len(reference_topology.atoms) == 6
    assert not reference_topology.molecules["SOL"][0].virtual_sites3
    reference = mda.Universe(
        output_dir / "topology.top",
        output_dir / "coordinates.gro",
        topology_format="ITP",
    )
    assert reference.atoms.names.tolist() == ["OW", "HW1", "HW2"] * 2


def test_write_reference_system_keeps_stable_paths_without_vsites(
    tmp_path: Path,
) -> None:
    topology, coordinates = _write_two_waters(tmp_path, virtual_site=False)
    output_dir = tmp_path / "reference"

    removed = write_reference_system(
        topology,
        coordinates,
        output_dir / "topology.top",
        output_dir / "coordinates.gro",
    )

    assert removed == 0
    assert (output_dir / "topology.top").is_file()
    assert (output_dir / "coordinates.gro").is_file()
    reference = mda.Universe(
        output_dir / "topology.top",
        output_dir / "coordinates.gro",
        topology_format="ITP",
    )
    assert len(reference.atoms) == 6
