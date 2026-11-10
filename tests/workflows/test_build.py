from pathlib import Path
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest
from gmxtopology import Topology

import bff.workflows.build.main as build_main
from bff.io.logs import Logger
from bff.workflows.build.box import load_residue_template
from bff.workflows.build.main import (
    equilibrate,
    write_coordinates,
    write_reference_system,
)


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
    assert not reference_topology.moleculetype("SOL").virtual_sites3
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


def test_load_residue_template_accepts_arbitrary_monoatomic_residue() -> None:
    residue = SimpleNamespace(
        name="SOD",
        atoms=[SimpleNamespace(name="SOD", mass=22.98977)],
    )

    template = load_residue_template(residue, {})

    assert template.positions.tolist() == [[0.0, 0.0, 0.0]]
    assert template.real_mask.tolist() == [True]


def test_load_residue_template_accepts_common_water_residue_alias() -> None:
    residue = SimpleNamespace(
        name="TIP3",
        atoms=[
            SimpleNamespace(name="OH2", mass=15.9994),
            SimpleNamespace(name="H1", mass=1.008),
            SimpleNamespace(name="H2", mass=1.008),
        ],
    )

    template = load_residue_template(residue, {})

    assert template.positions.shape == (3, 3)
    assert template.real_mask.tolist() == [True, True, True]


def test_build_averages_the_npt_box_and_rescales_the_last_frame(
    tmp_path: Path, monkeypatch
) -> None:
    topology, coordinates = _write_two_waters(tmp_path, virtual_site=False)
    template = mda.Universe(topology, coordinates, topology_format="ITP")
    water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.927, 0.0]])
    edges = [19.0, 20.0, 21.0, 22.0, 23.0]  # angstrom; the first is discarded

    def fake_run_md(deffnm, **kwargs):
        if not str(deffnm).endswith("-npt"):
            return
        with mda.Writer(str(deffnm) + ".xtc", n_atoms=6) as writer:
            for edge in edges:
                # Water 2 is split across the boundary, as GROMACS writes it.
                positions = np.vstack(
                    [water + 0.25 * edge, water + [edge - 0.3, 0.5 * edge, 0.5 * edge]]
                )
                template.atoms.positions = np.mod(positions, edge)
                template.dimensions = [edge, edge, edge, 90.0, 90.0, 90.0]
                writer.write(template.atoms)

    monkeypatch.setattr(
        build_main, "create_box", lambda *args, **kwargs: (None, Topology(topology))
    )
    monkeypatch.setattr(build_main, "run_md", fake_run_md)
    system = SimpleNamespace(
        topology_path=topology,
        templates={},
        box=None,
        mdp_em_path=tmp_path / "em.mdp",
        mdp_npt_path=tmp_path / "npt.mdp",
        nsteps_npt=100,
    )

    state = equilibrate(
        system,
        label="topology-000",
        equilibration_dir=tmp_path,
        gmx_cmd="gmx",
        fn_gmx_log=tmp_path / "gromacs.log",
        logger=Logger("build", verbose=False),
    )

    assert state.box.tolist() == pytest.approx([21.5, 21.5, 21.5, 90.0, 90.0, 90.0])

    last = state.universe.trajectory[-1]
    molecules_last = [
        molecule.positions.copy() for molecule in state.universe.atoms.split("molecule")
    ]
    assert last.dimensions[0] == pytest.approx(23.0)
    fn_coord = tmp_path / "coordinates.gro"
    write_coordinates(state, fn_coord)

    written = mda.Universe(topology, fn_coord, topology_format="ITP")
    assert written.dimensions.tolist() == pytest.approx(state.box.tolist(), abs=1e-3)
    scale = 21.5 / 23.0
    for before, molecule in zip(molecules_last, written.atoms.split("molecule")):
        after = molecule.positions
        # The molecule moves with the box but keeps its geometry (gro: 0.01 A).
        masses = molecule.masses[:, None]
        com_before = (before * masses).sum(axis=0) / masses.sum()
        assert molecule.center_of_mass() == pytest.approx(scale * com_before, abs=0.02)
        assert after - after[0] == pytest.approx(before - before[0], abs=0.02)
