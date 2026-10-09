import MDAnalysis as mda
import numpy as np
import pytest


def solvated_universe(n_water: int = 300, n_solute: int = 8, n_frames: int = 4):
    """Deterministic random solute-water system with O-H and N-H donors.

    Solute residues ``LIG`` contain OA-HO, NA-HN, and CA atoms; waters ``SOL``
    contain OW-HW. Hydrogens sit 1 Angstrom from their heavy atom.
    """
    rng = np.random.default_rng(7)
    solute = [
        ("OA", "O", 16.0),
        ("HO", "H", 1.0),
        ("NA", "N", 14.0),
        ("HN", "H", 1.0),
        ("CA", "C", 12.0),
    ]
    water = [("OW", "O", 16.0), ("HW", "H", 1.0), ("HW", "H", 1.0)]
    atoms, resindex, resnames, bonds = [], [], [], []
    for residue in range(n_solute):
        first = len(atoms)
        atoms += solute
        resindex += [residue] * len(solute)
        resnames.append("LIG")
        bonds += [(first, first + 1), (first + 2, first + 3), (first, first + 4)]
    for residue in range(n_solute, n_solute + n_water):
        first = len(atoms)
        atoms += water
        resindex += [residue] * len(water)
        resnames.append("SOL")
        bonds += [(first, first + 1), (first, first + 2)]

    universe = mda.Universe.empty(
        len(atoms),
        n_residues=n_solute + n_water,
        atom_resindex=resindex,
        trajectory=True,
    )
    universe.add_TopologyAttr("types", [atom[0] for atom in atoms])
    universe.add_TopologyAttr("elements", [atom[1] for atom in atoms])
    universe.add_TopologyAttr("masses", [atom[2] for atom in atoms])
    universe.add_TopologyAttr("resnames", resnames)
    universe.add_TopologyAttr("bonds", bonds)

    box = 20.0
    positions = rng.uniform(0.0, box, size=(n_frames, len(atoms), 3))
    for heavy, hydrogen in bonds:
        if atoms[hydrogen][1] == "H":
            direction = rng.normal(size=(n_frames, 3))
            positions[:, hydrogen] = positions[:, heavy] + (
                direction / np.linalg.norm(direction, axis=1)[:, None]
            )
    dimensions = np.tile([box, box, box, 90.0, 90.0, 90.0], (n_frames, 1))
    dimensions[:, :3] += rng.uniform(-0.5, 0.5, size=(n_frames, 3))
    universe.load_new(
        positions.astype(np.float32), format="MEMORY", dimensions=dimensions
    )
    return universe


@pytest.fixture
def solvated() -> mda.Universe:
    return solvated_universe()
