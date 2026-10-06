"""Fill a simulation box with molecules from a GROMACS topology."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from gmxtopology import Topology
from MDAnalysis.lib.distances import distance_array
from scipy.constants import atomic_mass
from scipy.spatial.transform import Rotation as R

MAX_INSERTION_ATTEMPTS = 100_000

WATER_4SITE = np.array(
    [
        [0.02909557, 0.03692916, -0.04588176],
        [0.43909732, -0.54306981, 0.60411833],
        [-0.90090283, -0.04307076, 0.12411808],
        [-0.0309039, -0.04307076, 0.05411814],
    ]
)

WATER_3SITE = np.array(
    [
        [0.02909557, 0.03692916, -0.04588176],
        [0.43909732, -0.54306981, 0.60411833],
        [-0.90090283, -0.04307076, 0.12411808],
    ]
)

WATERS = {
    "H2O",
    "HHO",
    "HOH",
    "OH2",
    "OHH",
    "SOL",
    "SPC",
    "SPCE",
    "T3P",
    "T4P",
    "TIP",
    "TIP2",
    "TIP3",
    "TIP4",
    "TP3M",
    "WAT",
    "WATER",
    "water",
}


@dataclass(frozen=True, slots=True)
class ResidueTemplate:
    positions: np.ndarray
    real_mask: np.ndarray


def random_placement(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Randomly place a molecule within a box."""
    displacement = np.random.rand(3) * box
    rotation = R.random().as_matrix()
    coords = coords @ rotation.T
    coords += displacement
    return coords


def guess_box(n_mol: int) -> np.ndarray:
    """Approximates a cubic box size based on density of neat water."""
    mass = float(n_mol) * 18.015 * atomic_mass  # kg
    density = 1000  # kg/m^3
    length = np.cbrt(mass / density) * 1e10  # Angstroms
    return np.array([length] * 3 + [90, 90, 90])


def create_box(
    fn_topol: str,
    templates: dict[str, str | Path],
    fn_out: str,
    box: np.ndarray = None,
    min_dist: float = 1.5,
) -> tuple:
    """Fill a box from topology molecule counts and residue templates."""

    topol = Topology(fn_topol)
    universe = fill_universe(topol)

    if box is None:
        n_heavy = sum(a.mass > 1.1 for a in topol.atoms)
        box = guess_box(n_heavy)
    else:
        box = np.array(box)
    universe.dimensions = box

    residue_templates = build_residue_templates(topol, templates)
    coords = insert_molecules(
        topol,
        residue_templates,
        box,
        min_dist=min_dist,
    )
    universe.atoms.positions = coords

    universe.atoms.write(fn_out)

    return universe, topol


def build_residue_templates(
    topol: Topology,
    templates: dict[str, str | Path],
) -> dict[tuple[str, int], ResidueTemplate]:
    """Load one centered placement template for each residue layout."""
    residue_templates: dict[tuple[str, int], ResidueTemplate] = {}
    for residue in topol.residues:
        key = (residue.name, len(residue.atoms))
        if key in residue_templates:
            continue
        residue_templates[key] = load_residue_template(residue, templates)
    return residue_templates


@lru_cache(maxsize=None)
def _load_positions_from_file(fn_template: str) -> tuple[np.ndarray, tuple[str, ...]]:
    universe = mda.Universe(fn_template)
    return (
        np.asarray(universe.atoms.positions, dtype=float).copy(),
        tuple(str(name) for name in universe.atoms.names),
    )


@lru_cache(maxsize=None)
def _load_water_template(n_atoms: int) -> tuple[np.ndarray, tuple[str, ...]]:
    if n_atoms == 3:
        return WATER_3SITE.copy(), ("OW", "HW1", "HW2")
    if n_atoms == 4:
        return WATER_4SITE.copy(), ("OW", "HW1", "HW2", "IW")
    raise ValueError(f"Unsupported water template with {n_atoms} atoms.")


def load_residue_template(
    residue,
    templates: dict[str, str | Path],
) -> ResidueTemplate:
    """Resolve one residue placement template from builtins or user input."""
    residue_names = tuple(atom.name for atom in residue.atoms)
    n_atoms = len(residue.atoms)
    if residue.name.upper() in WATERS:
        positions, _ = _load_water_template(n_atoms)
        template_names = residue_names
    elif n_atoms == 1:
        positions = np.zeros((1, 3), dtype=float)
        template_names = residue_names
    else:
        if residue.name not in templates:
            raise ValueError(
                f"Missing template for non-standard residue {residue.name!r}."
            )
        positions, template_names = _load_positions_from_file(
            str(Path(templates[residue.name]).resolve())
        )

    if len(template_names) != n_atoms:
        raise ValueError(
            f"Template for residue {residue.name!r} has {len(template_names)} atoms, "
            f"expected {n_atoms}."
        )
    if residue_names != template_names:
        raise ValueError(
            f"Template atom names for residue {residue.name!r} do not match the "
            f"topology order: expected {residue_names}, got {template_names}."
        )

    positions = np.asarray(positions, dtype=float)
    real_mask = np.asarray([atom.mass > 0.5 for atom in residue.atoms], dtype=bool)
    anchor = positions[real_mask].mean(axis=0) if np.any(real_mask) else positions[0]
    return ResidueTemplate(
        positions=positions - anchor,
        real_mask=real_mask,
    )


def _neighboring_cells(
    cell: tuple[int, int, int],
    shape: np.ndarray,
) -> list[tuple[int, int, int]]:
    neighbors: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbors.append(
                    (
                        (cell[0] + dx) % int(shape[0]),
                        (cell[1] + dy) % int(shape[1]),
                        (cell[2] + dz) % int(shape[2]),
                    )
                )
    return neighbors


def _cell_index(
    position: np.ndarray,
    cell_size: np.ndarray,
    shape: np.ndarray,
) -> tuple[int, int, int]:
    wrapped = np.mod(position, cell_size * shape)
    index = np.floor(wrapped / cell_size).astype(int) % shape.astype(int)
    return int(index[0]), int(index[1]), int(index[2])


def insert_molecules(
    topol: Topology,
    templates: dict[tuple[str, int], ResidueTemplate],
    box: np.ndarray,
    min_dist: float = 1.5,
) -> np.ndarray:
    """Insert residue templates into the box with a simple spatial grid."""

    coords = np.zeros((len(topol.atoms), 3))
    occupied: list[np.ndarray] = []
    cell_shape = np.maximum(1, np.floor(box[:3] / min_dist).astype(int))
    cell_size = box[:3] / cell_shape
    cells: dict[tuple[int, int, int], list[int]] = {}
    atom_index = 0

    for residue in topol.residues:
        n_atoms = len(residue.atoms)
        template = templates[(residue.name, n_atoms)]
        displacement_limit = box[:3]
        for _ in range(MAX_INSERTION_ATTEMPTS):
            pos_trial = random_placement(template.positions.copy(), displacement_limit)
            pos_trial = np.mod(pos_trial, box[:3])
            trial_real = pos_trial[template.real_mask]
            neighbor_ids: set[int] = set()
            for point in trial_real:
                cell = _cell_index(point, cell_size, cell_shape)
                for neighbor in _neighboring_cells(cell, cell_shape):
                    neighbor_ids.update(cells.get(neighbor, []))
            if neighbor_ids:
                existing = np.asarray([occupied[i] for i in sorted(neighbor_ids)])
                distances = distance_array(trial_real, existing, box=box)
                if np.any(distances < min_dist):
                    continue

            coords[atom_index : atom_index + n_atoms] = pos_trial
            for point in trial_real:
                occupied.append(point.copy())
                cell = _cell_index(point, cell_size, cell_shape)
                cells.setdefault(cell, []).append(len(occupied) - 1)
            break
        else:
            raise RuntimeError(
                f"Could not insert residue {residue.name!r} into box "
                f"{box[:3].tolist()} after {MAX_INSERTION_ATTEMPTS} attempts; "
                "enlarge the box or reduce the molecule counts."
            )
        atom_index += n_atoms

    return coords


def fill_universe(topol: Topology) -> mda.Universe:
    """Fill an empty universe with the topology information."""
    atoms = topol.atoms
    residues = topol.residues
    resindices = np.array([i for i, r in enumerate(residues) for _ in r.atoms])
    segindices = [0] * len(topol.residues)

    # Create universe
    universe = mda.Universe.empty(
        n_atoms=len(atoms),
        n_residues=len(residues),
        atom_resindex=resindices,
        residue_segindex=segindices,
        trajectory=True,
    )

    universe.add_TopologyAttr("name", [a.name for a in atoms])
    universe.add_TopologyAttr("type", [a.type.name for a in atoms])
    universe.add_TopologyAttr("resname", [r.name for r in residues])
    universe.add_TopologyAttr("resid", list(range(1, len(residues) + 1)))

    universe.guess_TopologyAttrs(to_guess=["elements", "masses"])

    return universe
