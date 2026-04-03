import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from gmxtopology import MoleculeType, Topology
from MDAnalysis.guesser.tables import masses as MDA_MASSES
from MDAnalysis.lib.distances import distance_array

from .data import IONS, WATER_3SITE, WATER_4SITE, WATERS
from .tools import guess_box, random_placement

# 1) MDAnalysis DeprecationWarning from ITPParser (elements guessing transition)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"MDAnalysis\.topology\.ITPParser",
)

# 2) MDAnalysis UserWarning about missing coordinate reader (topology-only file)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"No coordinate reader found for .*\.top\. Skipping this file\.",
    module=r"MDAnalysis\.core\.universe",
)

MASSES = np.array(list(MDA_MASSES.values()))
ELEMENTS = list(MDA_MASSES.keys())


@dataclass(frozen=True, slots=True)
class ResidueTemplate:
    positions: np.ndarray
    real_mask: np.ndarray


def guess_elements(u: mda.Universe) -> list:
    elements = [
        ELEMENTS[np.argmin(np.abs(MASSES - a.mass))]
        for a in u.atoms if np.isclose(MASSES, a.mass, atol=1e-2).any()
    ]
    u.add_TopologyAttr('elements', elements)


def check_unitcell(universe: mda.Universe, unitcell: list) -> None:
    """Ensure that the universe has a unitcell defined."""
    if not universe.dimensions:
        if not unitcell:
            raise ValueError("Unitcell is not specified.")
        for ts in universe.trajectory:
            ts.dimensions = unitcell


def check_topols(fn_topol_1: str, fn_topol_2: str) -> None:
    """Check if a pair of topologies is consistent."""

    # Load universes
    u1 = prepare_universe(fn_topol_1)
    u2 = prepare_universe(fn_topol_2)

    # Do not consider dummy atoms
    atoms1 = u1.select_atoms('not mass -1 to 0.5')
    atoms2 = u2.select_atoms('not mass -1 to 0.5')

    # Check if the atomtypes are the same
    at1 = np.unique(atoms1.types)
    at2 = np.unique(atoms2.types)

    if not len(at1) == len(at2):
        raise ValueError(
            (
                f'Number of atomtypes do not match between {fn_topol_1} '
                f'and {fn_topol_2}.'
            )
        )

    if not np.all(at1 == at2):
        raise ValueError(
            f'Atomtypes do not match between {fn_topol_1} and {fn_topol_2}.'
        )

    # Check if the atom names are the same
    names1 = np.unique(atoms1.names)
    names2 = np.unique(atoms2.names)
    if not np.all(names1 == names2):
        raise ValueError(
            f'Atom names do not match between {fn_topol_1} and {fn_topol_2}.'
        )


def prepare_universe(
        fn_topol: str, fn_coord: str = None, dt: float = 1.0) -> mda.Universe:
    """
    Prepares an MDAnalysis universe
    from Gromacs .itp topology and coordinates.

    Parameters
    ----------
    fn_topol : str
        Path to the topology file.
    fn_coord : str
        Path to the coordinate file.
    dt : float, optional
        Time step in ps. Default is 1.0.

    Returns
    -------
    universe : MDAnalysis.Universe
        A universe object with the topology and coordinates loaded,
        and guessed elements.
    """

    if fn_coord is None:
        universe = mda.Universe(
            fn_topol,
            topology_format='ITP', to_guess=('elements', 'masses'), dt=dt
        )
    else:
        universe = mda.Universe(
            fn_topol, fn_coord,
            topology_format='ITP', to_guess=('elements', 'masses'), dt=dt
        )

    guess_elements(universe)

    return universe


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
    key = residue.name.lower()

    if residue.name in WATERS:
        positions, template_names = _load_water_template(n_atoms)
    elif key in IONS:
        if n_atoms != 1:
            raise ValueError(
                f"Ion residue {residue.name!r} is expected to be monoatomic, "
                f"but has {n_atoms} atoms."
            )
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
        while True:
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

            coords[atom_index:atom_index + n_atoms] = pos_trial
            for point in trial_real:
                occupied.append(point.copy())
                cell = _cell_index(point, cell_size, cell_shape)
                cells.setdefault(cell, []).append(len(occupied) - 1)
            break
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
        trajectory=True)

    universe.add_TopologyAttr('name', [a.name for a in atoms])
    universe.add_TopologyAttr('type', [a.type.name for a in atoms])
    universe.add_TopologyAttr('resname', [r.name for r in residues])
    universe.add_TopologyAttr('resid', list(range(1, len(residues) + 1)))

    universe.guess_TopologyAttrs(to_guess=['elements', 'masses'])

    return universe


class TopologyModifier(Topology if Topology is not None else object):

    """A class for modifying Gromacs topologies.

    Parameters
    ----------
    fn_topol : str | Path
        Path to the topology file to be modified.
    mol_resname : str
        Residue name of the molecule to be modified.
    implicit_atomnames : list[str] | str
        Atom names or types to be treated as implicit
        (i.e., their charges will be adjusted to satisfy a charge constraint).
        Can be a single string if only one atom is implicit.

    Properties
    ----------
    mol : MoleculeType
        The molecule type corresponding to `mol_resname`.
    n_mol : int
        The number of molecules of type `mol_resname` in the topology.
    implicit_atoms : list[Atom]
        List of Atom objects corresponding to the implicit atoms.
    implicit_param : str
        A string representing the charge parameter for the implicit atoms.
    total_charge : float
        The total charge of the molecule before applying any constraints.

    Methods
    -------
    group_charge(atomnames: list[str]) -> float
        Calculate the total charge of a group of atoms specified by their names.
    resolve_parameter_names(params)
        Resolve charge-parameter keys that may contain atom types into names.
    apply_parameters(params, constraint_charge=None)
        Update topology parameters and apply an optional charge constraint.
    """

    def __init__(
        self,
        fn_topol: Path | str,
        mol_resname: str,
        implicit_atomnames: list[str] | str
    ) -> None:

        self.source = fn_topol
        self.mol_resname = mol_resname
        if isinstance(implicit_atomnames, str):
            self._implicit_atomnames = [implicit_atomnames]
        else:
            self._implicit_atomnames = implicit_atomnames

        super().__init__(self.source)
        self.universe = prepare_universe(self.source)

    @property
    def mol(self) -> MoleculeType:
        return self.molecules[self.mol_resname][0]

    @property
    def n_mol(self) -> int:
        return self.molecules[self.mol_resname][1]

    @property
    def implicit_atoms(self) -> list[str]:
        implicit_atoms = []
        for atom in self.mol.atoms:
            if (atom.name in self._implicit_atomnames
                    or atom.type.name in self._implicit_atomnames):
                implicit_atoms.append(atom)

        return implicit_atoms

    @property
    def implicit_param(self) -> str:
        atoms = " ".join([atom.name for atom in self.implicit_atoms])
        return f"charge {atoms}"

    @property
    def total_charge(self) -> float:
        return np.sum([atom.charge for atom in self.mol.atoms])

    def _update_charge(self, atomname: str, value: float) -> None:
        for atom in self.mol.atoms:
            if atom.name == atomname:
                atom.update(charge=value)
                return
        raise ValueError(f"Atom {atomname} not found in molecule {self.mol_resname}.")

    def _update_sigma(self, atomtype: str, value: float) -> None:
        for at in self.atomtypes:
            if at.name == atomtype:
                at.update(sigma=value)
                return
        raise ValueError(f"Atom type {atomtype} not found in topology.")

    def _update_epsilon(self, atomtype: str, value: float) -> None:
        for at in self.atomtypes:
            if at.name == atomtype:
                at.update(epsilon=value)
                return
        raise ValueError(f"Atom type {atomtype} not found in topology.")

    def _update_dihedraltype9(
        self,
        atoms: str,
        k: float,
        phase: float,
        multiplicity: int
    ) -> None:

        # normalize white spaces
        dihedraltype = " ".join(atoms.split())

        for d in self.mol.dihedrals:
            atoms = [d.ai, d.aj, d.ak, d.al]
            if d.func == 9:
                dt_str = " ".join([a.type.name for a in atoms])
            if dt_str == dihedraltype:
                d.update(kphi=k, phi_s=phase, mult=multiplicity)
                return
        raise ValueError(
            f"Dihedral type '{dihedraltype}' not found in molecule {self.mol_resname}."
        )

    def _update_define(self, directive: str, argument: float | int | str) -> None:
        for define in self.defines:
            if define.directive == directive:
                define.update(argument=argument)
                return

    def _constraint_charge(self, target: float) -> None:

        q_explicit = np.sum([
            atom.charge for atom in self.mol.atoms
            if atom not in self.implicit_atoms
        ])

        n_implicit = len(self.implicit_atoms)
        q_implicit = (target - q_explicit) / n_implicit
        for atom in self.implicit_atoms:
            self._update_charge(atom.name, q_implicit)

    def group_charge(self, atomnames: list[str]) -> float:
        return np.sum([
            atom.charge for atom in self.mol.atoms
            if atom.name in atomnames
        ])

    def resolve_parameter_names(
        self,
        params: dict[str, float | list[float]]
    ) -> dict[str, float | list[float]]:
        mol_atomnames = [atom.name for atom in self.mol.atoms]
        mol_atomtypes = [atom.type.name for atom in self.mol.atoms]

        resolved: dict[str, float | list[float]] = {}
        for p in params.keys():
            p_name, atoms = p.split(" ", maxsplit=1)
            if p_name != 'charge':
                resolved[p] = params[p]
                continue

            atomnames = " "
            for name in atoms.split(" "):
                if name in mol_atomnames:
                    atomnames += f"{name} "
                elif name in mol_atomtypes:
                    for atom in self.mol.atoms:
                        if atom.type.name == name:
                            atomnames += f"{atom.name} "
                else:
                    raise ValueError(
                        f"Atom name or type '{name}' not found "
                        f"in molecule {self.mol_resname}."
                    )
            resolved[f"charge {atomnames.strip()}"] = params[p]

        # check for duplicates
        flat: list[str] = []
        for r in resolved.keys():
            if r.startswith('charge'):
                flat.extend(r.split(" ")[1:])
        if len(flat) != len(set(flat)):
            raise ValueError("Duplicate atom names in resolved parameters.")

        return resolved

    def apply_parameters(
        self,
        params: dict[str, float | list[float]],
        constraint_charge: float | None = None,
    ) -> None:
        params_resolved = self.resolve_parameter_names(params)

        for p, value in params_resolved.items():
            if "dihedraltype9" in p:
                p_name, *dihedraltypes, mult, phase = p.split()
                for dt in dihedraltypes:
                    atoms = dt.replace("_", " ")
                    self._update_dihedraltype9(
                        atoms,
                        k=value,
                        phase=float(phase),
                        multiplicity=int(mult)
                    )

            elif p.startswith("define"):
                p_name, directive = p.split(" ")
                self._update_define(directive, value)

            else:
                p_name, *atoms = p.split(" ")
                if p_name == 'charge':
                    for atom in atoms:
                        self._update_charge(atom, value)
                elif p_name == 'sigma':
                    for atom in atoms:
                        self._update_sigma(atom, value)
                elif p_name == 'epsilon':
                    for atom in atoms:
                        self._update_epsilon(atom, value)
                else:
                    raise ValueError(f"Unsupported parameter name '{p_name}'.")

        if constraint_charge is not None:
            # Ensure the implicit charge group stays implicit.
            for atom in self.implicit_atoms:
                updated = any(
                    key.startswith("charge ")
                    and atom.name in key.split()[1:]
                    for key in params_resolved
                )
                if updated:
                    raise ValueError(
                        f"Implicit atom '{atom.name}' charge was set manually, "
                        "cannot apply constraint charge."
                    )
            self._constraint_charge(constraint_charge)
