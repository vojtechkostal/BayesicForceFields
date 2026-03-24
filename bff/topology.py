import numpy as np
import MDAnalysis as mda
from MDAnalysis.guesser.tables import masses as MDA_MASSES
from MDAnalysis.lib.distances import distance_array

from pathlib import Path
from gmxtopology import Topology, MoleculeType

from .data import WATER_3SITE, WATER_4SITE, IONS, WATERS
from .tools import random_placement, guess_box

import warnings


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
    fn_mol: str,
    fn_out: str,
    box: np.ndarray = None,
    min_dist: float = 1.5,
    disp_limit: float = 0.4
) -> tuple:
    """Fills a box with molecules, solvent and ions."""

    # Create universe
    topol = Topology(fn_topol)
    universe = fill_universe(topol)

    if box is None:
        n_heavy = sum(a.mass > 1.1 for a in topol.atoms)
        box = guess_box(n_heavy)
    else:
        box = np.array(box)
    universe.dimensions = box

    # Insert moleculal positions
    mol = mda.Universe(fn_mol)  # TODO: get rid of the elements warning
    pos_mol = mol.atoms.positions - mol.atoms.center_of_mass()
    coords = insert_molecules(
        universe, topol, box, pos_mol,
        min_dist=min_dist, disp_limit=disp_limit
    )
    universe.atoms.positions = coords

    # Write positons into a file
    universe.atoms.write(fn_out)

    return universe, topol


def insert_molecules(
    universe: mda.Universe,
    topol: Topology,
    box: np.ndarray,
    pos_mol: np.ndarray,
    min_dist: float = 1.5,
    disp_limit: float = 0.4
) -> np.ndarray:
    """Insert molecules into the box."""

    coords = np.zeros((len(topol.atoms), 3))
    i = 0
    for residue in topol.residues:
        n_atoms = len(residue.atoms)
        if residue.name in WATERS:
            pos = WATER_3SITE if n_atoms == 3 else WATER_4SITE
            displacement_limit = box[:3]
        elif residue.name.lower() in IONS:
            pos = np.zeros(3)
            displacement_limit = box[:3] * disp_limit
        else:
            pos = pos_mol
            displacement_limit = box[:3] * disp_limit
        while True:
            pos_trial = random_placement(pos.copy(), displacement_limit)
            distances = distance_array(
                pos_trial, coords, box=universe.dimensions
            )
            if not np.any(distances < min_dist):
                coords[i:i+n_atoms] = pos_trial
                break
        i += n_atoms

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
