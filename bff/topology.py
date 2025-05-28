"""Module to handle force field parameteres."""

import warnings
import numpy as np
import MDAnalysis as mda
import parmed as pmd
from MDAnalysis.guesser.tables import masses as MDA_MASSES
from MDAnalysis.lib.distances import distance_array

from .tools import random_placement, guess_box
from .data import WATER_3SITE, WATER_4SITE, WATERS, IONS

# Suppress specific warning from MDAnalysis
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="MDAnalysis.core.universe"
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="MDAnalysis.topology.ITPParser"
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

    # NOTE Guess elements manually as it has a problem with 'HAL3' atomtype
    # recognizing as AL instead of H
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
    topol = pmd.load_file(fn_topol)
    universe = fill_universe(topol)

    if box is None:
        box = guess_box(len(topol.residues))
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
    topol: pmd.gromacs.GromacsTopologyFile,
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


def fill_universe(topol: pmd.gromacs.GromacsTopologyFile) -> mda.Universe:
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
    universe.add_TopologyAttr('type', [a.type for a in atoms])
    universe.add_TopologyAttr('resname', [r.name for r in residues])
    universe.add_TopologyAttr('resid', list(range(1, len(residues) + 1)))

    universe.guess_TopologyAttrs(to_guess=['elements', 'masses'])

    return universe


class TopologyParser:
    """
    A class to parse and manipulate Gromacs topology files.

    Attributes
    ----------
    fn_topol : str
        Path to the GROMACS topology file.
    topol : GromacsTopologyFile
        The parsed topology data.
    mol_resname : str
        Name of the selected molecule in the topology.
    implicit_atomtype : str, optional
        Atom type whose charge is implicitly defined.
        by the total charge requirement.
    mol : Optional[str]
        The selected molecule.
    mol_atomtypes : List[str]
        Unique atom types in the selected molecule.
    mol_atomtype_counts : Dict[str, int]
        Number of atoms of each type in the selected molecule.
    implicit_charge : Optional[float]
        Charge of the implicit atomtype.
    implicit_param : str
        The implicit parameter name.
    avail_mol_params : List[str]
        Adjustable parameters for the selected molecule.

    Methods
    -------
    parse
        Parse the topology from an .itp file.
    write
        Write the topology into a file.
    select_mol
        Select a molecule from the topology by name.
    update_params
        Update the parameters of the selected molecule.
    """

    def __init__(self, fn_topol: str) -> None:
        """
        Initialize the TopologyParser with a given topology file path.

        Parameters
        ----------
        fn_topol : str
            Path to the GROMACS topology (.itp) file.
        """

        self.fn_topol = str(fn_topol)
        self.topol = None
        self.universe = None
        self.mol_resname = None
        self.implicit_atomtype = None

        self.parse()

    def parse(self):
        """Parse the topology from an .itp file."""
        self.topol = pmd.load_file(self.fn_topol)
        self.universe = prepare_universe(self.fn_topol)

    def write(self, fn_out: str):
        """Write the topology into file."""
        self.topol.save(fn_out)

    def select_mol(self, mol_resname: str, implicit_atomtype: str):
        """
        Select a molecule from the topology by name.

        Parameters
        ----------
        mol_resname : str
            Residue name of the molecule to select.
        implicit_atomtype : str
            Atom type whose charge will be adjusted implicitly
            to satisfy the total charge requirement.
        """
        self.mol_resname = mol_resname
        self.implicit_atomtype = implicit_atomtype

    @property
    def mol(self) -> str:
        """Retrieve the selected molecule."""
        mol_list = self.topol.molecules.get(self.mol_resname, None)
        if mol_list is None:
            raise ValueError(
                f"Molecule with resname '{self.mol_resname}' not found in the "
                "topology."
            )
        return mol_list[0]

    @property
    def n_mol(self) -> int:
        """Retrieve the number of molecules in the topology."""
        return sum(r.name == self.mol_resname for r in self.topol.residues)

    @property
    def mol_atomtypes(self) -> list:
        """Retrieve unique atom types in the selected molecule."""
        atomtypes = [atom.type for atom in self.mol.atoms]
        atomtypes = np.unique(atomtypes)
        return atomtypes

    @property
    def mol_atomtype_counts(self) -> dict:
        """Count the number of atoms of each type in the selected molecule."""
        atomtypes = [atom.type for atom in self.mol.atoms]
        unique_atomtypes, counts = np.unique(atomtypes, return_counts=True)
        return dict(zip(unique_atomtypes.tolist(), counts.tolist()))

    @property
    def implicit_charge(self) -> float:
        """Retrieve charge of the implicit atomtype."""
        q = [atom.charge
             for atom in self.mol.atoms
             if atom.type == self.implicit_atomtype]
        if not q:
            return None
        return q[0]

    @property
    def implicit_param(self) -> str:
        """Retrieve the implicit parameter name."""
        return f'q {self.implicit_atomtype}'

    @property
    def avail_mol_params(self) -> list:
        """Retrieve the adjustable parameters for the selected molecule."""
        kinds = ['charge', 'sigma', 'epsilon']
        params = []
        for kind in kinds:
            for atomtype in self.mol_atomtypes:
                params.append(f'{kind} {atomtype}')
        return params

    def _modify_charge(self, atomtype: str, value: float) -> None:
        """Modify the charge of a given atom type."""
        for atom in self.topol.atoms:
            if atomtype == atom.type:
                atom.charge = value

    def _modify_sigma(self, atomtype: str, value: float) -> None:
        """Modify the Lennard-Jones sigma of a given atom type."""
        for atom in self.topol.atoms:
            if atomtype == atom.type:
                atom.atom_type.sigma = value * 10

    def _modify_epsilon(self, atomtype: str, value: float) -> None:
        """Modify the Lennard-Jones epsilon of a given atom type."""
        for atom in self.topol.atoms:
            if atomtype == atom.type:
                atom.atom_type.epsilon = value / 4.184

    def _constraint_charge(self, total_charge: float) -> None:
        """Adjust the implicit atomtype charge
        to satisfy the total charge requirement."""

        assert self.n_mol != 0, "No molecule selected."
        n_implicit = self.mol_atomtype_counts[self.implicit_atomtype]
        q_explicit = sum(
            atom.charge for atom in self.topol.atoms
            if (atom.type in self.mol_atomtypes and
                atom.type != self.implicit_atomtype)
        )
        q_implicit = (total_charge - q_explicit / self.n_mol) / n_implicit
        self._modify_charge(self.implicit_atomtype, q_implicit)

    def update_params(self, params: dict, total_charge: float = None) -> None:
        """Update the parameters of the selected molecule.

        Parameters
        ----------
        params : dict
            A dictionary of parameters and the values to update.
        total_charge : float, optional
            The total charge of the system.
        """
        for param, value in params.items():
            param_type, atomtype = param.split()
            if param_type == 'charge':
                self._modify_charge(atomtype, value)
            elif param_type == 'sigma':
                self._modify_sigma(atomtype, value)
            elif param_type == 'epsilon':
                self._modify_epsilon(atomtype, value)

        if total_charge is not None:
            self._constraint_charge(total_charge)
