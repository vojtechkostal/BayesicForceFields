import numpy as np
import MDAnalysis as mda
import inspect
from scipy.constants import atomic_mass
from scipy.spatial.transform import Rotation as R

from MDAnalysis.guesser.tables import masses as MDA_MASSES
from MDAnalysis.lib.distances import distance_array
from scipy.constants import atomic_mass

from gmxtop import Topology
from .data import WATER_3SITE, WATER_4SITE, IONS, WATERS


def modify_dcd_frc_header(fn):
    """
    Modify the header of a DCD file to include the 'CORD' flag.

    Parameters
    ----------
    fn : str
        Path to the DCD file to be modified.

    Notes
    -----
    This function opens the file in binary mode and writes the 'CORD' flag
    at the appropriate position in the header.
    """
    flag = 'CORD'.encode()
    with open(fn, 'rb+') as f:
        f.seek(4)
        f.write(flag)


def compute_distances(
        universe, ag1, ag2, start=None, stop=None, step=None, pbc=True):
    """Compute distances between two AtomGroups over a trajectory."""

    start = start or 0
    stop = stop or len(universe.trajectory)
    step = step or 1
    displacements = [
        ag1.positions[:, np.newaxis] - ag2.positions
        for ts in universe.trajectory[start:stop:step]
    ]
    displacements = np.array(displacements)
    if pbc:
        box = universe.dimensions[:3]
        displacements -= np.round(displacements / box) * box
    distances = np.linalg.norm(displacements, axis=-1)
    return distances


def random_placement(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Randomly place a molecule within a box."""
    displacement = np.random.rand(3) * box
    rotation = R.random().as_matrix()
    coords = coords @ rotation.T
    coords += displacement
    return coords


def guess_box(n_mol: int):
    """Approximates a cubic box size based on density of neat water."""
    mass = float(n_mol) * 18.015 * atomic_mass  # kg
    density = 1000  # kg/m^3
    length = np.cbrt(mass / density) * 1e10  # Angstroms
    return np.array([length] * 3 + [90, 90, 90])


def sigmoid(x, x0=3, scale=5):
    """Smoothly transitions from 0 to 1 around x=3."""
    x = np.asarray(x)
    arg = (x - x0) * scale
    return 1 / (1 + np.exp(- arg))


def extract_defaults(fn):
    """
    Extract default values from the function signature.

    Parameters
    ----------
    fn : callable
        The function from which to extract default values.

    Returns
    -------
    dict
        A dictionary with parameter names as keys and their default values.
    """
    sig = inspect.signature(fn)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


MASSES = np.array(list(MDA_MASSES.values()))
ELEMENTS = list(MDA_MASSES.keys())


def guess_box(n_mol: int):
    """Approximates a cubic box size based on density of neat water."""
    mass = float(n_mol) * 18.015 * atomic_mass  # kg
    density = 1000  # kg/m^3
    length = np.cbrt(mass / density) * 1e10  # Angstroms
    return np.array([length] * 3 + [90, 90, 90])


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