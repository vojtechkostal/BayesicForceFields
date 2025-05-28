import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance, calc_angles

__all__ = ["compute_all_hbonds", "extract_hbonds"]


def count_hbonds(
    universe: mda.Universe,
    donor_indices: np.ndarray,
    acceptor_indices: np.ndarray,
    n_frames: int
) -> dict:
    """
    Counts the average number of hydrogen bonds per hydrogen bond type.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    donor_indices : np.ndarray
    acceptor_indices : np.ndarray
    n_frames : int
        Number of trajectory frames used for the analysis.
    threshold : float, optional
        Minimum frequency threshold for a hydrogen bond type. Default is 0.1.

    Returns
    -------
    results : dict
        Dictionary containing the hydrogen bond types and their frequencies.
    """

    # Identify unique hbonds and their frequencies
    hbonds = np.column_stack(
        (
            universe.atoms[donor_indices].resnames,
            universe.atoms[donor_indices].types,
            universe.atoms[acceptor_indices].resnames,
            universe.atoms[acceptor_indices].types,
        )
    ).astype("str")

    hbond_types, counts = np.unique(hbonds, return_counts=True, axis=0)

    # Filter and format results
    results = {}
    for hb, count in zip(hbond_types, counts):
        name = f"{hb[0]}({hb[1]}) to {hb[2]}({hb[3]})"
        # if count / n_frames > 0.1:
        results[name] = count / n_frames

    return results


def compute_hbonds(
    universe: mda.Universe,
    hydrogens: mda.AtomGroup,
    donors: mda.AtomGroup,
    acceptors: mda.AtomGroup,
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 150,
    start: int = 0,
    step: int = 1,
    stop: int = None,
) -> tuple:
    """
    Computes hydrogen bonds between donor and acceptor atoms.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    hydrogens : MDAnalysis.AtomGroup
    donors : MDAnalysis.AtomGroup
    acceptors : MDAnalysis.AtomGroup
    distance_cutoff : float, optional
        Maximum distance between donor and acceptor atoms. Default is 3.0A.
    angle_cutoff : float, optional
        Minimum angle between donor-hydrogen-acceptor atoms.
        Default is 150 degrees.
    start : int, optional
        Start frame for the analysis. Default is 0.
    step : int, optional
        Frame stride for the analysis. Default is 1.
    stop : int, optional
        End frame for the analysis. Default is None.

    Returns
    -------
    donor_indices : np.ndarray
    acceptor_indices : np.ndarray
    n_frames : int
    """

    sl = slice(start, stop or len(universe.trajectory), step)
    angle_cutoff = np.deg2rad(angle_cutoff)

    donor_indices, acceptor_indices = [], []
    for ts in universe.trajectory[sl]:
        # Compute donor-acceptor distances
        da_indices = capped_distance(
            donors,
            acceptors,
            max_cutoff=distance_cutoff,
            min_cutoff=1.0,
            box=universe.dimensions,
            return_distances=False,
        )

        d = donors[da_indices[:, 0]]
        h = hydrogens[da_indices[:, 0]]
        a = acceptors[da_indices[:, 1]]

        # Compute donor-hydrogen-acceptor angles
        dha_angles = calc_angles(d, h, a, box=universe.dimensions)

        # Select those that satisfity distance and angle criterion
        hbond_indices = np.where(dha_angles > angle_cutoff)[0]

        donor_indices.extend(d[hbond_indices].indices)
        acceptor_indices.extend(a[hbond_indices].indices)

    return (
        np.asarray(donor_indices, dtype=int),
        np.asarray(acceptor_indices, dtype=int),
        len(universe.trajectory[sl]),
    )


def compute_all_hbonds(
        universe: mda.Universe, mol_resname: str, **kwargs) -> dict:
    """
    Computes all hydrogen bonds between a molecule and water.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    mol_resname : str
        Residue name of the molecule.
    **kwargs : dict, optional
        Additional keyword arguments passed to the
        underlying hydrogen bond analysis function.
        - `distance_cutoff` (float): Maximum acceptable
        donor-acceptor distance in Angstroms.
        - `angle_cutoff` (float): Minimum acceptable
        donor-hydrogen-acceptor angle in degrees.
        - `start` (int): Start analysis from this frame.
        - `stop` (int): Stop analysis at this frame.
        - `step` (int): Frame stride for the analysis.

    Returns
    -------
    hbonds : dict
        Dictionary containing the hydrogen bond their counts.
    """

    hb_elements = {"O", "N", "S"}
    water_resnames = "SOL HOH WAT"
    selection_1 = f"resname {mol_resname}"
    selection_2 = f"resname {water_resnames}"

    selection_mesh = [[selection_1, selection_2], [selection_2, selection_1]]
    hbonds = {}
    for sel_donors, sel_acceptors in selection_mesh:
        hydrogens = universe.select_atoms(f"{sel_donors} and element H")
        donors = sum([
            h.bonded_atoms[0]
            for h in hydrogens
            if h.bonded_atoms and h.bonded_atoms[0].element in hb_elements
        ])

        if isinstance(donors, int):
            continue

        acceptors = universe.select_atoms(
            f"{sel_acceptors} and element O S N")

        donor_indices, acceptor_indices, n_frames = compute_hbonds(
            universe, hydrogens, donors, acceptors, **kwargs
        )

        counts = count_hbonds(
            universe, donor_indices, acceptor_indices, n_frames
        )
        hbonds.update(counts)

    return hbonds


def extract_hbonds(hbonds_true, hbonds_pred):
    """Gets the average number of hydrogen bonds
    with respect to the reference."""

    n = []
    for name in hbonds_true.keys():
        hbonds = hbonds_pred.get(name, 0.0)
        n.append(hbonds)
    return n
