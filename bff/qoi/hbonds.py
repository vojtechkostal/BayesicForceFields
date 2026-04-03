from typing import Dict

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import calc_angles, capped_distance

from ..tools import get_unitcell
from .data import QoI

__all__ = ["compute_all_hbonds"]


def count_hbonds(
    universe: mda.Universe,
    donor_indices: np.ndarray,
    acceptor_indices: np.ndarray,
    n_frames: int
) -> Dict[str, float]:
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

    if n_frames <= 0 or donor_indices.size == 0 or acceptor_indices.size == 0:
        return {}

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
        results[name] = count / n_frames

    return results


def _paired_donors_and_hydrogens(
    universe: mda.Universe,
    selection: str,
    *,
    hb_elements: set[str],
) -> tuple[mda.AtomGroup, mda.AtomGroup]:
    """Return donor and hydrogen groups with one-to-one indexing."""
    hydrogens_all = universe.select_atoms(f"{selection} and element H")
    donor_indices: list[int] = []
    hydrogen_indices: list[int] = []

    for hydrogen in hydrogens_all:
        if not hydrogen.bonded_atoms:
            continue
        donor = hydrogen.bonded_atoms[0]
        if donor.element not in hb_elements:
            continue
        donor_indices.append(donor.index)
        hydrogen_indices.append(hydrogen.index)

    if not donor_indices:
        empty = universe.atoms[np.asarray([], dtype=int)]
        return empty, empty

    donors = universe.atoms[np.asarray(donor_indices, dtype=int)]
    hydrogens = universe.atoms[np.asarray(hydrogen_indices, dtype=int)]
    return donors, hydrogens


def _all_possible_hbond_labels(
    donors: mda.AtomGroup,
    acceptors: mda.AtomGroup,
) -> set[str]:
    """Enumerate all donor/acceptor type combinations possible for this system."""
    if len(donors) == 0 or len(acceptors) == 0:
        return set()

    donor_types = np.unique(
        np.column_stack((donors.resnames, donors.types)).astype(str),
        axis=0,
    )
    acceptor_types = np.unique(
        np.column_stack((acceptors.resnames, acceptors.types)).astype(str),
        axis=0,
    )

    return {
        f"{d_res}({d_type}) to {a_res}({a_type})"
        for d_res, d_type in donor_types
        for a_res, a_type in acceptor_types
    }


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
) -> tuple[np.ndarray, np.ndarray, int]:
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
    n_frames = 0
    for ts in universe.trajectory[sl]:
        n_frames += 1
        box = get_unitcell(universe, ts)
        da_indices = capped_distance(
            donors,
            acceptors,
            max_cutoff=distance_cutoff,
            min_cutoff=1.0,
            box=box,
            return_distances=False,
        )
        if da_indices.size == 0:
            continue

        d = donors[da_indices[:, 0]]
        h = hydrogens[da_indices[:, 0]]
        a = acceptors[da_indices[:, 1]]

        dha_angles = calc_angles(d, h, a, box=box)
        hbond_indices = np.where(dha_angles > angle_cutoff)[0]
        if hbond_indices.size == 0:
            continue

        donor_indices.extend(d[hbond_indices].indices)
        acceptor_indices.extend(a[hbond_indices].indices)

    return (
        np.asarray(donor_indices, dtype=int),
        np.asarray(acceptor_indices, dtype=int),
        n_frames,
    )


def compute_all_hbonds(
    universe: mda.Universe,
    *,
    mol_resname: str,
    water_resname: str = "SOL",
    distance_cutoff: float = 3.5,
    angle_cutoff: float = 150,
    hb_elements: set[str] = {"O", "N", "S"},
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> QoI:
    """Compute all solute-water hydrogen-bond QoIs for one trajectory."""
    selection_1 = f"resname {mol_resname}"
    selection_2 = f"resname {water_resname}"

    hb_elements = set(hb_elements)
    hb_elements_str = " ".join(sorted(hb_elements))

    selection_pairs = ((selection_1, selection_2), (selection_2, selection_1))
    possible_labels: set[str] = set()
    hbonds: dict[str, float] = {}
    for sel_donors, sel_acceptors in selection_pairs:
        donors, hydrogens = _paired_donors_and_hydrogens(
            universe,
            sel_donors,
            hb_elements=hb_elements,
        )
        acceptors = universe.select_atoms(
            f"{sel_acceptors} and element {hb_elements_str}"
        )

        possible_labels.update(_all_possible_hbond_labels(donors, acceptors))

        if len(donors) == 0 or len(acceptors) == 0:
            continue

        donor_indices, acceptor_indices, n_frames = compute_hbonds(
            universe,
            hydrogens,
            donors,
            acceptors,
            distance_cutoff=distance_cutoff,
            angle_cutoff=angle_cutoff,
            start=start,
            stop=stop,
            step=step,
        )
        hbonds.update(count_hbonds(universe, donor_indices, acceptor_indices, n_frames))

    labels = tuple(sorted(possible_labels))
    values = np.asarray([hbonds.get(label, 0.0) for label in labels], dtype=float)
    metadata = {
        "mol_resname": mol_resname,
        "water_resname": water_resname,
        "distance_cutoff": float(distance_cutoff),
        "angle_cutoff": float(angle_cutoff),
    }
    return QoI(
        name="hb",
        values=values,
        labels=labels,
        values_per_label=1,
        settings=metadata,
    )
