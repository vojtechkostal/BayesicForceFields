"""Built-in solute-water hydrogen-bond routine."""

from __future__ import annotations

from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import calc_angles, capped_distance

from .dataset import QoI
from .trajectory import get_unitcell, select_atoms

OPTIONS = {
    "selection",
    "water_selection",
    "elements",
    "donor_acceptor_cutoff",
    "angle_cutoff",
    "pbc",
    "update_selections",
}


def _site_keys(atoms: mda.AtomGroup) -> tuple[np.ndarray, np.ndarray]:
    """Return unique ``resname(type)`` keys and each atom's index into them."""
    keys = [
        f"{resname}({atom_type})"
        for resname, atom_type in zip(atoms.resnames, atoms.types)
    ]
    return np.unique(np.asarray(keys, dtype=str), return_inverse=True)


def hydrogen_bonds(
    universe: mda.Universe,
    *,
    frames: slice,
    options: dict[str, Any],
) -> QoI:
    """Average number of hydrogen bonds per frame between a solute and water.

    Options: ``selection`` (solute, required), ``water_selection`` (default
    ``"resname SOL HOH WAT"``), ``elements`` (heavy-atom sites, default
    ``[O, N, S]``), ``donor_acceptor_cutoff`` (3.5 Angstrom), ``angle_cutoff``
    (150 degrees), ``pbc`` (true), and ``update_selections`` (false).

    Donor hydrogens come from topology bonds. Both solute-to-water and
    water-to-solute bonds are counted, one value per
    ``"donor_resname(type) to acceptor_resname(type)"`` label.
    """
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(
            f"unsupported hydrogen-bond option(s): {', '.join(sorted(unknown))}."
        )
    if "selection" not in options:
        raise ValueError("Hydrogen bonds require the 'selection' option.")
    selection = options["selection"]
    water_selection = options.get("water_selection", "resname SOL HOH WAT")
    elements = options.get("elements", ("O", "N", "S"))
    distance_cutoff = options.get("donor_acceptor_cutoff", 3.5)
    angle_cutoff = options.get("angle_cutoff", 150.0)
    pbc = options.get("pbc", True)
    update_selections = options.get("update_selections", False)
    if not isinstance(pbc, bool) or not isinstance(update_selections, bool):
        raise ValueError(
            "Hydrogen-bond options pbc and update_selections must be booleans."
        )
    if (
        not isinstance(distance_cutoff, (int, float))
        or isinstance(distance_cutoff, bool)
        or distance_cutoff <= 0
    ):
        raise ValueError(
            f"donor_acceptor_cutoff must be positive, got {distance_cutoff!r}."
        )
    if (
        not isinstance(angle_cutoff, (int, float))
        or isinstance(angle_cutoff, bool)
        or not 0 < angle_cutoff <= 180
    ):
        raise ValueError(f"angle_cutoff must be in (0, 180], got {angle_cutoff!r}.")
    if not (
        isinstance(elements, (list, tuple))
        and elements
        and all(isinstance(element, str) and element for element in elements)
    ):
        raise ValueError("Hydrogen-bond elements must be a non-empty string list.")
    elements = tuple(sorted({element.capitalize() for element in elements}))
    if "H" in elements:
        raise ValueError("Hydrogen-bond elements must be heavy atoms, not H.")

    solute = select_atoms(
        universe, selection, updating=update_selections, field="selection"
    )
    water = select_atoms(
        universe, water_selection, updating=update_selections, field="water_selection"
    )
    try:
        bonds = universe.bonds.indices
        is_hydrogen = universe.atoms.elements == "H"
    except NoDataError as exc:
        raise ValueError(
            "Hydrogen-bond analysis requires topology bond and element information."
        ) from exc
    # Every (heavy atom, bonded hydrogen) pair of the topology.
    donor_hydrogen = np.concatenate(
        [bonds[is_hydrogen[bonds[:, 1]]], bonds[is_hydrogen[bonds[:, 0]]][:, ::-1]]
    )
    angle_threshold = np.deg2rad(angle_cutoff)

    possible_labels: set[str] = set()
    counts: dict[str, int] = {}
    directions: list[dict[str, Any]] = []
    n_frames = 0
    for ts in universe.trajectory[frames]:
        box = get_unitcell(universe, ts) if pbc else None

        if n_frames == 0 or update_selections:
            for field, text, group in (
                ("selection", selection, solute),
                ("water_selection", water_selection, water),
            ):
                if len(group) == 0:
                    raise ValueError(
                        f"Hydrogen-bond {field} became empty at frame {ts.frame}: "
                        f"{text!r}."
                    )
            solute_sites = solute[np.isin(solute.elements, elements)]
            water_sites = water[np.isin(water.elements, elements)]
            for field, text, sites in (
                ("selection", selection, solute_sites),
                ("water_selection", water_selection, water_sites),
            ):
                if len(sites) == 0:
                    raise ValueError(
                        f"Hydrogen-bond {field} contains no {list(elements)} "
                        f"sites: {text!r}."
                    )

            directions = []
            for donor_sites, acceptors in (
                (solute_sites, water_sites),
                (water_sites, solute_sites),
            ):
                pairs = donor_hydrogen[
                    np.isin(donor_hydrogen[:, 0], donor_sites.indices)
                ]
                donors = universe.atoms[pairs[:, 0]]
                donor_keys, donor_codes = _site_keys(donors)
                acceptor_keys, acceptor_codes = _site_keys(acceptors)
                pair_labels = [
                    f"{donor_key} to {acceptor_key}"
                    for donor_key in donor_keys
                    for acceptor_key in acceptor_keys
                ]
                # A label is possible unless its only donor/acceptor is one atom.
                donor_atoms = [
                    set(donors.indices[donor_codes == i])
                    for i in range(len(donor_keys))
                ]
                acceptor_atoms = [
                    set(acceptors.indices[acceptor_codes == j])
                    for j in range(len(acceptor_keys))
                ]
                for code, label in enumerate(pair_labels):
                    i, j = divmod(code, len(acceptor_keys))
                    if len(donor_atoms[i]) > 1 or donor_atoms[i] != acceptor_atoms[j]:
                        possible_labels.add(label)
                directions.append(
                    {
                        "donors": donors,
                        "hydrogens": universe.atoms[pairs[:, 1]],
                        "acceptors": acceptors,
                        "donor_codes": donor_codes * len(acceptor_keys),
                        "acceptor_codes": acceptor_codes,
                        "pair_labels": pair_labels,
                    }
                )

        for direction in directions:
            donors = direction["donors"]
            acceptors = direction["acceptors"]
            if len(donors) == 0:
                continue
            pairs = capped_distance(
                donors.positions,
                acceptors.positions,
                max_cutoff=float(distance_cutoff),
                box=box,
                return_distances=False,
            )
            pairs = pairs[donors.indices[pairs[:, 0]] != acceptors.indices[pairs[:, 1]]]
            if len(pairs) == 0:
                continue
            angles = calc_angles(
                donors.positions[pairs[:, 0]],
                direction["hydrogens"].positions[pairs[:, 0]],
                acceptors.positions[pairs[:, 1]],
                box=box,
            )
            pairs = pairs[angles >= angle_threshold]
            label_counts = np.bincount(
                direction["donor_codes"][pairs[:, 0]]
                + direction["acceptor_codes"][pairs[:, 1]],
                minlength=len(direction["pair_labels"]),
            )
            for code in np.flatnonzero(label_counts):
                label = direction["pair_labels"][code]
                counts[label] = counts.get(label, 0) + int(label_counts[code])
        n_frames += 1

    if n_frames == 0:
        raise ValueError("Hydrogen-bond frame slice selects no trajectory frames.")
    labels = tuple(sorted(possible_labels))
    if not labels:
        raise ValueError(
            "Hydrogen-bond selections contain no non-self donor/acceptor pairs."
        )
    return QoI(
        name="hydrogen_bonds",
        values=np.asarray([counts.get(label, 0) / n_frames for label in labels]),
        labels=labels,
        values_per_label=1,
        settings={
            "selection": selection,
            "water_selection": water_selection,
            "elements": elements,
            "donor_acceptor_cutoff": float(distance_cutoff),
            "angle_cutoff": float(angle_cutoff),
            "pbc": pbc,
            "update_selections": update_selections,
        },
    )
