"""Explicit-selection hydrogen-bond quantities of interest."""

from __future__ import annotations

from typing import Any, Mapping

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import calc_angles, capped_distance

from .data import QoI
from .rdf import _box_and_volume, _select_group

__all__ = ["compute_hydrogen_bond_qoi"]


def validate_hydrogen_bond_options(
    options: Mapping[str, Any],
    *,
    context: str = "Hydrogen-bond options",
) -> set[str]:
    """Validate hydrogen-bond options and return canonical heavy elements."""
    if not isinstance(options.get("pbc", True), bool) or not isinstance(
        options.get("update_selections", False), bool
    ):
        raise ValueError(f"{context}: boolean options must be true or false.")

    distance_cutoff = options.get("donor_acceptor_cutoff", 3.5)
    if (
        not isinstance(distance_cutoff, (int, float))
        or isinstance(distance_cutoff, bool)
        or distance_cutoff <= 0
    ):
        raise ValueError(
            f"{context}.donor_acceptor_cutoff must be positive, "
            f"got {distance_cutoff!r}."
        )
    angle_cutoff = options.get("angle_cutoff", 150.0)
    if (
        not isinstance(angle_cutoff, (int, float))
        or isinstance(angle_cutoff, bool)
        or not 0 < angle_cutoff <= 180
    ):
        raise ValueError(
            f"{context}.angle_cutoff must be in (0, 180], got {angle_cutoff!r}."
        )

    elements = options.get("elements", ("O", "N", "S"))
    if not (
        isinstance(elements, (list, tuple))
        and elements
        and all(isinstance(element, str) and element for element in elements)
    ):
        raise ValueError(f"{context}.elements must be a non-empty string list.")
    allowed_elements = {element.capitalize() for element in elements}
    if "H" in allowed_elements:
        raise ValueError(f"{context}.elements must contain heavy atoms, not H.")
    return allowed_elements


def _donor_hydrogen_pairs(
    universe: mda.Universe,
    sites: mda.AtomGroup,
    *,
    frame_index: int,
) -> tuple[mda.AtomGroup, mda.AtomGroup]:
    paired_donors: list[int] = []
    paired_hydrogens: list[int] = []
    try:
        for donor in sites:
            for bonded in donor.bonded_atoms:
                if bonded.element == "H":
                    paired_donors.append(donor.index)
                    paired_hydrogens.append(bonded.index)
    except NoDataError as exc:
        raise ValueError(
            "Hydrogen-bond analysis requires topology bond information; "
            f"none is available at frame {frame_index}."
        ) from exc
    if not paired_donors:
        empty = universe.atoms[np.asarray([], dtype=int)]
        return empty, empty
    return (
        universe.atoms[np.asarray(paired_donors, dtype=int)],
        universe.atoms[np.asarray(paired_hydrogens, dtype=int)],
    )


def _label(donor: Any, acceptor: Any) -> str:
    return (
        f"{donor.resname}({donor.type}) to "
        f"{acceptor.resname}({acceptor.type})"
    )


def compute_hydrogen_bond_qoi(
    universe: mda.Universe,
    *,
    selection: str,
    water_selection: str = "resname SOL HOH WAT",
    elements: tuple[str, ...] | list[str] = ("O", "N", "S"),
    donor_acceptor_cutoff: float = 3.5,
    angle_cutoff: float = 150.0,
    pbc: bool = True,
    update_selections: bool = False,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> QoI:
    """Count all solute-water hydrogen bonds for selected heavy-atom sites."""
    allowed_elements = validate_hydrogen_bond_options(
        {
            "elements": elements,
            "donor_acceptor_cutoff": donor_acceptor_cutoff,
            "angle_cutoff": angle_cutoff,
            "pbc": pbc,
            "update_selections": update_selections,
        }
    )

    solute_group = _select_group(
        universe,
        selection,
        updating=update_selections,
        field="selections.selection",
    )
    water_group = _select_group(
        universe,
        water_selection,
        updating=update_selections,
        field="selections.water_selection",
    )
    counts: dict[str, int] = {}
    possible_labels: set[str] = set()
    n_frames = 0
    angle_threshold = np.deg2rad(angle_cutoff)
    directions: list[tuple[mda.AtomGroup, mda.AtomGroup, mda.AtomGroup]] | None = (
        None
    )

    for frame_index, ts in enumerate(
        universe.trajectory[slice(start, stop, step)], start=start
    ):
        box = None
        if pbc:
            box, _ = _box_and_volume(
                ts, context=f"Hydrogen-bond frame {frame_index}"
            )

        if directions is None or update_selections:
            for field, selection_text, group in (
                ("selection", selection, solute_group),
                ("water_selection", water_selection, water_group),
            ):
                if len(group) == 0:
                    raise ValueError(
                        f"Hydrogen-bond selection {field} became empty at frame "
                        f"{frame_index}: {selection_text!r}."
                    )
            try:
                solute_sites = solute_group[
                    np.isin(solute_group.elements, tuple(allowed_elements))
                ]
                water_sites = water_group[
                    np.isin(water_group.elements, tuple(allowed_elements))
                ]
            except NoDataError as exc:
                raise ValueError(
                    "Hydrogen-bond analysis requires topology element information."
                ) from exc
            if len(solute_sites) == 0:
                raise ValueError(
                    f"Hydrogen-bond selection contains no "
                    f"{sorted(allowed_elements)} sites: {selection!r}."
                )
            if len(water_sites) == 0:
                raise ValueError(
                    "Hydrogen-bond water_selection contains no "
                    f"{sorted(allowed_elements)} sites: {water_selection!r}."
                )

            directions = []
            for donor_sites, acceptor_sites in (
                (solute_sites, water_sites),
                (water_sites, solute_sites),
            ):
                paired_donors, paired_hydrogens = _donor_hydrogen_pairs(
                    universe,
                    donor_sites,
                    frame_index=frame_index,
                )
                directions.append(
                    (paired_donors, paired_hydrogens, acceptor_sites)
                )
                for donor in paired_donors:
                    for acceptor in acceptor_sites:
                        if donor.index != acceptor.index:
                            possible_labels.add(_label(donor, acceptor))

        for paired_donors, paired_hydrogens, acceptor_sites in directions:
            if len(paired_donors) == 0:
                continue

            pairs = capped_distance(
                paired_donors.positions,
                acceptor_sites.positions,
                max_cutoff=float(donor_acceptor_cutoff),
                box=box,
                return_distances=False,
            )
            if pairs.size:
                nonself = (
                    paired_donors.indices[pairs[:, 0]]
                    != acceptor_sites.indices[pairs[:, 1]]
                )
                pairs = pairs[nonself]
            if pairs.size:
                donor_atoms = paired_donors[pairs[:, 0]]
                hydrogen_atoms = paired_hydrogens[pairs[:, 0]]
                acceptor_atoms = acceptor_sites[pairs[:, 1]]
                angles = calc_angles(
                    donor_atoms.positions,
                    hydrogen_atoms.positions,
                    acceptor_atoms.positions,
                    box=box,
                )
                for index in np.flatnonzero(angles >= angle_threshold):
                    label = _label(donor_atoms[index], acceptor_atoms[index])
                    counts[label] = counts.get(label, 0) + 1
        n_frames += 1

    if n_frames == 0:
        raise ValueError("Hydrogen-bond frame slice selects no trajectory frames.")
    labels = tuple(sorted(possible_labels))
    if not labels:
        raise ValueError(
            "Hydrogen-bond selections contain no non-self donor/acceptor pairs."
        )
    values = np.asarray([counts.get(label, 0) / n_frames for label in labels])
    return QoI(
        name="hydrogen_bonds",
        values=values,
        labels=labels,
        values_per_label=1,
        settings={
            "selection": selection,
            "water_selection": water_selection,
            "elements": tuple(sorted(allowed_elements)),
            "donor_acceptor_cutoff": float(donor_acceptor_cutoff),
            "angle_cutoff": float(angle_cutoff),
            "pbc": pbc,
            "update_selections": update_selections,
        },
    )
