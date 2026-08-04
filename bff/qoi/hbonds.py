"""Explicit-selection hydrogen-bond quantities of interest."""

from __future__ import annotations

from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import calc_angles, capped_distance

from .data import QoI
from .rdf import _box_and_volume, _select_group

__all__ = ["compute_hydrogen_bond_qoi"]


def _donor_hydrogen_pairs(
    universe: mda.Universe,
    donors: mda.AtomGroup,
    hydrogens: mda.AtomGroup,
    *,
    frame_index: int,
) -> tuple[mda.AtomGroup, mda.AtomGroup]:
    donor_indices = set(int(index) for index in donors.indices)
    paired_donors: list[int] = []
    paired_hydrogens: list[int] = []
    try:
        for hydrogen in hydrogens:
            bonded = [
                atom.index
                for atom in hydrogen.bonded_atoms
                if atom.index in donor_indices
            ]
            if len(bonded) > 1:
                raise ValueError(
                    f"Hydrogen atom {hydrogen.index} matches multiple selected "
                    f"donors at frame {frame_index}."
                )
            if bonded:
                paired_donors.append(bonded[0])
                paired_hydrogens.append(hydrogen.index)
    except NoDataError as exc:
        raise ValueError(
            "Hydrogen-bond analysis requires topology bond information; "
            f"none is available at frame {frame_index}."
        ) from exc
    if not paired_donors:
        raise ValueError(
            "Hydrogen-bond selections contain no bonded donor/hydrogen pairs "
            f"at frame {frame_index}."
        )
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
    donors: str,
    hydrogens: str,
    acceptors: str,
    donor_acceptor_cutoff: float = 3.5,
    angle_cutoff: float = 150.0,
    pbc: bool = True,
    update_selections: bool = False,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> QoI:
    """Count mean hydrogen bonds using explicit atom selections."""
    if not isinstance(pbc, bool) or not isinstance(update_selections, bool):
        raise ValueError(
            "Hydrogen-bond pbc and update_selections options must be booleans."
        )
    if donor_acceptor_cutoff <= 0:
        raise ValueError("donor_acceptor_cutoff must be positive.")
    if not 0 < angle_cutoff <= 180:
        raise ValueError("angle_cutoff must be in (0, 180].")
    donor_group = _select_group(
        universe, donors, updating=update_selections, field="selections.donors"
    )
    hydrogen_group = _select_group(
        universe,
        hydrogens,
        updating=update_selections,
        field="selections.hydrogens",
    )
    acceptor_group = _select_group(
        universe,
        acceptors,
        updating=update_selections,
        field="selections.acceptors",
    )
    counts: dict[str, int] = {}
    possible_labels: set[str] = set()
    n_frames = 0
    angle_threshold = np.deg2rad(angle_cutoff)

    for frame_index, ts in enumerate(
        universe.trajectory[slice(start, stop, step)], start=start
    ):
        for field, selection, group in (
            ("donors", donors, donor_group),
            ("hydrogens", hydrogens, hydrogen_group),
            ("acceptors", acceptors, acceptor_group),
        ):
            if len(group) == 0:
                raise ValueError(
                    f"Hydrogen-bond selection {field} became empty at frame "
                    f"{frame_index}: {selection!r}."
                )
        paired_donors, paired_hydrogens = _donor_hydrogen_pairs(
            universe,
            donor_group,
            hydrogen_group,
            frame_index=frame_index,
        )
        for donor in paired_donors:
            for acceptor in acceptor_group:
                if donor.index != acceptor.index:
                    possible_labels.add(_label(donor, acceptor))

        box = None
        if pbc:
            box, _ = _box_and_volume(
                ts, context=f"Hydrogen-bond frame {frame_index}"
            )
        pairs = capped_distance(
            paired_donors.positions,
            acceptor_group.positions,
            max_cutoff=float(donor_acceptor_cutoff),
            box=box,
            return_distances=False,
        )
        if pairs.size:
            nonself = (
                paired_donors.indices[pairs[:, 0]]
                != acceptor_group.indices[pairs[:, 1]]
            )
            pairs = pairs[nonself]
        if pairs.size:
            donor_atoms = paired_donors[pairs[:, 0]]
            hydrogen_atoms = paired_hydrogens[pairs[:, 0]]
            acceptor_atoms = acceptor_group[pairs[:, 1]]
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
            "donors": donors,
            "hydrogens": hydrogens,
            "acceptors": acceptors,
            "donor_acceptor_cutoff": float(donor_acceptor_cutoff),
            "angle_cutoff": float(angle_cutoff),
            "pbc": pbc,
            "update_selections": update_selections,
        },
    )
