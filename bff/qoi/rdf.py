"""Selection-driven radial distribution functions."""

from __future__ import annotations

from typing import Any, Mapping

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError, SelectionError
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.mdamath import triclinic_vectors
from scipy.ndimage import gaussian_filter

from .data import QoI


def validate_rdf_options(
    options: Mapping[str, Any],
    *,
    context: str = "RDF options",
) -> None:
    """Validate RDF options without requiring a trajectory."""
    boolean_values = [
        options.get("pbc", True),
        options.get("update_selections", False),
        options.get("smooth", False),
    ]
    if not all(isinstance(value, bool) for value in boolean_values):
        raise ValueError(f"{context}: boolean options must be true or false.")

    bins = options.get("bins", 200)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError(f"{context}.bins must be a positive integer, got {bins!r}.")

    distance_range = options.get("range", (0.0, 10.0))
    if not (
        isinstance(distance_range, (list, tuple))
        and len(distance_range) == 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in distance_range
        )
        and 0 <= distance_range[0] < distance_range[1]
    ):
        raise ValueError(
            f"{context}.range must contain two increasing non-negative numbers, "
            f"got {distance_range!r}."
        )


def _select_group(
    universe: mda.Universe,
    selection: str,
    *,
    updating: bool,
    field: str,
) -> mda.AtomGroup:
    try:
        group = universe.select_atoms(selection, updating=updating)
    except SelectionError as exc:
        raise ValueError(
            f"Invalid atom selection for {field}: {selection!r}."
        ) from exc
    if len(group) == 0:
        raise ValueError(f"Atom selection for {field} is empty: {selection!r}.")
    return group


def _box_and_volume(ts: Any, *, context: str) -> tuple[np.ndarray, float]:
    box = None if ts.dimensions is None else np.asarray(ts.dimensions, dtype=float)
    if box is None or box.shape != (6,) or not np.all(np.isfinite(box)):
        raise ValueError(f"{context}: PBC requires six finite box dimensions.")
    if np.any(box[:3] <= 0) or np.any(box[3:] <= 0) or np.any(box[3:] >= 180):
        raise ValueError(f"{context}: invalid PBC box dimensions {box.tolist()}.")
    volume = abs(float(np.linalg.det(triclinic_vectors(box))))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"{context}: invalid triclinic box volume {volume!r}.")
    return box, volume


def compute_rdf(
    universe: mda.Universe,
    atoms_a: mda.AtomGroup,
    atoms_b: mda.AtomGroup,
    *,
    distance_range: tuple[float, float] = (0.0, 10.0),
    bins: int = 200,
    pbc: bool = True,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an RDF between two AtomGroups over a trajectory slice."""
    validate_rdf_options(
        {
            "range": distance_range,
            "bins": bins,
            "pbc": pbc,
            "smooth": smooth,
        }
    )
    edges = np.linspace(distance_range[0], distance_range[1], bins + 1)
    shell_volumes = (4.0 * np.pi / 3.0) * (
        edges[1:] ** 3 - edges[:-1] ** 3
    )
    counts = np.zeros(bins, dtype=float)
    normalization = np.zeros(bins, dtype=float)
    n_frames = 0

    for frame_index, ts in enumerate(
        universe.trajectory[slice(start, stop, step)], start=start
    ):
        box = None
        volume = 1.0
        if pbc:
            box, volume = _box_and_volume(ts, context=f"RDF frame {frame_index}")
        distances = distance_array(
            atoms_a.positions,
            atoms_b.positions,
            box=box,
        )
        same_atoms = atoms_a.indices[:, None] == atoms_b.indices[None, :]
        valid_distances = distances[~same_atoms]
        n_pairs = valid_distances.size
        if n_pairs <= 0:
            raise ValueError(
                f"RDF frame {frame_index} has no non-self atom pairs "
                f"(centers={len(atoms_a)}, neighbors={len(atoms_b)})."
            )
        counts += np.histogram(valid_distances, bins=edges)[0]
        normalization += n_pairs * shell_volumes / volume
        n_frames += 1

    if n_frames == 0:
        raise ValueError("RDF frame slice selects no trajectory frames.")
    if np.any(normalization <= 0):
        raise ValueError("RDF normalization is zero for one or more bins.")
    values = counts / normalization
    if smooth:
        values = gaussian_filter(values, sigma=3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, values


def compute_rdf_qoi(
    universe: mda.Universe,
    *,
    group_a: str,
    group_b: str,
    range: tuple[float, float] = (0.0, 10.0),
    bins: int = 200,
    pbc: bool = True,
    update_selections: bool = False,
    smooth: bool = False,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
) -> QoI:
    validate_rdf_options(
        {
            "range": range,
            "bins": bins,
            "pbc": pbc,
            "update_selections": update_selections,
            "smooth": smooth,
        }
    )
    atoms_a = _select_group(
        universe,
        group_a,
        updating=update_selections,
        field="selections.group_a",
    )
    atoms_b = _select_group(
        universe,
        group_b,
        updating=update_selections,
        field="selections.group_b",
    )
    try:
        physical_atoms = universe.atoms[universe.atoms.masses > 0.5]
    except NoDataError as exc:
        raise ValueError(
            "RDF group_a requires topology masses to exclude virtual sites."
        ) from exc
    atoms_a = atoms_a.select_atoms(
        "group physical_atoms",
        physical_atoms=physical_atoms,
        updating=update_selections,
    )
    if len(atoms_a) == 0:
        raise ValueError(
            f"RDF group_a contains no atoms with mass greater than 0.5: "
            f"{group_a!r}."
        )

    labels = tuple(sorted({str(atom_type) for atom_type in atoms_a.types}))
    curves: list[np.ndarray] = []
    for atom_type in labels:
        matching_atoms = physical_atoms[physical_atoms.types == atom_type]
        centers = atoms_a.select_atoms(
            "group matching_atoms",
            matching_atoms=matching_atoms,
            updating=update_selections,
        )
        _, values = compute_rdf(
            universe,
            centers,
            atoms_b,
            distance_range=tuple(float(value) for value in range),
            bins=int(bins),
            pbc=pbc,
            start=start,
            stop=stop,
            step=step,
            smooth=smooth,
        )
        curves.append(values)

    settings = {
        "group_a": group_a,
        "group_b": group_b,
        "range": tuple(float(value) for value in range),
        "bins": int(bins),
        "pbc": pbc,
        "update_selections": update_selections,
        "smooth": smooth,
    }
    return QoI(
        name="rdf",
        values=np.concatenate(curves),
        labels=labels,
        values_per_label=int(bins),
        settings=settings,
    )
