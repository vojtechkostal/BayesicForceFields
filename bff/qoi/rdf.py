"""Built-in radial distribution function routine."""

from __future__ import annotations

from typing import Any

import MDAnalysis as mda
import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.mdamath import triclinic_vectors
from scipy.ndimage import gaussian_filter

from .dataset import QoI
from .trajectory import get_unitcell, select_atoms

OPTIONS = {"group_a", "group_b", "range", "bins", "pbc", "update_selections", "smooth"}


def rdf(
    universe: mda.Universe,
    *,
    frames: slice,
    options: dict[str, Any],
) -> QoI:
    """Radial distribution functions from each atom type of group_a to group_b.

    Options: ``group_a`` and ``group_b`` (required selections), ``range``
    (default ``[0, 10]`` Angstrom), ``bins`` (200), ``pbc`` (true),
    ``update_selections`` (false), and ``smooth`` (false; Gaussian filter with
    a width of 3 bins).

    One curve is emitted per atom type in ``group_a``, sorted by type name.
    Atoms with mass <= 0.5 (virtual sites) are not used as centers. Without
    ``pbc`` the curves are not divided by the box volume.
    """
    unknown = set(options) - OPTIONS
    if unknown:
        raise ValueError(f"unsupported RDF option(s): {', '.join(sorted(unknown))}.")
    missing = {"group_a", "group_b"} - set(options)
    if missing:
        raise ValueError(f"RDF requires selection(s): {', '.join(sorted(missing))}.")
    distance_range = options.get("range", (0.0, 10.0))
    bins = options.get("bins", 200)
    pbc = options.get("pbc", True)
    update_selections = options.get("update_selections", False)
    smooth = options.get("smooth", False)
    if not all(isinstance(value, bool) for value in (pbc, update_selections, smooth)):
        raise ValueError(
            "RDF options pbc, update_selections, and smooth must be booleans."
        )
    if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
        raise ValueError(f"RDF bins must be a positive integer, got {bins!r}.")
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
            "RDF range must contain two increasing non-negative numbers, "
            f"got {distance_range!r}."
        )
    distance_range = tuple(float(value) for value in distance_range)

    group_a = select_atoms(
        universe, options["group_a"], updating=update_selections, field="group_a"
    )
    group_b = select_atoms(
        universe, options["group_b"], updating=update_selections, field="group_b"
    )
    try:
        centers = group_a[group_a.masses > 0.5]
    except NoDataError as exc:
        raise ValueError(
            "RDF requires topology masses to exclude virtual sites."
        ) from exc
    if len(centers) == 0:
        raise ValueError(
            f"RDF group_a contains no atoms with mass greater than 0.5: "
            f"{options['group_a']!r}."
        )
    labels = tuple(sorted(set(centers.types.astype(str))))

    edges = np.linspace(distance_range[0], distance_range[1], bins + 1)
    shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    counts = np.zeros((len(labels), bins))
    normalization = np.zeros((len(labels), bins))
    n_frames = 0
    for ts in universe.trajectory[frames]:
        box = None
        volume = 1.0
        if pbc:
            box = get_unitcell(universe, ts)
            volume = abs(float(np.linalg.det(triclinic_vectors(box))))
        centers = group_a[group_a.masses > 0.5]
        distances = distance_array(centers.positions, group_b.positions, box=box)
        nonself = centers.indices[:, None] != group_b.indices[None, :]
        center_types = centers.types.astype(str)
        for index, label in enumerate(labels):
            rows = center_types == label
            pair_distances = distances[rows][nonself[rows]]
            if pair_distances.size == 0:
                raise ValueError(
                    f"RDF frame {ts.frame} has no non-self atom pairs for "
                    f"center type {label!r}."
                )
            counts[index] += np.histogram(pair_distances, bins=edges)[0]
            normalization[index] += pair_distances.size * shell_volumes / volume
        n_frames += 1
    if n_frames == 0:
        raise ValueError("RDF frame slice selects no trajectory frames.")

    values = counts / normalization
    if smooth:
        values = gaussian_filter(values, sigma=(0, 3))
    return QoI(
        name="rdf",
        values=values.reshape(-1),
        labels=labels,
        values_per_label=bins,
        settings={
            "group_a": options["group_a"],
            "group_b": options["group_b"],
            "range": distance_range,
            "bins": bins,
            "pbc": pbc,
            "update_selections": update_selections,
            "smooth": smooth,
        },
    )
