from dataclasses import dataclass
from pathlib import Path
import re

import MDAnalysis as mda

from .mdp import patch_mdp


PathLike = str | Path


@dataclass(frozen=True, slots=True)
class ColvarsDistanceBias:
    """Parsed distance bias from a COLVARS file.

    Parameters
    ----------
    name
        Colvar name.
    atom_indices
        One-based atom indices of the restrained pair.
    center
        Optional harmonic center in nm.
    """

    name: str
    atom_indices: tuple[int, int]
    center: float | None = None


def _extract_named_blocks(text: str, name: str) -> list[str]:
    """Extract balanced top-level blocks of one COLVARS section type."""
    pattern = re.compile(rf"\b{name}\s*\{{", re.MULTILINE)
    blocks: list[str] = []
    start = 0
    while True:
        match = pattern.search(text, start)
        if match is None:
            return blocks

        i = match.end() - 1
        depth = 0
        end = i
        while end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[i + 1:end])
                    start = end + 1
                    break
            end += 1
        else:
            raise ValueError(f"Unbalanced '{name}' block in COLVARS file.")


def _extract_atom_numbers(block: str, group_name: str) -> tuple[int, ...]:
    """Extract atom numbers from one COLVARS group block."""
    groups = _extract_named_blocks(block, group_name)
    if not groups:
        return ()
    match = re.search(r"\batomNumbers\s+([0-9\s]+)", groups[0])
    if match is None:
        return ()
    return tuple(int(value) for value in match.group(1).split())


def parse_distance_biases(fn_colvars: PathLike) -> list[ColvarsDistanceBias]:
    """Parse simple distance/harmonic restraints from a COLVARS file.

    Parameters
    ----------
    fn_colvars
        Path to the COLVARS input file.

    Returns
    -------
    list of ColvarsDistanceBias
        Parsed single-pair distance colvars. Only colvars with one atom in
        ``group1`` and one atom in ``group2`` are returned.
    """
    text = Path(fn_colvars).read_text()

    colvars: dict[str, tuple[int, int]] = {}
    for block in _extract_named_blocks(text, "colvar"):
        name_match = re.search(r"^\s*name\s+(\S+)", block, re.MULTILINE)
        if name_match is None:
            continue
        distance_blocks = _extract_named_blocks(block, "distance")
        if not distance_blocks:
            continue
        group_1 = _extract_atom_numbers(distance_blocks[0], "group1")
        group_2 = _extract_atom_numbers(distance_blocks[0], "group2")
        if len(group_1) != 1 or len(group_2) != 1:
            continue
        colvars[name_match.group(1)] = (group_1[0], group_2[0])

    centers: dict[str, float] = {}
    for block in _extract_named_blocks(text, "harmonic"):
        name_match = re.search(r"^\s*colvars\s+(\S+)", block, re.MULTILINE)
        center_match = re.search(r"^\s*centers\s+([-+0-9.eE]+)", block, re.MULTILINE)
        if name_match is None or center_match is None:
            continue
        centers[name_match.group(1)] = float(center_match.group(1))

    return [
        ColvarsDistanceBias(
            name=name,
            atom_indices=indices,
            center=centers.get(name),
        )
        for name, indices in colvars.items()
    ]


def resolve_distance_bias_metadata(
    fn_colvars: PathLike,
    fn_system: PathLike,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Resolve atom-name pairs and centers from a COLVARS file.

    Parameters
    ----------
    fn_colvars
        Path to the COLVARS input file.
    fn_system
        Structure file used to map atom indices to atom names.

    Returns
    -------
    tuple
        Tuple of ``(pair_labels, centers)`` where labels are atom-name pairs
        such as ``"C2 CAL"`` and centers are harmonic centers in nm.
    """
    universe = mda.Universe(str(fn_system))
    distances = parse_distance_biases(fn_colvars)

    labels: list[str] = []
    centers: list[float] = []
    for distance in distances:
        index_1, index_2 = distance.atom_indices
        atom_1 = universe.atoms[index_1 - 1].name
        atom_2 = universe.atoms[index_2 - 1].name
        labels.append(f"{atom_1} {atom_2}")
        if distance.center is not None:
            centers.append(float(distance.center))

    return tuple(labels), tuple(centers)


def write_mdp_with_colvars(
    fn_mdp: PathLike,
    fn_colvars: PathLike,
    fn_out: PathLike,
    *,
    seed: int | None = None,
) -> None:
    """Write an MDP file with GROMACS Colvars enabled.

    Parameters
    ----------
    fn_mdp
        Input MDP file.
    fn_colvars
        Colvars configuration file referenced from the generated MDP.
    fn_out
        Output MDP file with Colvars options injected.
    seed
        Optional seed passed to the Colvars module.
    """
    updates = {
        "colvars-active": "yes",
        "colvars-configfile": str(Path(fn_colvars).resolve()),
    }
    if seed is not None:
        updates["colvars-seed"] = str(int(seed))
    patch_mdp(fn_mdp, updates, fn_out)
