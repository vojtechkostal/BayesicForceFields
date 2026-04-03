import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda

PathLike = str | Path


@dataclass(frozen=True, slots=True)
class PlumedDistanceBias:
    """Parsed distance bias from a PLUMED file.

    Parameters
    ----------
    name
        Name of the distance collective variable.
    atom_indices
        One-based atom indices defining the distance.
    center
        Optional restraint center in nm.
    """

    name: str
    atom_indices: tuple[int, int]
    center: float | None = None


def _parse_arguments(text: str) -> dict[str, str]:
    """Parse key-value arguments from one PLUMED line."""
    pairs = re.findall(r"([A-Za-z_]+)=([^\s]+)", text)
    return {key.upper(): value for key, value in pairs}


def find_plumed_kernel() -> Path | None:
    """Locate a PLUMED kernel library if one is available.

    Returns
    -------
    pathlib.Path or None
        Path to a loadable PLUMED kernel library, or ``None`` if one could
        not be located from the current environment.
    """
    env_value = os.environ.get("PLUMED_KERNEL")
    if env_value:
        env_path = Path(env_value).expanduser().resolve()
        if env_path.exists():
            return env_path

    candidate_dirs = [
        Path(sys.prefix) / "lib",
        Path(sys.prefix).resolve().parent / "lib",
    ]

    plumed_executable = shutil.which("plumed")
    if plumed_executable is not None:
        candidate_dirs.append(Path(plumed_executable).resolve().parents[1] / "lib")

    seen: set[Path] = set()
    for lib_dir in candidate_dirs:
        lib_dir = lib_dir.resolve()
        if lib_dir in seen or not lib_dir.exists():
            continue
        seen.add(lib_dir)
        candidates = sorted(lib_dir.glob("libplumedKernel*"))
        if candidates:
            return candidates[0]

    return None


def ensure_plumed_kernel() -> Path:
    """Ensure that a PLUMED kernel is available for biased MD runs.

    Returns
    -------
    pathlib.Path
        Path to the PLUMED kernel library.

    Raises
    ------
    RuntimeError
        If no PLUMED kernel can be located.
    """
    kernel = find_plumed_kernel()
    if kernel is None:
        raise RuntimeError(
            "PLUMED biasing requires a loadable PLUMED kernel. "
            "Set the PLUMED_KERNEL environment variable or install PLUMED "
            "so that libplumedKernel is discoverable."
        )
    return kernel


def parse_distance_biases(fn_plumed: PathLike) -> list[PlumedDistanceBias]:
    """Parse simple distance restraints from a PLUMED input file.

    Parameters
    ----------
    fn_plumed
        Path to the PLUMED input file.

    Returns
    -------
    list of PlumedDistanceBias
        Parsed distance collective variables with optional restraint centers.
    """
    lines = Path(fn_plumed).read_text().splitlines()
    distances: dict[str, tuple[int, int]] = {}
    centers: dict[str, float] = {}

    for raw_line in lines:
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or ":" not in line:
            continue

        name, expression = [part.strip() for part in line.split(":", maxsplit=1)]
        keyword = expression.split(maxsplit=1)[0].upper()
        args = _parse_arguments(expression)

        if keyword == "DISTANCE" and "ATOMS" in args:
            atom_indices = tuple(int(value) for value in args["ATOMS"].split(","))
            if len(atom_indices) == 2:
                distances[name] = atom_indices  # type: ignore[assignment]
            continue

        if keyword in {"RESTRAINT", "UPPER_WALLS", "LOWER_WALLS"}:
            arg_name = args.get("ARG")
            if arg_name is None:
                continue
            center_value = args.get("AT")
            if center_value is None:
                continue
            try:
                centers[arg_name] = float(center_value.split(",")[0])
            except ValueError:
                continue

    return [
        PlumedDistanceBias(
            name=name,
            atom_indices=atom_indices,
            center=centers.get(name),
        )
        for name, atom_indices in distances.items()
    ]


def resolve_distance_bias_metadata(
    fn_plumed: PathLike,
    fn_system: PathLike,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Resolve atom-name pairs and centers from a PLUMED file.

    Parameters
    ----------
    fn_plumed
        Path to the PLUMED input file.
    fn_system
        Structure file used to map atom indices to atom names.

    Returns
    -------
    tuple
        Tuple of ``(pair_labels, centers)`` where labels are atom-name pairs
        and centers are restraint centers in nm.
    """
    universe = mda.Universe(str(fn_system))
    distances = parse_distance_biases(fn_plumed)

    labels: list[str] = []
    centers: list[float] = []
    for distance in distances:
        index_1, index_2 = distance.atom_indices
        atom_1 = universe.atoms[index_1 - 1].name
        atom_2 = universe.atoms[index_2 - 1].name
        labels.append(f"{atom_1} {atom_2}")
        if distance.center is not None:
            centers.append(distance.center)

    return tuple(labels), tuple(centers)
