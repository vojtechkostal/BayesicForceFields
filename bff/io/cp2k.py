import re
from pathlib import Path
from typing import Sequence

from .extxyz import last_xyz_frame, read_extxyz_frame, write_extxyz_frames

ATOMIC_NUMBERS = {
    'h': 1,
    'li': 3,
    'c': 6,
    'n': 7,
    'o': 8,
    'f': 9,
    'na': 11,
    'mg': 12,
    'p': 15,
    's': 16,
    'cl': 17,
    'k': 19,
    'ca': 20,
    'br': 35,
    'rb': 37,
    'i': 53,
}

def write_cp2k_single_atom_xyz(
    element: str,
    fn_out: str | Path,
    *,
    box_length: float = 10.0,
) -> None:
    """Write a centered one-atom XYZ file for isolated vacuum calculations."""
    center = float(box_length) / 2.0
    symbol = element.capitalize()
    xyz = (
        "1\n"
        f"{symbol} isolated in vacuum\n"
        f"{symbol} {center:.6f} {center:.6f} {center:.6f}\n"
    )
    Path(fn_out).write_text(xyz)


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
ENERGY_RE = re.compile(
    rf"ENERGY\| Total FORCE_EVAL .*?({FLOAT_RE})\s*$",
    re.MULTILINE,
)
PIPE_FORCE_RE = re.compile(
    r"^\s*FORCES\|\s+\d+\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s+"
    rf"{FLOAT_RE}\s*$",
    re.MULTILINE,
)
LEGACY_FORCE_ROW_RE = re.compile(
    r"^\s*\d+\s+\d+\s+\S+\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s+"
    rf"({FLOAT_RE})\s*$"
)


def read_cp2k_energy(
    fn_output: str | Path,
    *,
    context: str = 'energy',
) -> float:
    """Read a CP2K total energy from standard output and return it in eV."""
    fn_output = Path(fn_output)
    match = ENERGY_RE.search(fn_output.read_text(encoding='utf-8', errors='ignore'))
    if match is None:
        raise ValueError(f'Could not extract {context} from {fn_output}.')
    return float(match.group(1)) * HARTREE_TO_EV


def _convert_force(values: tuple[str, str, str]) -> list[float]:
    return [
        float(value) * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM
        for value in values
    ]


def _parse_pipe_forces(text: str) -> list[list[float]]:
    return [
        _convert_force(match.group(1, 2, 3))
        for match in PIPE_FORCE_RE.finditer(text)
    ]


def _parse_legacy_atomic_forces(text: str) -> list[list[float]]:
    tables: list[list[list[float]]] = []
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        if not re.match(r"^\s*ATOMIC FORCES\b", lines[index]):
            index += 1
            continue

        table: list[list[float]] = []
        index += 1
        while index < len(lines):
            line = lines[index]
            if re.match(r"^\s*SUM OF ATOMIC FORCES\b", line):
                break

            match = LEGACY_FORCE_ROW_RE.match(line)
            if match is not None:
                table.append(_convert_force(match.group(1, 2, 3)))
            index += 1

        if table:
            tables.append(table)
        index += 1

    return tables[-1] if tables else []


def read_cp2k_forces(fn_output: str | Path) -> list[list[float]]:
    """Read atomic forces from CP2K output and return them in eV/Angstrom."""
    fn_output = Path(fn_output)
    text = fn_output.read_text(encoding='utf-8', errors='ignore')
    forces = _parse_pipe_forces(text)
    if not forces:
        forces = _parse_legacy_atomic_forces(text)
    if not forces:
        raise ValueError(f'Could not extract atomic forces from {fn_output}.')
    return forces


def write_cp2k_snapshot_extxyz(
    run_dir: str | Path,
    *,
    snapshot_filename: str = 'pos.xyz',
    trajectory_filename: str = 'md-pos-1.xyz',
    output_filename: str = 'sp.out',
    extxyz_filename: str = 'sp.extxyz',
) -> Path:
    """Write one reference extxyz frame from a staged CP2K snapshot job."""
    run_dir = Path(run_dir)
    snapshot_frame = read_extxyz_frame(run_dir / snapshot_filename)
    symbols, positions = last_xyz_frame(run_dir / trajectory_filename)
    forces = read_cp2k_forces(run_dir / output_filename)
    energy = read_cp2k_energy(run_dir / output_filename, context='energy')
    if len(symbols) != len(forces):
        raise ValueError(
            f'Force extraction failed for {run_dir}: expected {len(symbols)} rows, '
            f'got {len(forces)}.'
        )

    frame = {
        'atoms': symbols,
        'positions': positions,
        'forces': forces,
        'energy': energy,
        'lattice': snapshot_frame.get('lattice'),
        'pbc': snapshot_frame.get('pbc') or 'T T T',
        'source': 'sp',
    }
    fn_extxyz = run_dir / extxyz_filename
    write_extxyz_frames([frame], fn_extxyz)
    return fn_extxyz


def collect_single_atom_energies(
    single_atom_dirs: Sequence[str | Path],
) -> dict[int, float]:
    """Collect isolated-atom CP2K energies from staged single-atom jobs."""
    energies: dict[int, float] = {}
    for atom_dir in sorted(Path(path) for path in single_atom_dirs):
        fn_output = atom_dir / 'atom.out'
        if not fn_output.exists():
            raise FileNotFoundError(f'Missing isolated-atom output file: {fn_output}')
        lines = (atom_dir / 'pos.xyz').read_text(encoding='utf-8').splitlines()
        if len(lines) < 3:
            raise ValueError(f'Invalid isolated-atom XYZ file: {atom_dir / "pos.xyz"}')
        symbol = lines[2].split()[0].lower()
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(
                f'Unsupported isolated-atom element in {atom_dir / "pos.xyz"}: {symbol}'
            )
        energies[ATOMIC_NUMBERS[symbol]] = read_cp2k_energy(
            fn_output,
            context='isolated-atom energy',
        )
    return energies
