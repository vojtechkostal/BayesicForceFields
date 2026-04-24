"""Read and write helpers for extended XYZ files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PathLike = str | Path
_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
_ENERGY_RE = re.compile(rf"\benergy=({_FLOAT_RE})")
_SOURCE_RE = re.compile(r'source="([^"]+)"')
_PBC_RE = re.compile(r'pbc="([^"]+)"')


def _atom_symbols(atoms: Any) -> list[str]:
    if hasattr(atoms, 'elements'):
        elements = [str(element) for element in atoms.elements]
        if all(elements):
            return elements
    return [str(name) for name in atoms.names]


def cell_vectors_from_dimensions(
    dimensions: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Convert MDAnalysis box dimensions to a 3x3 lattice matrix."""
    a, b, c, alpha, beta, gamma = np.asarray(dimensions, dtype=float)[:6]
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    sin_gamma = np.sin(gamma)
    if np.isclose(sin_gamma, 0.0):
        raise ValueError('Cannot build lattice vectors for a degenerate unit cell.')

    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * np.cos(gamma), b * sin_gamma, 0.0
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    cz_sq = c**2 - cx**2 - cy**2
    cz = np.sqrt(max(cz_sq, 0.0))
    return np.array(
        [
            [ax, ay, az],
            [bx, by, bz],
            [cx, cy, cz],
        ],
        dtype=float,
    )


def format_lattice(lattice: np.ndarray | Sequence[float]) -> str:
    """Format lattice values with 4 decimals and clean near-zero noise."""
    values = np.asarray(lattice, dtype=float).reshape(-1).copy()
    values[np.abs(values) < 5e-5] = 0.0
    return ' '.join(f'{value:.4f}' for value in values)


def write_extxyz_frame(
    atoms: Any,
    fn_out: PathLike,
    *,
    dimensions: np.ndarray | Sequence[float],
    properties: str = 'species:S:1:pos:R:3',
) -> None:
    """Write one frame in extended XYZ format including the lattice."""
    lattice = cell_vectors_from_dimensions(dimensions).reshape(-1)
    comment = (
        f'Lattice="{format_lattice(lattice)}" '
        f'Properties={properties} pbc="T T T"'
    )
    positions = np.asarray(atoms.positions, dtype=float)
    path = Path(fn_out)
    with path.open('w', encoding='utf-8') as handle:
        handle.write(f'{len(atoms)}\n')
        handle.write(comment + '\n')
        for symbol, position in zip(_atom_symbols(atoms), positions, strict=True):
            handle.write(
                symbol
                + ' '
                + ' '.join(f'{value:.12g}' for value in position)
                + '\n'
            )


def read_xyz_comment(path: PathLike) -> str:
    """Return the comment line from the first XYZ frame."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines) < 2:
        raise ValueError(f'{path} is not a valid XYZ file.')
    return lines[1].strip()


def last_xyz_frame(path: PathLike) -> tuple[list[str], list[list[float]]]:
    """Return symbols and coordinates from the last XYZ frame in a file."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if len(lines) < 2:
        raise ValueError(f'{path} is not a valid XYZ trajectory file.')

    n_atoms = int(lines[0].strip())
    frame_len = n_atoms + 2
    if len(lines) < frame_len:
        raise ValueError(f'{path} does not contain a complete XYZ frame.')

    last_frame = lines[-frame_len:]
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in last_frame[2:]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f'Invalid XYZ atom line in {path}: {line}')
        symbols.append(parts[0])
        positions.append([float(value) for value in parts[1:4]])
    return symbols, positions


def _read_extxyz_properties(comment: str) -> tuple[list[float] | None, float | None, str | None, str | None]:
    lattice = None
    match = _LATTICE_RE.search(comment)
    if match is not None:
        values = [float(value) for value in match.group(1).split()]
        if len(values) != 9:
            raise ValueError('Invalid Lattice field in extxyz comment.')
        lattice = values

    energy = None
    match = _ENERGY_RE.search(comment)
    if match is not None:
        energy = float(match.group(1))

    source = None
    match = _SOURCE_RE.search(comment)
    if match is not None:
        source = match.group(1)

    pbc = None
    match = _PBC_RE.search(comment)
    if match is not None:
        pbc = match.group(1)

    return lattice, energy, source, pbc


def read_extxyz_frame(path: PathLike) -> dict[str, object]:
    """Read one extended XYZ frame into a plain mapping."""
    lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    if len(lines) < 2:
        raise ValueError(f'{path} is not a valid extxyz file.')

    n_atoms = int(lines[0].strip())
    lattice, energy, source, pbc = _read_extxyz_properties(lines[1])
    atoms: list[str] = []
    positions: list[list[float]] = []
    forces: list[list[float]] | None = []

    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f'Invalid extxyz atom line in {path}: {line}')
        atoms.append(parts[0])
        positions.append([float(value) for value in parts[1:4]])
        if len(parts) >= 7:
            assert forces is not None
            forces.append([float(value) for value in parts[4:7]])
        else:
            forces = None

    return {
        'atoms': atoms,
        'positions': positions,
        'forces': forces,
        'energy': energy,
        'lattice': lattice,
        'pbc': pbc,
        'source': source,
    }


def write_extxyz_frames(frames: Sequence[dict[str, object]], path: PathLike) -> None:
    """Write one or more extended XYZ frames."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for frame in frames:
            atoms = frame['atoms']
            positions = frame['positions']
            forces = frame.get('forces')
            properties = frame.get('properties')
            if properties is None:
                properties = (
                    'species:S:1:pos:R:3:forces:R:3'
                    if forces is not None
                    else 'species:S:1:pos:R:3'
                )

            handle.write(f"{len(atoms)}\n")
            header: list[str] = []
            lattice = frame.get('lattice')
            if lattice is not None:
                header.append(f'Lattice="{format_lattice(lattice)}"')
                header.append(f'pbc="{frame.get("pbc") or "T T T"}"')
            header.append(f'Properties={properties}')

            energy = frame.get('energy')
            if energy is not None:
                header.append(f'energy={float(energy):.16g}')
            source = frame.get('source')
            if source is not None:
                header.append(f'source="{source}"')
            handle.write(' '.join(header) + '\n')

            if forces is None:
                for atom, position in zip(atoms, positions, strict=True):
                    handle.write(
                        atom
                        + ' '
                        + ' '.join(f'{value:.12g}' for value in position)
                        + '\n'
                    )
                continue

            for atom, position, force in zip(atoms, positions, forces, strict=True):
                values = [*position, *force]
                handle.write(
                    atom + ' ' + ' '.join(f'{value:.12g}' for value in values) + '\n'
                )
    tmp.replace(path)
