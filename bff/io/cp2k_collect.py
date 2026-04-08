#!/usr/bin/env python3
"""Collect CP2K short-MD snapshot runs into extxyz datasets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import MDAnalysis as mda

POSITION_PATTERNS = ("*-pos-*.xyz", "*-pos-*.dcd")
FORCE_PATTERNS = ("*-frc-*.xyz", "*-frc-*.dcd", "*-force-*.xyz", "*-force-*.dcd")


def read_last_frame(path: Path, topology: Path | None = None):
    """Read the last coordinate frame with MDAnalysis."""
    if path.suffix.lower() == ".dcd":
        if topology is None:
            raise ValueError(f"Reading {path} requires a topology file.")
        universe = mda.Universe(str(topology), str(path))
    else:
        universe = mda.Universe(str(path))

    if universe.trajectory.n_frames <= 0:
        raise ValueError(f"No frames found in {path}")
    universe.trajectory[-1]
    atoms = [str(name) for name in universe.atoms.names]
    values = universe.atoms.positions.astype(float).tolist()
    return atoms, values


def last_potential_energy(run_dir: Path) -> float:
    """Read the last CP2K potential energy from a run directory."""
    for path in sorted(run_dir.glob("*.ener")):
        lines = path.read_text(errors="ignore").splitlines()
        for line in reversed(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) >= 5:
                return values[4]
            if values:
                return values[-1]
    raise FileNotFoundError(f"No readable CP2K .ener file in {run_dir}")


def first_existing(patterns: tuple[str, ...], run_dir: Path) -> Path:
    """Return the last matching file for the first glob pattern with matches."""
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        f"No file matching {', '.join(patterns)} in {run_dir}"
    )


def collect_frame(run_dir: Path, topology_name: str = "pos.xyz"):
    """Collect one extxyz frame from one finished CP2K short-MD run."""
    pos_path = first_existing(POSITION_PATTERNS, run_dir)
    frc_path = first_existing(FORCE_PATTERNS, run_dir)
    topology = run_dir / topology_name
    topology = topology if topology.exists() else None
    atoms, positions = read_last_frame(pos_path, topology=topology)
    force_atoms, forces = read_last_frame(frc_path, topology=topology)
    if atoms != force_atoms:
        raise ValueError(f"Atom labels differ between {pos_path} and {frc_path}")
    return {
        "source": run_dir.name,
        "atoms": atoms,
        "positions": positions,
        "forces": forces,
        "energy": last_potential_energy(run_dir),
    }


def write_extxyz(frames: list[dict], path: Path) -> None:
    """Write frames in extended XYZ format."""
    with path.open("w") as handle:
        for frame in frames:
            handle.write(f"{len(frame['atoms'])}\n")
            handle.write(
                "Properties=species:S:1:pos:R:3:forces:R:3 "
                f"energy={frame['energy']:.16g} "
                f"source=\"{frame['source']}\"\n"
            )
            for atom, position, force in zip(
                frame["atoms"],
                frame["positions"],
                frame["forces"],
                strict=True,
            ):
                values = [*position, *force]
                handle.write(
                    atom
                    + " "
                    + " ".join(f"{value:.12g}" for value in values)
                    + "\n"
                )


def train_valid_split(frames: list[dict], train_fraction: float):
    """Split frames deterministically into train and validation blocks."""
    if not frames:
        return [], []
    if len(frames) == 1:
        return frames, []
    n_train = round(len(frames) * train_fraction)
    n_train = min(max(n_train, 1), len(frames) - 1)
    return frames[:n_train], frames[n_train:]


def collect_outputs(
    *,
    runs: Path,
    train: Path,
    valid: Path,
    train_fraction: float = 0.8,
    seed: int = 2026,
    topology_name: str = "pos.xyz",
) -> tuple[int, int]:
    """Collect CP2K outputs and return train/validation frame counts."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    runs_dir = Path(runs)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")

    frames = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        try:
            frames.append(collect_frame(run_dir, topology_name=topology_name))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping {run_dir}: {exc}")

    random.Random(seed).shuffle(frames)
    train_frames, valid_frames = train_valid_split(frames, train_fraction)
    write_extxyz(train_frames, Path(train))
    write_extxyz(valid_frames, Path(valid))
    return len(train_frames), len(valid_frames)


def main() -> None:
    """Collect CP2K short-MD outputs into train/valid extxyz files."""
    parser = argparse.ArgumentParser(
        description="Collect CP2K snapshot MD outputs into extxyz datasets."
    )
    parser.add_argument("--runs", default="runs", help="Directory with runs.")
    parser.add_argument("--train", default="train.extxyz")
    parser.add_argument("--valid", default="valid.extxyz")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--topology",
        default="pos.xyz",
        help="Per-run topology file used when CP2K outputs DCD files.",
    )
    args = parser.parse_args()

    n_train, n_valid = collect_outputs(
        runs=Path(args.runs),
        train=Path(args.train),
        valid=Path(args.valid),
        train_fraction=args.train_fraction,
        seed=args.seed,
        topology_name=args.topology,
    )
    print(f"Wrote {n_train} training frames to {args.train}")
    print(f"Wrote {n_valid} validation frames to {args.valid}")


if __name__ == "__main__":
    main()
