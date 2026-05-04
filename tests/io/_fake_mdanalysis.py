from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class _FakeAtoms:
    def __init__(self, names: list[str], positions: list[list[float]]) -> None:
        self.names = np.asarray(names, dtype=object)
        self.elements = np.asarray(names, dtype=object)
        self.positions = np.asarray(positions, dtype=float)


class _FakeTrajectory:
    n_frames = 1

    def __getitem__(self, index: object) -> "_FakeTrajectory":
        return self


class _FakeUniverse:
    def __init__(self, *paths: str) -> None:
        path = Path(paths[-1])
        lines = path.read_text().splitlines()
        n_atoms = int(lines[0].strip())
        names: list[str] = []
        positions: list[list[float]] = []

        for line in lines[2 : 2 + n_atoms]:
            atom, x, y, z, *_ = line.split()
            names.append(atom)
            positions.append([float(x), float(y), float(z)])

        self.atoms = _FakeAtoms(names, positions)
        self.trajectory = _FakeTrajectory()


def install(monkeypatch: Any | None = None) -> None:
    """Install a tiny MDAnalysis stub for unit tests."""
    module = ModuleType("MDAnalysis")
    module.Universe = _FakeUniverse
    if monkeypatch is None:
        sys.modules["MDAnalysis"] = module
    else:
        monkeypatch.setitem(sys.modules, "MDAnalysis", module)
