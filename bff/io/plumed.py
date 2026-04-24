"""Helpers for running bias-driven simulations with PLUMED."""

import os
import shutil
import sys
from pathlib import Path

PathLike = str | Path


def find_plumed_kernel() -> Path | None:
    """Locate a PLUMED kernel library if one is available."""
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
    """Ensure that a PLUMED kernel is available for biased MD runs."""
    kernel = find_plumed_kernel()
    if kernel is None:
        raise RuntimeError(
            "PLUMED biasing requires a loadable PLUMED kernel. "
            "Set the PLUMED_KERNEL environment variable or install PLUMED "
            "so that libplumedKernel is discoverable."
        )
    return kernel
