"""I/O helpers for BFF, loaded lazily to keep imports lightweight."""

from importlib import import_module

__all__ = [
    "colvars",
    "cp2k",
    "cp2k_collect",
    "logs",
    "mdp",
    "plumed",
    "progress",
    "schedulers",
    "utils",
]


def __getattr__(name: str):
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
