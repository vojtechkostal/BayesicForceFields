"""Bayesic Force Fields public package namespace."""

from importlib import import_module

__version__ = "0.1.0.dev3"

__all__ = [
    "bayes",
    "data",
    "domain",
    "io",
    "plotting",
    "qoi",
    "tools",
]


def __getattr__(name: str):
    if name in {
        "bayes",
        "data",
        "domain",
        "io",
        "plotting",
        "qoi",
        "tools",
    }:
        return import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
