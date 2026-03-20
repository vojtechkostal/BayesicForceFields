import os
import platform
from importlib import import_module

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

__version__ = "0.1.0"

__all__ = [
    "BFFLearner",
    "bayes",
    "data",
    "io",
    "plotting",
    "qoi",
    "structures",
    "tools",
]


def __getattr__(name: str):
    if name == "BFFLearner":
        from .bff import BFFLearner

        return BFFLearner

    if name in {"bayes", "data", "io", "plotting", "qoi", "structures", "tools"}:
        return import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
