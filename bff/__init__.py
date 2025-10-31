import os
import platform

if platform.system() == "Darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # For macOS compatibility with OpenMP
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads to 1
    os.environ["MKL_NUM_THREADS"] = "1"  # Limit MKL threads to 1

from .bff import BFFOptimizer
from . import postprocessing
from . import qoi
from . import data
from . import topology
from . import tools
from . import io
from . import structures
from . import bayes

__version__ = "0.1.0"

__all__ = [
    "BFFOptimizer",
    "postprocessing",
    "qoi",
    "data",
    "topology",
    "tools",
    "io",
    "structures",
    "bayes",
]
