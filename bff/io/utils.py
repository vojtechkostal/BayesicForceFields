import json
import tarfile
import yaml
import numpy as np
from pathlib import Path

from .mdp import get_restraints


def path_to_str(path: Path) -> str:
    """Convert a Path object to a string."""
    return str(path)


class NumpyYAMLEncoder(yaml.SafeDumper):
    """YAML encoder for numpy arrays and numpy scalar types."""

    def represent_numpy(self, obj):
        if isinstance(obj, np.ndarray):
            return self.represent_list(obj.tolist())
        elif isinstance(obj, (np.generic, np.number)):
            if isinstance(obj, np.floating):
                return self.represent_float(obj.item())
            else:
                return self.represent_int(obj.item())
        return super().represent_data(obj)


# Add the custom representers for NumPy types
NumpyYAMLEncoder.add_multi_representer(np.ndarray, NumpyYAMLEncoder.represent_numpy)
NumpyYAMLEncoder.add_multi_representer(np.generic, NumpyYAMLEncoder.represent_numpy)


def save_yaml(data: dict, fn: Path | str) -> None:
    """Save a dictionary as YAML to a file."""
    fn = str(fn) if isinstance(fn, Path) else fn
    with open(fn, "w") as f:
        yaml.dump(data, f, Dumper=NumpyYAMLEncoder, default_flow_style=False)


def load_yaml(fn: Path | str) -> dict:
    """Load .yaml file into a dictionary"""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with open(fn, "r") as f:
        file = yaml.safe_load(f)
    return file


class NumpyArrayEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays, scalars, and floats with 3 decimals."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Convert to nested lists, rounding elementwise
            return self._round_nested(obj.tolist())
        elif isinstance(obj, (np.generic, np.number)):
            return round(obj.item(), 3)
        elif isinstance(obj, float):
            return round(obj, 3)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

    def _round_nested(self, obj):
        """Recursively round floats in nested lists/structs."""
        if isinstance(obj, list):
            return [self._round_nested(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 3)
        return obj


def save_json(data: dict, fn: Path | str) -> None:
    """Save a dictionary as JSON with floats rounded to 3 decimals."""
    fn = str(fn) if isinstance(fn, Path) else fn
    with open(fn, "w") as f:
        json.dump(data, f, cls=NumpyArrayEncoder)


def load_json(fn: Path | str) -> dict:
    """Save a dictionary as JSON to a file."""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with open(fn, "r") as f:
        file = json.load(f)
    return file


def load_qoi(fn_qoi: Path | str) -> np.ndarray:
    """
    Load quantities of interest from a .npz file.
    """
    return np.load(fn_qoi, allow_pickle=True)


def save_qoi(qoi, fn_out: str | Path, settings: dict) -> None:
    """Save quantities of interest to a compressed .npz file.

    Parameters
    ----------
    qoi : list of list of objects
        Quantities of interest to be saved.
    fn_out : str or Path
        Output filename for the .npz file.
    """

    qoi = np.atleast_2d(qoi)
    data_dict = np.array([[trj.__dict__ for trj in sample] for sample in qoi])

    data = {'settings': settings} | {'samples': data_dict}
    fn_out = Path(fn_out).resolve()
    np.savez_compressed(fn_out, **data)


def compress_results(source_dir: Path | str) -> None:
    """Compress the results into a tarball."""

    extensions = {'.tpr', '.xtc', '.yaml', '.top', '.gro'}
    source_path = Path(source_dir).resolve()
    tar_filename = source_path.with_suffix('.tar.gz')
    with tarfile.open(tar_filename, "w:gz") as tar:
        for file_path in source_path.rglob('*'):
            if file_path.suffix in extensions:
                arcname = file_path.relative_to(source_path.parent)
                tar.add(file_path, arcname=str(arcname))


def extract_tarball(fn: Path) -> None:
    """Extract a tarball file into a directory."""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with tarfile.open(fn, "r:gz") as tar:
        tar.extractall(path=fn.parent)


def prepare_path(fn: Path) -> Path:
    """Prepare a directory from a tarball or validate its existence."""
    if fn.suffix == ".gz":
        train_dir = fn.parent / fn.name.replace(".tar.gz", "")
        if not train_dir.exists():
            extract_tarball(fn)
        return train_dir
    elif fn.is_dir():
        return fn
    else:
        raise ValueError(f"Unknown file type: {fn}")


def extract_train_dir(
    train_dir: Path
) -> tuple[dict, dict, list[Path], list[Path], list[dict]]:

    """Extract training directory contents.

    Parameters
    ----------
    train_dir : Path
        Path to the training directory.

    Returns
    -------
    tuple
        A tuple containing:
        - specs (dict): Specifications loaded from specs.yaml, or None if not present.
        - samples (dict): Samples loaded from samples.yaml, or None if not present.
        - fn_topol (list of Path): List of topology file paths.
        - fn_coord (list of Path): List of coordinate file paths.
        - restraints (list of dict): List of restraint dictionaries
        extracted from .mdp files.

    """
    if not train_dir.is_dir():
        raise ValueError(f"Provided train_dir '{train_dir}' is not a valid directory.")

    fn_specs = train_dir / "specs.yaml"
    specs = load_yaml(fn_specs) if fn_specs.exists() else None

    fn_samples = train_dir / "samples.yaml"
    samples = load_yaml(fn_samples) if fn_samples.exists() else None

    fn_mdp = sorted(train_dir.glob("*prod*.mdp"))
    fn_topol = sorted(train_dir.glob("topol-*.top"))
    fn_coord = sorted(train_dir.glob("coords-*.gro"))
    restraints = [get_restraints(f) for f in fn_mdp]

    if not (len(fn_mdp) == len(fn_topol) == len(fn_coord)):
        raise ValueError(f"Missing .mdp, .top, or .gro files in {train_dir}.")

    return specs, samples, list(fn_topol), list(fn_coord), restraints
