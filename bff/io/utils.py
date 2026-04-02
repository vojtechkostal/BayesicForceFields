import json
import tarfile
from pathlib import Path
from typing import Union

import numpy as np
import yaml

PathLike = Union[str, Path]


def path_to_str(path: PathLike) -> str:
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


def save_yaml(data: dict, fn: PathLike) -> None:
    """Save a dictionary as YAML to a file."""
    fn = str(fn) if isinstance(fn, Path) else fn
    with open(fn, "w") as f:
        yaml.dump(data, f, Dumper=NumpyYAMLEncoder, default_flow_style=False)


def load_yaml(fn: PathLike) -> dict:
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


def save_json(data: dict, fn: PathLike) -> None:
    """Save a dictionary as JSON with floats rounded to 3 decimals."""
    fn = str(fn) if isinstance(fn, Path) else fn
    with open(fn, "w") as f:
        json.dump(data, f, cls=NumpyArrayEncoder)


def load_json(fn: PathLike) -> dict:
    """Save a dictionary as JSON to a file."""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with open(fn, "r") as f:
        file = json.load(f)
    return file


def _to_json_compatible(data: object) -> object:
    """Convert nested numpy objects into plain JSON-compatible values."""
    if isinstance(data, dict):
        return {str(key): _to_json_compatible(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_to_json_compatible(value) for value in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.generic, np.number)):
        return data.item()
    return data


def save_pt(data: object, fn: PathLike) -> None:
    """Save a JSON-serializable object to a `.pt` file."""
    path = Path(fn)
    if path.suffix != ".pt":
        raise ValueError(f"Expected a '.pt' file, got {path}.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_compatible(data), f)


def load_pt(fn: PathLike) -> object:
    """Load an object from a `.pt` file.

    The preferred format is plain JSON. Legacy Torch `.pt` archives are still
    accepted when PyTorch is available.
    """
    path = Path(fn)
    if path.suffix != ".pt":
        raise ValueError(f"Expected a '.pt' file, got {path}.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Legacy binary .pt file detected at {path}, but PyTorch is not "
                "installed. Install PyTorch or regenerate the file with the "
                "current BFF version."
            ) from exc
        return torch.load(path, map_location="cpu", weights_only=False)


def compress_results(source_dir: PathLike) -> None:
    """Compress the results into a tarball."""

    extensions = {'.tpr', '.xtc', '.yaml', '.top', '.gro'}
    source_path = Path(source_dir).resolve()
    tar_filename = source_path.with_suffix('.tar.gz')
    with tarfile.open(tar_filename, "w:gz") as tar:
        for file_path in source_path.rglob('*'):
            if file_path.suffix in extensions:
                arcname = file_path.relative_to(source_path.parent)
                tar.add(file_path, arcname=str(arcname))


def extract_tarball(fn: PathLike) -> None:
    """Extract a tarball file into a directory."""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with tarfile.open(fn, "r:gz") as tar:
        tar.extractall(path=fn.parent)


def prepare_path(fn: PathLike) -> Path:
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
    train_dir: PathLike
) -> tuple[dict | None, list[dict], dict | None]:

    """Extract training directory contents.

    Parameters
    ----------
    train_dir : Path
        Path to the training directory.

    Returns
    -------
    tuple
        A tuple containing:
        - specs (dict | None): Specifications loaded from specs.yaml.
        - systems (list[dict]): Staged system records from samples.yaml.
        - samples (dict | None): Sample records loaded from samples.yaml.
    """
    if not train_dir.is_dir():
        raise ValueError(f"Provided train_dir '{train_dir}' is not a valid directory.")

    fn_specs = train_dir / "specs.yaml"
    specs = load_yaml(fn_specs) if fn_specs.exists() else None

    fn_samples = train_dir / "samples.yaml"
    campaign = load_yaml(fn_samples) if fn_samples.exists() else None
    systems = None if campaign is None else campaign.get("systems")
    samples = None if campaign is None else campaign.get("samples")

    if not systems:
        raise ValueError(f"No staged simulation systems found in {train_dir}.")

    fn_topol = [train_dir / system["topology"] for system in systems]
    fn_coord = [train_dir / system["coordinates"] for system in systems]

    if not all(path.exists() for path in fn_topol + fn_coord):
        raise ValueError(f"Missing staged topology or coordinate files in {train_dir}.")

    return specs, list(systems), samples
