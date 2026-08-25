import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import yaml

PathLike = Union[str, Path]


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
    fn = str(fn)
    with open(fn, "r") as f:
        file = yaml.safe_load(f)
    return file


class NumpyArrayEncoder(json.JSONEncoder):
    """Lossless JSON encoder for numpy arrays and scalar values."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.number)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

def save_json(data: dict, fn: PathLike) -> None:
    """Save a dictionary as JSON without changing numeric precision."""
    fn = str(fn) if isinstance(fn, Path) else fn
    with open(fn, "w") as f:
        json.dump(data, f, cls=NumpyArrayEncoder)


def load_json(fn: PathLike) -> dict:
    """Save a dictionary as JSON to a file."""
    fn = str(fn)
    with open(fn, "r") as f:
        file = json.load(f)
    return file


def file_sha256(filename: PathLike) -> str:
    """Return the SHA-256 hash of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_fingerprint(data: object) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible metadata."""
    encoded = json.dumps(
        _to_json_compatible(data),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_torch_save(data: object, filename: PathLike) -> None:
    """Atomically replace a Torch artifact in its destination directory."""
    import torch

    path = Path(filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
