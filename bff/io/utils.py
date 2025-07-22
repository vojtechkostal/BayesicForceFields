import json
import tarfile
import yaml
import numpy as np
from pathlib import Path


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
    """JSON encoder for numpy arrays, numpy scalar types,
    and other custom objects."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.number)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


def save_json(data: dict, fn: Path | str) -> None:
    """Save a dictionary as JSON to a file."""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with open(fn, "w") as f:
        json.dump(data, f, cls=NumpyArrayEncoder)


def load_json(fn: Path | str) -> dict:
    """Save a dictionary as JSON to a file."""
    fn = path_to_str(fn) if isinstance(fn, Path) else fn
    with open(fn, "r") as f:
        file = json.load(f)
    return file


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


def load_md_files(train_dir: Path) -> tuple:
    """Load specs, restraints, topology, and coordinates from a directory."""
    fn_specs = train_dir / "specs.yaml"
    fn_specs = fn_specs if fn_specs.exists() else None

    fn_samples = train_dir / "samples.yaml"
    fn_samples = load_yaml(fn_samples) if fn_samples.exists() else None

    fn_mdp = sorted([f for f in train_dir.glob("*prod*.mdp")])
    fn_topol = sorted(list(train_dir.glob("*.top")))
    fn_coord = sorted(list(train_dir.glob("*.gro")))

    assert len(fn_mdp) == len(fn_topol) == len(fn_coord), \
        f"Some files in {train_dir} are missing (.mdp, .top or .gro)."

    return fn_specs, fn_mdp, fn_topol, fn_coord, fn_samples
