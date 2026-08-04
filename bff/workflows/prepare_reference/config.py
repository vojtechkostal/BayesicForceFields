from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...domain.systems import load_system_ids
from ...io.utils import load_yaml
from .._shared.config import PathLike, _resolve_path
from .._shared.preparation import PreparedSystem, load_build_system


@dataclass(frozen=True)
class PrepareReferenceConfig:
    fn_config: Path
    source: Path
    output_dir: Path
    log: Path
    systems: tuple[PreparedSystem, ...]
    n_single_point_snapshots: int

    @classmethod
    def load(cls, fn_config: PathLike) -> "PrepareReferenceConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError("Prepare-reference configuration must contain a mapping.")
        unknown = set(config) - {
            "source", "output", "log", "systems", "n_single_point_snapshots"
        }
        if unknown:
            raise ValueError(
                "Prepare-reference configuration contains unsupported key(s): "
                + ", ".join(sorted(unknown))
            )
        missing = [key for key in ("source", "systems") if key not in config]
        if missing:
            raise ValueError(
                "Prepare-reference configuration requires key(s): "
                + ", ".join(repr(key) for key in missing)
            )
        source = _resolve_path(base_dir, config["source"], kind="build stage root")
        output_dir = _resolve_path(
            base_dir,
            config.get("output", "./"),
            must_exist=False,
            kind="reference output directory",
        )
        system_ids = load_system_ids(config["systems"])
        n_snapshots = int(config.get("n_single_point_snapshots", 1000))
        if n_snapshots <= 0:
            raise ValueError("n_single_point_snapshots must be a positive integer.")
        return cls(
            fn_config=fn_config,
            source=source,
            output_dir=output_dir,
            log=_resolve_path(
                base_dir,
                config.get("log", output_dir / "prepare-reference.log"),
                must_exist=False,
                kind="prepare-reference log file",
            ),
            systems=tuple(load_build_system(source, value) for value in system_ids),
            n_single_point_snapshots=n_snapshots,
        )
