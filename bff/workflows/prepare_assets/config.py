from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...io.utils import load_yaml
from .._shared.config import PathLike, _resolve_path
from .._shared.preparation import (
    BuildManifest,
    PreparedSystem,
    load_manifest_system_ids,
    select_prepared_systems,
)


@dataclass(frozen=True)
class PrepareAssetsConfig:
    fn_config: Path
    manifest: BuildManifest
    ffmd_dir: Path
    reference_dir: Path
    systems: list[PreparedSystem]
    n_single_point_snapshots: int

    @classmethod
    def load(cls, fn_config: PathLike) -> "PrepareAssetsConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, dict):
            raise ValueError("Asset-preparation configuration must contain a mapping.")
        if "manifest" not in config:
            raise ValueError("Missing required asset-preparation key 'manifest'.")

        manifest = BuildManifest.load(
            _resolve_path(base_dir, config["manifest"], kind="build manifest")
        )
        ffmd_dir = _resolve_path(
            base_dir,
            config.get("ffmd_dir", manifest.fn_manifest.parent / "ffmd"),
            must_exist=False,
            kind="FFMD asset output directory",
        )
        reference_dir = _resolve_path(
            base_dir,
            config.get("reference_dir", manifest.fn_manifest.parent / "reference"),
            must_exist=False,
            kind="reference asset output directory",
        )
        n_snapshots = int(config.get("n_single_point_snapshots", 1000))
        if n_snapshots <= 0:
            raise ValueError("'n_single_point_snapshots' must be a positive integer.")

        system_ids = load_manifest_system_ids(config.get("systems"))
        return cls(
            fn_config=fn_config,
            manifest=manifest,
            ffmd_dir=ffmd_dir,
            reference_dir=reference_dir,
            systems=select_prepared_systems(manifest.systems, system_ids),
            n_single_point_snapshots=n_snapshots,
        )
