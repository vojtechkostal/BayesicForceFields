"""Shared workflow configuration helpers and base models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

from ...domain.bias import BiasSpec
from ...io.utils import load_yaml

PathLike = Union[str, Path]
SchedulerName = Literal["local", "slurm"]


def _resolve_path(
    base_dir: Path,
    path: PathLike,
    *,
    must_exist: bool = True,
    kind: str = "path",
) -> Path:
    resolved = (base_dir / path).resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{kind.capitalize()} not found: {resolved}")
    return resolved


def _resolve_optional_path(
    base_dir: Path,
    path: PathLike | None,
    *,
    must_exist: bool = True,
    kind: str = "file",
) -> Path | None:
    if path is None:
        return None
    return _resolve_path(base_dir, path, must_exist=must_exist, kind=kind)


def _normalize_store(value: Any) -> list[str]:
    if value in (None, True):
        return ["xtc"]
    if value is False:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("'store' entries must all be strings.")
        return list(value)
    raise ValueError("'store' must be a bool, string, or list of strings.")


@dataclass(frozen=True)
class SlurmConfig:
    max_parallel_jobs: int = 1
    sbatch: dict[str, Any] | None = None
    setup: tuple[str, ...] = ()
    teardown: tuple[str, ...] = ()


def _load_slurm_config(slurm_raw: Any) -> SlurmConfig:
    if not isinstance(slurm_raw, dict):
        raise ValueError("Missing 'slurm' configuration for slurm scheduler.")
    if "sbatch" not in slurm_raw:
        raise ValueError("Scheduler 'slurm' must define 'sbatch'.")
    if not isinstance(slurm_raw["sbatch"], dict):
        raise ValueError("slurm.sbatch must be a mapping.")

    setup = slurm_raw.get("setup", [])
    teardown = slurm_raw.get("teardown", [])
    if not isinstance(setup, list) or not all(isinstance(cmd, str) for cmd in setup):
        raise ValueError("slurm.setup must be a list of shell commands.")
    if not isinstance(teardown, list) or not all(
        isinstance(cmd, str) for cmd in teardown
    ):
        raise ValueError("slurm.teardown must be a list of shell commands.")

    max_parallel_jobs = int(slurm_raw.get("max_parallel_jobs", 1))
    if max_parallel_jobs == 0 or max_parallel_jobs < -1:
        raise ValueError("'slurm.max_parallel_jobs' must be positive or -1.")

    return SlurmConfig(
        max_parallel_jobs=max_parallel_jobs,
        sbatch=dict(slurm_raw["sbatch"]),
        setup=tuple(setup),
        teardown=tuple(teardown),
    )


def _validate_bounds(bounds: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(bounds, dict):
        raise ValueError("'bounds' must be a mapping of parameter names to bounds.")

    normalized: dict[str, tuple[float, float]] = {}
    for name, value in bounds.items():
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(x, (int, float)) for x in value)
        ):
            raise ValueError(f"Invalid bounds for {name!r}: {value}")
        lower, upper = float(value[0]), float(value[1])
        if lower > upper:
            raise ValueError(
                f"Lower bound {lower} is greater than upper bound {upper} "
                f"for parameter {name!r}."
            )
        normalized[name] = (lower, upper)
    return normalized


def _normalize_implicit_atoms(value: Any) -> list[str]:
    if isinstance(value, str):
        return value.split()
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    raise ValueError("'implicit_atoms' must be a string or a list of strings.")


def _load_model_paths(
    base_dir: Path,
    models_raw: Any,
) -> dict[str, Path]:
    if not isinstance(models_raw, Mapping) or not models_raw:
        raise ValueError("'models' must be a non-empty mapping.")

    models: dict[str, Path] = {}
    for name, path in models_raw.items():
        if not isinstance(name, str):
            raise ValueError("Model names in 'models' must be strings.")
        if not isinstance(path, (str, Path)):
            raise ValueError(f"Model path for {name!r} must be a file path.")
        models[name] = _resolve_path(
            base_dir,
            path,
            kind=f"model file for {name!r}",
        )
    return models


@dataclass(frozen=True)
class SimulationSystemConfig:
    system_id: str
    fn_topol: Path
    fn_coordinates: Path
    fn_mdp_em: Path | None
    fn_mdp_prod: Path
    fn_ndx: Path
    bias: BiasSpec
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        bias_file = self.bias.input_file
        return {
            "system_id": self.system_id,
            "topology": str(self.fn_topol),
            "coordinates": str(self.fn_coordinates),
            "mdp": {
                "em": None if self.fn_mdp_em is None else str(self.fn_mdp_em),
                "prod": str(self.fn_mdp_prod),
            },
            "index": str(self.fn_ndx),
            "bias": None if bias_file is None else str(bias_file),
            "n_steps": int(self.n_steps),
        }


def _load_simulation_systems(
    base_dir: Path,
    systems_raw: Any,
    *,
    key: str,
) -> list[SimulationSystemConfig]:
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError(f"'{key}' must be a non-empty list.")

    systems: list[SimulationSystemConfig] = []
    for i, system in enumerate(systems_raw):
        if not isinstance(system, dict):
            raise ValueError(f"{key}[{i}] must be a mapping.")
        if "assets" in system:
            if "n_steps" not in system:
                raise ValueError(f"{key}[{i}] must define 'n_steps'.")
            training_assets = _resolve_path(
                base_dir,
                system["assets"],
                kind=f"{key}[{i}] assets directory",
            )
            systems.append(
                _load_prepared_simulation_system(
                    training_assets,
                    int(system["n_steps"]),
                    system_id=f"{i:03d}",
                )
            )
            continue

        for required_key in ("topology", "coordinates", "mdp", "index", "n_steps"):
            if required_key not in system:
                raise ValueError(
                    f"{key}[{i}] is missing required key {required_key!r}."
                )

        mdp = system["mdp"]
        if not isinstance(mdp, dict):
            raise ValueError(f"{key}[{i}].mdp must be a mapping.")
        if "prod" not in mdp:
            raise ValueError(f"{key}[{i}].mdp is missing required key 'prod'.")

        n_steps = int(system["n_steps"])
        if n_steps <= 0:
            raise ValueError(f"{key}[{i}].n_steps must be a positive integer.")

        systems.append(
            SimulationSystemConfig(
                system_id=f"{i:03d}",
                fn_topol=_resolve_path(
                    base_dir,
                    system["topology"],
                    kind=f"{key}[{i}] topology file",
                ),
                fn_coordinates=_resolve_path(
                    base_dir,
                    system["coordinates"],
                    kind=f"{key}[{i}] coordinate file",
                ),
                fn_mdp_em=_resolve_optional_path(
                    base_dir,
                    mdp.get("em"),
                    kind=f"{key}[{i}] EM MDP file",
                ),
                fn_mdp_prod=_resolve_path(
                    base_dir,
                    mdp["prod"],
                    kind=f"{key}[{i}] production MDP file",
                ),
                fn_ndx=_resolve_path(
                    base_dir,
                    system["index"],
                    kind=f"{key}[{i}] index file",
                ),
                bias=BiasSpec.from_any(system.get("bias"), base_dir=base_dir),
                n_steps=n_steps,
            )
        )
    return systems


def _load_prepared_simulation_system(
    training_assets: Path,
    n_steps: int,
    *,
    system_id: str,
) -> SimulationSystemConfig:
    if n_steps <= 0:
        raise ValueError(
            f"Prepared system assets {training_assets} have invalid n_steps={n_steps}."
        )

    topologies = sorted(training_assets.glob("*.top"))
    if len(topologies) != 1:
        raise ValueError(
            f"{training_assets} must contain exactly one prepared system, "
            f"but found {len(topologies)} topology files."
        )

    fn_topol = topologies[0]
    system_label = fn_topol.stem
    fn_coordinates = training_assets / f"{system_label}.gro"
    fn_mdp_em = training_assets / f"{system_label}.em.mdp"
    fn_mdp_prod = training_assets / f"{system_label}.mdp"
    fn_ndx = training_assets / f"{system_label}.ndx"
    fn_bias = None
    for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
        candidate = training_assets / f"{system_label}.{suffix}"
        if candidate.exists():
            fn_bias = candidate
            break

    for path, kind in [
        (fn_coordinates, "coordinate"),
        (fn_mdp_em, "EM MDP"),
        (fn_mdp_prod, "production MDP"),
        (fn_ndx, "index"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Prepared {kind} file not found for {system_label}: {path}"
            )

    return SimulationSystemConfig(
        system_id=system_id,
        fn_topol=fn_topol,
        fn_coordinates=fn_coordinates,
        fn_mdp_em=fn_mdp_em,
        fn_mdp_prod=fn_mdp_prod,
        fn_ndx=fn_ndx,
        bias=BiasSpec.from_any(fn_bias, base_dir=training_assets),
        n_steps=n_steps,
    )


@dataclass(frozen=True, kw_only=True)
class SimulationCampaignConfig:
    fn_config: Path
    campaign_dir: Path
    log: Path
    gmx_cmd: str
    job_scheduler: SchedulerName
    systems: list[SimulationSystemConfig]
    dispatch: bool = True
    compress: bool = False
    cleanup: bool = False
    store: tuple[str, ...] = ()
    slurm: Optional[SlurmConfig] = None

    @classmethod
    def _load_common(
        cls,
        fn_config: PathLike,
    ) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
        return _load_campaign_common(fn_config)


def _load_campaign_common(
    fn_config: PathLike,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    fn_config = Path(fn_config).resolve()
    base_dir = fn_config.parent
    config = load_yaml(fn_config)

    required = ["campaign_dir", "systems", "job_scheduler", "gmx_cmd"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            "Missing required configuration key(s): "
            + ", ".join(repr(key) for key in missing)
        )

    scheduler = config["job_scheduler"]
    if scheduler not in {"local", "slurm"}:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r}. Supported values are "
            "'local' and 'slurm'."
        )

    systems = _load_simulation_systems(base_dir, config["systems"], key="systems")

    slurm = None
    if scheduler == "slurm":
        slurm = _load_slurm_config(config.get("slurm"))

    common = dict(
        fn_config=fn_config,
        campaign_dir=_resolve_path(
            base_dir,
            config["campaign_dir"],
            must_exist=False,
            kind="campaign directory",
        ),
        log=_resolve_path(
            base_dir,
            config.get("log", "./out.log"),
            must_exist=False,
            kind="log file",
        ),
        gmx_cmd=str(config["gmx_cmd"]),
        job_scheduler=scheduler,
        systems=systems,
        dispatch=bool(config.get("dispatch", True)),
        compress=bool(config.get("compress", False)),
        cleanup=bool(config.get("cleanup", False)),
        store=tuple(_normalize_store(config.get("store"))),
        slurm=slurm,
    )
    return fn_config, base_dir, config, common
