"""Shared workflow configuration helpers and base models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

from ...domain.bias import BiasSpec
from ...domain.systems import (
    resolve_explicit_inputs,
    validate_system_id,
    validate_unique_system_ids,
)
from ...io.utils import load_yaml
from .preparation import load_build_system

PathLike = Union[str, Path]
SchedulerName = Literal["local", "slurm"]


def _strict_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false, got {value!r}.")
    return value


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
    unknown = set(slurm_raw) - {
        "max_parallel_jobs",
        "sbatch",
        "setup",
        "teardown",
    }
    if unknown:
        raise ValueError(
            "slurm contains unsupported key(s): " + ", ".join(sorted(unknown))
        )
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


@dataclass(frozen=True)
class SimulationSystemConfig:
    system_id: str
    topology_path: Path
    coordinates_path: Path
    mdp_em_path: Path | None
    mdp_production_path: Path
    index_path: Path
    bias: BiasSpec
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        bias_file = self.bias.input_file
        return {
            "system_id": self.system_id,
            "inputs": {
                "topology": str(self.topology_path),
                "coordinates": str(self.coordinates_path),
                "mdp_em": (
                    None if self.mdp_em_path is None else str(self.mdp_em_path)
                ),
                "mdp_production": str(self.mdp_production_path),
                "index": str(self.index_path),
                "bias": None if bias_file is None else str(bias_file),
            },
            "n_steps": int(self.n_steps),
        }


def _load_simulation_systems(
    base_dir: Path,
    systems_raw: Any,
    *,
    key: str,
    source: Path | None = None,
) -> list[SimulationSystemConfig]:
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError(f"'{key}' must be a non-empty list.")

    systems: list[SimulationSystemConfig] = []
    for i, system in enumerate(systems_raw):
        if not isinstance(system, dict):
            raise ValueError(f"{key}[{i}] must be a mapping.")
        for required_key in ("system_id", "n_steps"):
            if required_key not in system:
                raise ValueError(
                    f"{key}[{i}] is missing required key {required_key!r}."
                )
        system_id = validate_system_id(
            system["system_id"], field=f"{key}[{i}].system_id"
        )
        n_steps = int(system["n_steps"])
        if n_steps <= 0:
            raise ValueError(f"{key}[{i}].n_steps must be a positive integer.")

        if source is not None:
            if "inputs" in system or "assets" in system:
                raise ValueError(
                    f"{key}[{i}] mixes source selection with direct inputs; "
                    "remove 'inputs'/'assets'."
                )
            unknown = set(system) - {"system_id", "n_steps"}
            if unknown:
                raise ValueError(
                    f"{key}[{i}] contains unsupported key(s): "
                    + ", ".join(sorted(unknown))
                )
            built = load_build_system(source, system_id)
            systems.append(
                SimulationSystemConfig(
                    system_id=system_id,
                    topology_path=built.topology_path,
                    coordinates_path=built.production_coordinates_path,
                    mdp_em_path=built.mdp_em_path,
                    mdp_production_path=built.mdp_production_path,
                    index_path=built.index_path,
                    bias=built.bias,
                    n_steps=n_steps,
                )
            )
            continue
        else:
            if "assets" in system:
                raise ValueError(
                    f"{key}[{i}].assets directory discovery is unsupported; "
                    "use top-level source or explicit inputs."
                )
            if "inputs" not in system or not isinstance(system["inputs"], dict):
                raise ValueError(
                    f"{key}[{i}].inputs must be a mapping of named file roles."
                )
            unknown = set(system) - {"system_id", "n_steps", "inputs"}
            if unknown:
                raise ValueError(
                    f"{key}[{i}] contains unsupported key(s): "
                    + ", ".join(sorted(unknown))
                )
            inputs = resolve_explicit_inputs(
                system["inputs"],
                base_dir=base_dir,
                system_id=system_id,
                field=f"{key}[{i}].inputs",
            )

        allowed_roles = {
            "topology",
            "coordinates",
            "index",
            "mdp_em",
            "mdp_npt",
            "mdp_production",
            "bias",
        }
        unsupported = set(inputs.inputs) - allowed_roles
        if unsupported:
            raise ValueError(
                f"{key}[{i}].inputs contains unsupported role(s): "
                + ", ".join(sorted(unsupported))
            )

        systems.append(
            SimulationSystemConfig(
                system_id=system_id,
                topology_path=inputs.require_path("topology"),
                coordinates_path=inputs.require_path("coordinates"),
                mdp_em_path=inputs.optional_path("mdp_em"),
                mdp_production_path=inputs.require_path("mdp_production"),
                index_path=inputs.require_path("index"),
                bias=BiasSpec.from_any(inputs.optional_path("bias")),
                n_steps=n_steps,
            )
        )
    validate_unique_system_ids(
        [system.system_id for system in systems], field=key
    )
    return systems


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

def _load_campaign_common(
    fn_config: PathLike,
    *,
    log_name: str = "sample-parameters.log",
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    fn_config = Path(fn_config).resolve()
    base_dir = fn_config.parent
    config = load_yaml(fn_config)
    if not isinstance(config, dict):
        raise ValueError("Simulation campaign configuration must contain a mapping.")

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

    source_raw = config.get("source")
    source = (
        None
        if source_raw is None
        else _resolve_path(base_dir, source_raw, kind="build stage root")
    )
    systems = _load_simulation_systems(
        base_dir,
        config["systems"],
        key="systems",
        source=source,
    )

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
            config.get("log", Path(config["campaign_dir"]) / log_name),
            must_exist=False,
            kind="log file",
        ),
        gmx_cmd=str(config["gmx_cmd"]),
        job_scheduler=scheduler,
        systems=systems,
        dispatch=_strict_bool(config.get("dispatch", True), field="dispatch"),
        compress=_strict_bool(config.get("compress", False), field="compress"),
        cleanup=_strict_bool(config.get("cleanup", False), field="cleanup"),
        store=tuple(_normalize_store(config.get("store"))),
        slurm=slurm,
    )
    return fn_config, base_dir, config, common
