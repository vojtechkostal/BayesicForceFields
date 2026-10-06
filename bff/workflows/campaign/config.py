"""Configuration shared by simulation campaigns (sample-parameters, validate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ...domain.bias import BiasSpec
from ...domain.systems import (
    load_build_system_metadata,
    resolve_explicit_inputs,
    validate_system_id,
    validate_unique_system_ids,
)
from ...io.utils import load_yaml
from ...slurm import SlurmConfig, load_slurm_config
from ..config import PathLike, check_keys, resolve_path, strict_bool

SchedulerName = Literal["local", "slurm"]
CAMPAIGN_KEYS = {
    "campaign_dir",
    "log",
    "gmx_cmd",
    "job_scheduler",
    "source",
    "systems",
    "dispatch",
    "compress",
    "cleanup",
    "store",
    "scratch_dir",
    "max_restarts",
    "slurm",
}
INPUT_ROLES = {"topology", "coordinates", "index", "mdp_em", "mdp_production", "bias"}


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
        """Serialize for the per-sample job configuration."""
        return {
            "system_id": self.system_id,
            "inputs": {
                "topology": str(self.topology_path),
                "coordinates": str(self.coordinates_path),
                "mdp_em": None if self.mdp_em_path is None else str(self.mdp_em_path),
                "mdp_production": str(self.mdp_production_path),
                "index": str(self.index_path),
                "bias": None
                if self.bias.input_file is None
                else str(self.bias.input_file),
            },
            "n_steps": int(self.n_steps),
        }


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
    scratch_dir: str | None = None
    max_restarts: int = 0
    slurm: SlurmConfig | None = None


def normalize_store(value: Any) -> tuple[str, ...]:
    """File suffixes kept per system after a sample finishes (default: xtc)."""
    if value in (None, True):
        return ("xtc",)
    if value is False:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError("'store' must be a bool, string, or list of strings.")
    return tuple(item.lstrip(".") for item in value)


def _build_stage_system(
    source: Path, system_id: str, n_steps: int
) -> SimulationSystemConfig:
    """Read one system from the fixed file names of a ``bff build`` directory."""
    load_build_system_metadata(source, system_id)
    system_dir = source / "systems" / system_id

    def required(name: str) -> Path:
        path = system_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"system {system_id!r}: expected build output {name!r} at {path}; "
                "regenerate the build stage or correct 'source'."
            )
        return path

    colvars = system_dir / "bias.colvars.dat"
    plumed = system_dir / "bias.plumed.dat"
    if colvars.is_file() and plumed.is_file():
        raise ValueError(
            f"system {system_id!r}: both reserved bias files exist ({colvars} and "
            f"{plumed}); retain exactly one."
        )
    if colvars.is_file():
        bias = BiasSpec(kind="colvars", colvars_file=colvars)
    elif plumed.is_file():
        bias = BiasSpec(kind="plumed", plumed_file=plumed)
    else:
        bias = BiasSpec()
    required("npt.mdp")
    required("production.xtc")
    return SimulationSystemConfig(
        system_id=system_id,
        topology_path=required("topology.top"),
        coordinates_path=required("production.gro"),
        mdp_em_path=required("em.mdp"),
        mdp_production_path=required("production.mdp"),
        index_path=required("index.ndx"),
        bias=bias,
        n_steps=n_steps,
    )


def load_simulation_systems(
    systems_raw: Any,
    *,
    base_dir: Path,
    source: Path | None = None,
) -> list[SimulationSystemConfig]:
    """Load systems either from a build directory or from explicit inputs."""
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError("'systems' must be a non-empty list.")

    systems: list[SimulationSystemConfig] = []
    for i, raw in enumerate(systems_raw):
        where = f"systems[{i}]"
        allowed = (
            {"system_id", "n_steps"}
            if source is not None
            else {
                "system_id",
                "n_steps",
                "inputs",
            }
        )
        check_keys(raw, where=where, allowed=allowed, required=allowed)
        system_id = validate_system_id(raw["system_id"], field=f"{where}.system_id")
        n_steps = int(raw["n_steps"])
        if n_steps <= 0:
            raise ValueError(f"{where}.n_steps must be a positive integer.")
        if source is not None:
            systems.append(_build_stage_system(source, system_id, n_steps))
            continue

        if not isinstance(raw["inputs"], dict):
            raise ValueError(f"{where}.inputs must be a mapping of named file roles.")
        inputs = resolve_explicit_inputs(
            raw["inputs"],
            base_dir=base_dir,
            system_id=system_id,
            field=f"{where}.inputs",
        )
        unsupported = set(inputs.inputs) - INPUT_ROLES
        if unsupported:
            raise ValueError(
                f"{where}.inputs contains unsupported role(s): "
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
        [system.system_id for system in systems], field="systems"
    )
    return systems


def load_campaign_config(
    fn_config: PathLike,
    *,
    stage: str,
    stage_keys: set[str],
    stage_required: tuple[str, ...] = (),
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load the campaign part of a stage configuration.

    Returns the configuration directory, the raw mapping, and keyword
    arguments for :class:`SimulationCampaignConfig`.
    """
    fn_config = Path(fn_config).resolve()
    base_dir = fn_config.parent
    config = check_keys(
        load_yaml(fn_config),
        where=f"{stage} configuration",
        allowed=CAMPAIGN_KEYS | stage_keys,
        required=(
            "campaign_dir",
            "systems",
            "job_scheduler",
            "gmx_cmd",
            *stage_required,
        ),
    )
    scheduler = config["job_scheduler"]
    if scheduler not in {"local", "slurm"}:
        raise ValueError(
            f"Unsupported scheduler {scheduler!r}. Supported values are "
            "'local' and 'slurm'."
        )
    scratch_dir = config.get("scratch_dir")
    if scratch_dir is not None and not isinstance(scratch_dir, str):
        raise ValueError("'scratch_dir' must be a path; it may use $VARIABLES.")
    max_restarts = config.get("max_restarts", 0)
    if (
        not isinstance(max_restarts, int)
        or isinstance(max_restarts, bool)
        or (max_restarts < 0)
    ):
        raise ValueError("'max_restarts' must be a non-negative integer.")
    source = config.get("source")
    if source is not None:
        source = resolve_path(base_dir, source, kind="build stage root")
    campaign_dir = resolve_path(
        base_dir, config["campaign_dir"], must_exist=False, kind="campaign directory"
    )
    common = dict(
        fn_config=fn_config,
        campaign_dir=campaign_dir,
        log=resolve_path(
            base_dir,
            config.get("log", campaign_dir / f"{stage}.log"),
            must_exist=False,
            kind="log file",
        ),
        gmx_cmd=str(config["gmx_cmd"]),
        job_scheduler=scheduler,
        systems=load_simulation_systems(
            config["systems"], base_dir=base_dir, source=source
        ),
        dispatch=strict_bool(config.get("dispatch", True), field="dispatch"),
        compress=strict_bool(config.get("compress", False), field="compress"),
        cleanup=strict_bool(config.get("cleanup", False), field="cleanup"),
        store=normalize_store(config.get("store")),
        scratch_dir=scratch_dir,
        max_restarts=max_restarts,
        slurm=load_slurm_config(config.get("slurm")) if scheduler == "slurm" else None,
    )
    return base_dir, dict(config), common
