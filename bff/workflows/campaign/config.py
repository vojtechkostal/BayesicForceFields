"""Configuration shared by simulation campaigns (sample-parameters, validate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ...domain.bias import BiasSpec
from ...domain.systems import validate_system_id, validate_unique_system_ids
from ...slurm import SlurmConfig
from ..config import ConfigSection, PathLike, load_config

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
    "overwrite",
    "resume",
    "local",
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

    def to_dict(self, base_dir: Path) -> dict[str, Any]:
        """Serialize for ``campaign.yaml``, with paths relative to ``base_dir``."""

        def relative(path: Path | None) -> str | None:
            return None if path is None else str(path.relative_to(base_dir))

        return {
            "system_id": self.system_id,
            "inputs": {
                "topology": relative(self.topology_path),
                "coordinates": relative(self.coordinates_path),
                "mdp_em": relative(self.mdp_em_path),
                "mdp_production": relative(self.mdp_production_path),
                "index": relative(self.index_path),
                "bias": relative(self.bias.input_file),
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
    overwrite: bool = False
    resume: bool = False
    # Samples run at once with ``job_scheduler: local``.
    local_max_parallel_jobs: int = 1
    slurm: SlurmConfig | None = None


def _build_stage_system(
    source: Path, system_id: str, n_steps: int
) -> SimulationSystemConfig:
    """Read one system from the fixed file names of a ``bff build`` directory.

    Only the files a campaign uses are required; the build's own trajectory,
    NpT input, and metadata are not.
    """
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
    config: ConfigSection, *, source: Path | None = None
) -> list[SimulationSystemConfig]:
    """Load ``systems`` from a build directory or from explicit inputs."""
    systems: list[SimulationSystemConfig] = []
    keys = ("system_id", "n_steps") if source else ("system_id", "n_steps", "inputs")
    for raw in config.sections("systems", allowed=keys, required=keys):
        system_id = validate_system_id(
            raw.get("system_id"), field=raw.field("system_id")
        )
        n_steps = raw.integer("n_steps", minimum=1)
        if source is not None:
            systems.append(_build_stage_system(source, system_id, n_steps))
            continue

        inputs = raw.section(
            "inputs",
            allowed=INPUT_ROLES,
            required=("topology", "coordinates", "index", "mdp_production"),
        )
        bias = inputs.path("bias", None)
        systems.append(
            SimulationSystemConfig(
                system_id=system_id,
                topology_path=inputs.path("topology"),
                coordinates_path=inputs.path("coordinates"),
                mdp_em_path=inputs.path("mdp_em", None),
                mdp_production_path=inputs.path("mdp_production"),
                index_path=inputs.path("index"),
                bias=BiasSpec() if bias is None else BiasSpec.load(bias),
                n_steps=n_steps,
            )
        )
    validate_unique_system_ids(
        [system.system_id for system in systems], field="systems"
    )
    return systems


def load_slurm_config(config: ConfigSection) -> SlurmConfig:
    slurm = config.section(
        "slurm",
        allowed=("max_parallel_jobs", "max_array_size", "sbatch", "setup", "teardown"),
        required=("sbatch",),
    )
    sbatch = slurm.mapping("sbatch")
    if "array" in sbatch:
        raise ValueError(
            "slurm.sbatch.array is set by BFF; use slurm.max_parallel_jobs to "
            "limit concurrently running tasks."
        )
    return SlurmConfig(
        max_parallel_jobs=slurm.integer(
            "max_parallel_jobs", 1, minimum=1, special=(-1,)
        ),
        max_array_size=slurm.integer("max_array_size", 1000, minimum=1),
        sbatch=sbatch,
        setup=slurm.strings("setup", ()),
        teardown=slurm.strings("teardown", ()),
    )


def load_campaign_config(
    fn_config: PathLike,
    *,
    stage: str,
    stage_keys: set[str],
    stage_required: tuple[str, ...] = (),
) -> tuple[ConfigSection, dict[str, Any]]:
    """Load the campaign part of a stage configuration.

    Returns the configuration and keyword arguments for
    :class:`SimulationCampaignConfig`.
    """
    config = load_config(
        fn_config,
        stage=stage,
        allowed=CAMPAIGN_KEYS | stage_keys,
        required=(
            "campaign_dir", "systems", "job_scheduler", "gmx_cmd", *stage_required
        ),
    )
    scheduler = config.string("job_scheduler", choices=("local", "slurm"))
    campaign_dir = config.path("campaign_dir", must_exist=False)
    store = config.strings("store", ("xtc",))
    overwrite = config.boolean("overwrite", False)
    resume = config.boolean("resume", False)
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be true.")
    max_restarts = config.integer("max_restarts", 0, minimum=0)
    if max_restarts and scheduler == "local":
        raise ValueError(
            "max_restarts applies only to job_scheduler: slurm; local runs have "
            "no time limit."
        )
    local = config.section("local", allowed=("max_parallel_jobs",))
    common = dict(
        fn_config=Path(fn_config).resolve(),
        campaign_dir=campaign_dir,
        log=config.path("log", campaign_dir / f"{stage}.log", must_exist=False),
        gmx_cmd=config.string("gmx_cmd"),
        job_scheduler=scheduler,
        systems=load_simulation_systems(config, source=config.path("source", None)),
        dispatch=config.boolean("dispatch", True),
        compress=config.boolean("compress", False),
        cleanup=config.boolean("cleanup", False),
        store=tuple(suffix.lstrip(".") for suffix in store),
        # Expanded on the compute node, so it may contain $VARIABLES.
        scratch_dir=config.string("scratch_dir", None),
        max_restarts=max_restarts,
        overwrite=overwrite,
        resume=resume,
        local_max_parallel_jobs=local.integer("max_parallel_jobs", 1, minimum=1),
        slurm=load_slurm_config(config) if scheduler == "slurm" else None,
    )
    return config, common
