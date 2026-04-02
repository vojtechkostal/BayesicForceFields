"""Configuration models and YAML loaders for BFF workflows."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

from ..domain.bias import BiasSpec
from ..io.utils import load_yaml
from ..qoi.routines import (
    AnalysisRoutineConfig,
    AnalysisRuntimeConfig,
    normalize_analysis_runtime_config,
    normalize_routine_list,
)


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


def _load_simulation_asset_systems(
    base_dir: Path,
    systems_raw: Any,
) -> list["SimulationSystemConfig"]:
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError("'systems' must be a non-empty list.")

    systems: list[SimulationSystemConfig] = []
    for index, system in enumerate(systems_raw):
        if not isinstance(system, dict):
            raise ValueError(f"systems[{index}] must be a mapping.")
        if "assets" not in system or "n_steps" not in system:
            raise ValueError(
                f"systems[{index}] must define 'assets' and 'n_steps'."
            )
        training_assets = _resolve_path(
            base_dir,
            system["assets"],
            kind=f"systems[{index}] assets directory",
        )
        systems.append(
            _load_prepared_simulation_system(
                training_assets,
                int(system["n_steps"]),
                system_id=f"{index:03d}",
            )
        )
    return systems


def _load_prepared_simulation_system(
    training_assets: Path,
    n_steps: int,
    *,
    system_id: str,
) -> "SimulationSystemConfig":
    if n_steps <= 0:
        raise ValueError(
            f"Prepared system assets {training_assets} have invalid n_steps={n_steps}."
        )

    topologies = sorted(training_assets.glob("window-*.top"))
    if len(topologies) != 1:
        raise ValueError(
            f"{training_assets} must contain exactly one prepared system, "
            f"but found {len(topologies)} topology files."
        )

    fn_topol = topologies[0]
    window_label = fn_topol.stem
    fn_coordinates = training_assets / f"{window_label}.gro"
    fn_mdp_em = training_assets / f"{window_label}.em.mdp"
    fn_mdp_prod = training_assets / f"{window_label}.mdp"
    fn_ndx = training_assets / f"{window_label}.ndx"
    fn_bias = None
    for suffix in ("bias.colvars.dat", "bias.plumed.dat"):
        candidate = training_assets / f"{window_label}.{suffix}"
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
                f"Prepared {kind} file not found for {window_label}: {path}"
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


def _load_slurm_config(slurm_raw: Any) -> "SlurmConfig":
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


def _load_campaign_common(
    fn_config: PathLike,
    *,
    asset_systems: bool,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    fn_config = Path(fn_config).resolve()
    base_dir = fn_config.parent
    config = load_yaml(fn_config)

    required = ["trainset_dir", "systems", "job_scheduler", "gmx_cmd"]
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

    if asset_systems:
        systems = _load_simulation_asset_systems(base_dir, config["systems"])
    else:
        systems = _load_simulation_systems(base_dir, config["systems"], key="systems")

    slurm = None
    if scheduler == "slurm":
        slurm = _load_slurm_config(config.get("slurm"))

    common = dict(
        fn_config=fn_config,
        trainset_dir=_resolve_path(
            base_dir,
            config["trainset_dir"],
            must_exist=False,
            kind="trainset directory",
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


@dataclass(frozen=True)
class SlurmConfig:
    max_parallel_jobs: int = 1
    sbatch: dict[str, Any] | None = None
    setup: tuple[str, ...] = ()
    teardown: tuple[str, ...] = ()


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


@dataclass(frozen=True, kw_only=True)
class SimulationCampaignConfig:
    fn_config: Path
    trainset_dir: Path
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
        return _load_campaign_common(fn_config, asset_systems=False)


@dataclass(frozen=True, kw_only=True)
class SimulateConfig(SimulationCampaignConfig):
    mol_resname: str
    bounds: dict[str, tuple[float, float]]
    total_charge: float
    implicit_atoms: list[str]
    n_samples: int

    @classmethod
    def load(cls, fn_config: PathLike) -> "SimulateConfig":
        _, base_dir, config, common = _load_campaign_common(
            fn_config,
            asset_systems=True,
        )

        required = [
            "mol_resname",
            "bounds",
            "total_charge",
            "implicit_atoms",
            "n_samples",
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                "Simulation mode requires configuration key(s): "
                + ", ".join(repr(key) for key in missing)
            )

        bounds = _validate_bounds(config["bounds"])
        implicit_atoms = _normalize_implicit_atoms(config["implicit_atoms"])
        bound_charge_groups = {
            name.split(maxsplit=1)[1]
            for name in bounds
            if name.startswith("charge ")
        }
        implicit_group = " ".join(implicit_atoms)
        if bound_charge_groups and implicit_group not in bound_charge_groups:
            raise ValueError(
                f"'implicit_atoms' ({implicit_group}) must match one of the "
                "charge parameters defined in 'bounds'."
            )

        n_samples = int(config["n_samples"])
        if n_samples <= 0:
            raise ValueError("'n_samples' must be a positive integer.")

        return cls(
            **common,
            mol_resname=str(config["mol_resname"]),
            bounds=bounds,
            total_charge=float(config["total_charge"]),
            implicit_atoms=implicit_atoms,
            n_samples=n_samples,
        )


@dataclass(frozen=True, kw_only=True)
class ValidateConfig(SimulationCampaignConfig):
    specs: Path
    parameters: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "ValidateConfig":
        _, base_dir, config, common = _load_campaign_common(
            fn_config,
            asset_systems=True,
        )

        if "specs" not in config:
            raise ValueError("Validation mode requires 'specs'.")
        if "parameters" not in config:
            raise ValueError("Validation mode requires 'parameters'.")

        return cls(
            **common,
            specs=_resolve_path(base_dir, config["specs"], kind="specs file"),
            parameters=_resolve_path(
                base_dir,
                config["parameters"],
                kind="parameter samples file",
            ),
        )


@dataclass(frozen=True)
class PrepareSystemConfig:
    fn_topol: Path
    templates: dict[str, Path]
    charge: int
    mult: int
    box: list[float] | None
    bias: BiasSpec
    nsteps_npt: int
    nsteps_prod: int
    fn_mdp_em: Path
    fn_mdp_npt: Path
    fn_mdp_prod: Path


@dataclass(frozen=True)
class PrepareConfig:
    fn_config: Path
    project_dir: Path
    gmx_cmd: str
    fn_log: Optional[Path]
    systems: list[PrepareSystemConfig]
    n_single_point_snapshots: int = 1000

    @classmethod
    def load(cls, fn_config: PathLike) -> "PrepareConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        required = ["project", "gromacs", "systems"]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                "Missing required preparation option(s): "
                + ", ".join(repr(key) for key in missing)
            )

        project = config["project"]
        if isinstance(project, str):
            project_dir = _resolve_path(
                base_dir,
                project,
                must_exist=False,
                kind="project directory",
            )
            fn_log_raw = config.get("fn_log")
        elif isinstance(project, dict):
            if "directory" not in project:
                raise ValueError("project.directory is required.")
            project_dir = _resolve_path(
                base_dir,
                project["directory"],
                must_exist=False,
                kind="project directory",
            )
            fn_log_raw = project.get("log")
        else:
            raise ValueError("'project' must be a string or mapping.")

        gromacs = config["gromacs"]
        if not isinstance(gromacs, dict):
            raise ValueError("'gromacs' must be a mapping.")
        if "command" not in gromacs:
            raise ValueError("gromacs.command is required.")
        defaults = config.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("'defaults' must be a mapping.")
        default_steps = defaults.get("nsteps", {})
        if default_steps is None:
            default_steps = {}
        if not isinstance(default_steps, dict):
            raise ValueError("'defaults.nsteps' must be a mapping.")
        nsteps_npt_default = int(default_steps.get("npt", 0))
        nsteps_prod_default = int(default_steps.get("prod", 100000))
        if nsteps_npt_default < 0 or nsteps_prod_default < 0:
            raise ValueError("defaults.nsteps values must be non-negative.")

        systems_raw = config["systems"]
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'systems' must be a non-empty list.")

        systems: list[PrepareSystemConfig] = []
        for i, system in enumerate(systems_raw):
            if not isinstance(system, dict):
                raise ValueError(f"System {i} must be a mapping.")
            for key in ("topology", "templates", "charge", "multiplicity"):
                if key not in system:
                    raise ValueError(f"System {i} is missing required key {key!r}.")
            templates_raw = system["templates"]
            if not isinstance(templates_raw, dict) or not templates_raw:
                raise ValueError(f"System {i} templates must be a non-empty mapping.")
            if not all(
                isinstance(name, str) and isinstance(path, (str, Path))
                for name, path in templates_raw.items()
            ):
                raise ValueError(
                    f"System {i} templates must map residue names to file paths."
                )

            box = system.get("box")
            if box is None:
                box_values = None
            else:
                if not isinstance(box, list) or len(box) not in {3, 6}:
                    raise ValueError(
                        f"Invalid box dimensions at index {i}: {box}. "
                        "Expected 3 or 6 numeric values."
                    )
                if not all(isinstance(value, (int, float)) for value in box):
                    raise ValueError(f"Invalid box dimensions at index {i}: {box}.")
                if len(box) == 3:
                    box = [*box, 90.0, 90.0, 90.0]
                box_values = [float(value) for value in box]

            steps = system.get("nsteps", {})
            if steps is None:
                steps = {}
            if not isinstance(steps, dict):
                raise ValueError(f"System {i} nsteps must be a mapping.")
            nsteps_npt = int(steps.get("npt", nsteps_npt_default))
            nsteps_prod = int(steps.get("prod", nsteps_prod_default))
            if nsteps_npt < 0 or nsteps_prod < 0:
                raise ValueError(f"System {i} nsteps values must be non-negative.")
            if (
                box_values is not None
                and len(box_values) == 6
                and box_values[3:] == [90.0, 90.0, 90.0]
                and len(system.get("box", [])) == 3
            ):
                nsteps_npt = 0

            mdp = system.get("mdp")
            if not isinstance(mdp, dict):
                raise ValueError(f"System {i} mdp must be a mapping.")
            missing_mdp = [key for key in ("em", "npt", "prod") if key not in mdp]
            if missing_mdp:
                raise ValueError(
                    f"System {i} mdp is missing required key(s): "
                    + ", ".join(repr(key) for key in missing_mdp)
                )

            systems.append(
                PrepareSystemConfig(
                    fn_topol=_resolve_path(
                        base_dir,
                        system["topology"],
                        kind=f"system {i} topology file",
                    ),
                    templates={
                        name: _resolve_path(
                            base_dir,
                            path,
                            kind=f"system {i} template file for {name!r}",
                        )
                        for name, path in templates_raw.items()
                    },
                    charge=int(system["charge"]),
                    mult=int(system["multiplicity"]),
                    box=box_values,
                    bias=BiasSpec.from_any(system.get("bias"), base_dir=base_dir),
                    nsteps_npt=nsteps_npt,
                    nsteps_prod=nsteps_prod,
                    fn_mdp_em=_resolve_path(
                        base_dir,
                        mdp["em"],
                        kind=f"system {i} em mdp file",
                    ),
                    fn_mdp_npt=_resolve_path(
                        base_dir,
                        mdp["npt"],
                        kind=f"system {i} npt mdp file",
                    ),
                    fn_mdp_prod=_resolve_path(
                        base_dir,
                        mdp["prod"],
                        kind=f"system {i} production mdp file",
                    ),
                )
            )

        reference = config.get("reference", {})
        if reference is None:
            reference = {}
        if not isinstance(reference, dict):
            raise ValueError("'reference' must be a mapping.")

        resolved_log = None if fn_log_raw is None else _resolve_path(
            base_dir,
            fn_log_raw,
            must_exist=False,
            kind="log file",
        )
        n_single_point_snapshots = int(
            reference.get("n_single_point_snapshots", 1000)
        )
        if n_single_point_snapshots <= 0:
            raise ValueError(
                "'reference.n_single_point_snapshots' must be a positive integer."
            )
        return cls(
            fn_config=fn_config,
            project_dir=project_dir,
            gmx_cmd=str(gromacs["command"]),
            fn_log=resolved_log,
            systems=systems,
            n_single_point_snapshots=n_single_point_snapshots,
        )


@dataclass(frozen=True)
class AnalyzeSystemConfig:
    fn_coord: Path
    fn_topol: Path
    fn_trj: Path
    routines: tuple[AnalysisRoutineConfig, ...]


@dataclass(frozen=True)
class AnalyzeTrainsetConfig:
    dir: Path
    start: int = 1
    stop: Optional[int] = None
    step: int = 1
    workers: int = -1
    progress_stride: int = 10


@dataclass(frozen=True)
class AnalyzeRefsetConfig:
    systems: list[AnalyzeSystemConfig]
    start: int = 0
    stop: int = -1
    step: int = 1


@dataclass(frozen=True)
class AnalyzeOutputConfig:
    path: Path = Path("./qoi")
    log: Path = Path("./out.log")
    write_raw: bool = False


@dataclass(frozen=True)
class QoIConfig:
    fn_config: Path
    trainset: AnalyzeTrainsetConfig
    refset: AnalyzeRefsetConfig
    run: AnalysisRuntimeConfig = AnalysisRuntimeConfig()
    output: AnalyzeOutputConfig = AnalyzeOutputConfig()

    @classmethod
    def load(cls, fn_config: PathLike) -> "QoIConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ("trainset", "refset"):
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key!r}.")

        trainset = config["trainset"]
        if not isinstance(trainset, Mapping):
            raise ValueError("'trainset' must be a mapping.")
        trainset_dir = trainset.get("dir")
        if trainset_dir is None:
            raise ValueError("Missing 'dir' in trainset configuration.")

        refset = config["refset"]
        if not isinstance(refset, Mapping):
            raise ValueError("'refset' must be a mapping.")
        systems_raw = refset.get("systems")
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'refset.systems' must be a non-empty list.")

        systems: list[AnalyzeSystemConfig] = []
        for i, system in enumerate(systems_raw):
            if not isinstance(system, Mapping):
                raise ValueError(f"refset.systems[{i}] must be a mapping.")
            for key in ("coordinates", "topology", "trajectory"):
                if key not in system:
                    raise ValueError(
                        f"refset.systems[{i}] is missing required key {key!r}."
                    )
            routines = system.get("routines")
            if not isinstance(routines, list) or not routines:
                raise ValueError(
                    f"refset.systems[{i}].routines must be a non-empty list."
                )

            systems.append(
                AnalyzeSystemConfig(
                    fn_coord=_resolve_path(
                        base_dir,
                        system["coordinates"],
                        kind=f"refset.systems[{i}] coordinates file",
                    ),
                    fn_topol=_resolve_path(
                        base_dir,
                        system["topology"],
                        kind=f"refset.systems[{i}] topology file",
                    ),
                    fn_trj=_resolve_path(
                        base_dir,
                        system["trajectory"],
                        kind=f"refset.systems[{i}] trajectory file",
                    ),
                    routines=normalize_routine_list(routines, base_dir=base_dir),
                )
            )

        run = normalize_analysis_runtime_config(config.get("run"))
        if "analysis" in config:
            raise ValueError("Use 'run' instead of 'analysis' in QoI config.")
        output = config.get("output", {})
        if not isinstance(output, Mapping):
            raise ValueError("'output' must be a mapping.")

        return cls(
            fn_config=fn_config,
            trainset=AnalyzeTrainsetConfig(
                dir=_resolve_path(
                    base_dir,
                    trainset_dir,
                    kind="trainset directory",
                ),
                start=int(trainset.get("start", 1)),
                stop=trainset.get("stop"),
                step=int(trainset.get("step", 1)),
                workers=int(trainset.get("workers", -1)),
                progress_stride=int(trainset.get("progress_stride", 10)),
            ),
            refset=AnalyzeRefsetConfig(
                systems=systems,
                start=int(refset.get("start", 0)),
                stop=refset.get("stop", -1),
                step=int(refset.get("step", 1)),
            ),
            run=run,
            output=AnalyzeOutputConfig(
                path=_resolve_path(
                    base_dir,
                    output.get("path", "./qoi"),
                    must_exist=False,
                    kind="QoI output path",
                ),
                log=_resolve_path(
                    base_dir,
                    output.get("log", "./out.log"),
                    must_exist=False,
                    kind="log file",
                ),
                write_raw=bool(output.get("write_raw", False)),
            ),
        )


@dataclass(frozen=True)
class TrainDatasetConfig:
    name: str
    fn_data: Path
    mean: Any = 0
    nuisance: float | None = None
    observation_scale: float = 1.0
    fn_model: Path | None = None


@dataclass(frozen=True)
class TrainOptionsConfig:
    model_dir: Path
    reuse_models: bool = True
    n_hyper_max: int = 200
    committee_size: int = 1
    test_fraction: float = 0.2
    device: str = "cuda"
    opt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    fn_config: Path
    datasets: tuple[TrainDatasetConfig, ...]
    training: TrainOptionsConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "TrainConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ("datasets", "training"):
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key!r}.")

        datasets_raw = config["datasets"]
        if not isinstance(datasets_raw, Mapping) or not datasets_raw:
            raise ValueError("'datasets' must be a non-empty mapping.")

        training = config["training"]
        if not isinstance(training, Mapping):
            raise ValueError("'training' must be a mapping.")

        model_dir = _resolve_path(
            base_dir,
            training.get("model_dir", "./models"),
            must_exist=False,
            kind="model directory",
        )

        training_known_keys = {
            "model_dir",
            "reuse_models",
            "n_hyper_max",
            "committee_size",
            "test_fraction",
            "device",
        }
        opt_kwargs = {
            key: value
            for key, value in training.items()
            if key not in training_known_keys
        }
        training_config = TrainOptionsConfig(
            model_dir=model_dir,
            reuse_models=bool(training.get("reuse_models", True)),
            n_hyper_max=int(training.get("n_hyper_max", 200)),
            committee_size=int(training.get("committee_size", 1)),
            test_fraction=float(training.get("test_fraction", 0.2)),
            device=str(training.get("device", "cuda")),
            opt_kwargs=opt_kwargs,
        )

        if not (0 < training_config.test_fraction < 1):
            raise ValueError("'training.test_fraction' must be between 0 and 1.")
        if training_config.n_hyper_max < 1:
            raise ValueError("'training.n_hyper_max' must be positive.")
        if training_config.committee_size < 1:
            raise ValueError("'training.committee_size' must be positive.")

        datasets: list[TrainDatasetConfig] = []
        for name, dataset in datasets_raw.items():
            if not isinstance(dataset, Mapping):
                raise ValueError(f"Dataset {name!r} must be a mapping.")
            if "data" not in dataset:
                raise ValueError(f"Dataset {name!r} is missing required key 'data'.")
            nuisance = dataset.get("nuisance")
            if nuisance is not None:
                nuisance = float(nuisance)
                if nuisance <= 0:
                    raise ValueError(
                        "Dataset "
                        f"{name!r} nuisance must be a positive standard deviation."
                    )
            observation_scale = float(dataset.get("observation_scale", 1.0))
            if observation_scale <= 0:
                raise ValueError(
                    f"Dataset {name!r} observation_scale must be positive."
                )
            fn_model = dataset.get("model")
            if fn_model is None:
                fn_model_resolved = model_dir / f"{name}.lgp"
            else:
                fn_model_resolved = _resolve_path(
                    base_dir,
                    fn_model,
                    must_exist=False,
                    kind=f"dataset {name!r} model file",
                )
            datasets.append(
                TrainDatasetConfig(
                    name=str(name),
                    fn_data=_resolve_path(
                        base_dir,
                        dataset["data"],
                        kind=f"dataset {name!r} data file",
                    ),
                    mean=dataset.get("mean", 0),
                    nuisance=nuisance,
                    observation_scale=observation_scale,
                    fn_model=fn_model_resolved,
                )
            )

        log = _resolve_path(
            base_dir,
            config.get("log", "./out.log"),
            must_exist=False,
            kind="log file",
        )

        return cls(
            fn_config=fn_config,
            datasets=tuple(datasets),
            training=training_config,
            log=log,
        )


@dataclass(frozen=True)
class LearnMCMCConfig:
    priors_disttype: str = "normal"
    total_steps: int = 1500
    warmup: int = 500
    thin: int = 1
    progress_stride: int = 100
    n_walkers: int | None = None
    checkpoint: Path | None = None
    posterior: Path = Path("./posterior.pt")
    priors: Path | None = Path("./priors.pt")
    restart: bool = True
    device: str = "cuda"
    rhat_tol: float = 1.01
    ess_min: int = 100
    include_implicit_charge: bool = False


@dataclass(frozen=True)
class LearnConfig:
    fn_config: Path
    specs: Path
    models: dict[str, Path]
    mcmc: LearnMCMCConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "LearnConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ("specs", "models", "mcmc"):
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key!r}.")

        mcmc = config["mcmc"]
        if not isinstance(mcmc, Mapping):
            raise ValueError("'mcmc' must be a mapping.")

        total_steps = int(mcmc.get("total_steps", 1500))
        warmup = int(mcmc.get("warmup", 500))
        thin = int(mcmc.get("thin", 1))
        progress_stride = int(mcmc.get("progress_stride", 100))
        if total_steps < 1:
            raise ValueError("'mcmc.total_steps' must be positive.")
        if warmup < 0 or warmup >= total_steps:
            raise ValueError("'mcmc.warmup' must satisfy 0 <= warmup < total_steps.")
        if thin < 1:
            raise ValueError("'mcmc.thin' must be positive.")
        if progress_stride < 1:
            raise ValueError("'mcmc.progress_stride' must be positive.")

        mcmc_config = LearnMCMCConfig(
            priors_disttype=str(mcmc.get("priors_disttype", "normal")),
            total_steps=total_steps,
            warmup=warmup,
            thin=thin,
            progress_stride=progress_stride,
            n_walkers=None
            if mcmc.get("n_walkers") is None
            else int(mcmc["n_walkers"]),
            checkpoint=_resolve_optional_path(
                base_dir,
                mcmc.get("checkpoint", "./mcmc-checkpoint.pt"),
                must_exist=False,
                kind="MCMC checkpoint file",
            ),
            posterior=_resolve_path(
                base_dir,
                mcmc.get("posterior", "./posterior.pt"),
                must_exist=False,
                kind="posterior output file",
            ),
            priors=_resolve_optional_path(
                base_dir,
                mcmc.get("priors", "./priors.pt"),
                must_exist=False,
                kind="priors output file",
            ),
            restart=bool(mcmc.get("restart", True)),
            device=str(mcmc.get("device", "cuda")),
            rhat_tol=float(mcmc.get("rhat_tol", 1.01)),
            ess_min=int(mcmc.get("ess_min", 100)),
            include_implicit_charge=bool(
                mcmc.get("include_implicit_charge", False)
            ),
        )

        log = _resolve_path(
            base_dir,
            config.get("log", "./out.log"),
            must_exist=False,
            kind="log file",
        )

        return cls(
            fn_config=fn_config,
            specs=_resolve_path(base_dir, config["specs"], kind="specs file"),
            models=_load_model_paths(base_dir, config["models"]),
            mcmc=mcmc_config,
            log=log,
        )


@dataclass(frozen=True)
class MDJobConfig:
    fn_config: Path
    sample_id: str
    params: list[float]
    trainset_dir: Path
    fn_specs: Optional[Path]
    gmx_cmd: str
    job_scheduler: SchedulerName
    store: tuple[str, ...]
    run: bool
    systems: list[SimulationSystemConfig]

    @classmethod
    def load(cls, fn_config: PathLike) -> "MDJobConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        required = [
            "sample_id",
            "params",
            "trainset_dir",
            "gmx_cmd",
            "job_scheduler",
            "systems",
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                "Missing required MD job option(s): "
                + ", ".join(repr(key) for key in missing)
            )

        job = cls(
            fn_config=fn_config,
            sample_id=str(config["sample_id"]),
            params=[float(value) for value in config["params"]],
            trainset_dir=_resolve_path(
                base_dir,
                config["trainset_dir"],
                kind="trainset directory",
            ),
            fn_specs=_resolve_optional_path(
                base_dir,
                config.get("fn_specs"),
                kind="specs file",
            ),
            gmx_cmd=str(config["gmx_cmd"]),
            job_scheduler=config["job_scheduler"],
            store=tuple(_normalize_store(config.get("store"))),
            run=bool(config.get("run", True)),
            systems=_load_simulation_systems(
                base_dir,
                config["systems"],
                key="systems",
            ),
        )
        return job
