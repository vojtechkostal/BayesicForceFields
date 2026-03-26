from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from ..domain.bias import BiasSpec
from ..io.utils import load_yaml
from ..qoi.routines import AnalysisRoutinesConfig, normalize_analysis_config


PathLike = Union[str, Path]
SchedulerName = Literal["local", "slurm"]


def _sequence_length(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 1


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


def _normalize_sequence(
    value: Any,
    length: int,
    *,
    key: str,
) -> list[Any]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value] * length

    if len(items) != length:
        raise ValueError(
            f"Inconsistent list length for {key}: expected {length}, "
            f"got {len(items)}."
        )
    return items


def _resolve_paths(
    base_dir: Path,
    values: Sequence[PathLike],
    *,
    key: str,
) -> list[Path]:
    return [_resolve_path(base_dir, value, kind=f"{key} file") for value in values]


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


def _default_mdp_path(name: str) -> Path:
    return (Path(__file__).resolve().parents[2] / "data" / "mdp" / name).resolve()


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

        systems = _load_simulation_systems(base_dir, config["systems"], key="systems")

        slurm = None
        if scheduler == "slurm":
            slurm_raw = config.get("slurm")
            if not isinstance(slurm_raw, dict):
                raise ValueError("Missing 'slurm' configuration for slurm scheduler.")
            if "sbatch" not in slurm_raw:
                raise ValueError("Scheduler 'slurm' must define 'sbatch'.")
            if not isinstance(slurm_raw["sbatch"], dict):
                raise ValueError("slurm.sbatch must be a mapping.")

            setup = slurm_raw.get("setup", [])
            teardown = slurm_raw.get("teardown", [])
            if not isinstance(setup, list) or not all(
                isinstance(cmd, str) for cmd in setup
            ):
                raise ValueError("slurm.setup must be a list of shell commands.")
            if not isinstance(teardown, list) or not all(
                isinstance(cmd, str) for cmd in teardown
            ):
                raise ValueError("slurm.teardown must be a list of shell commands.")

            max_parallel_jobs = int(slurm_raw.get("max_parallel_jobs", 1))
            if max_parallel_jobs == 0 or max_parallel_jobs < -1:
                raise ValueError(
                    "'slurm.max_parallel_jobs' must be positive or -1."
                )

            slurm = SlurmConfig(
                max_parallel_jobs=max_parallel_jobs,
                sbatch=dict(slurm_raw["sbatch"]),
                setup=tuple(setup),
                teardown=tuple(teardown),
            )

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


@dataclass(frozen=True, kw_only=True)
class SimulateConfig(SimulationCampaignConfig):
    mol_resname: str
    bounds: dict[str, tuple[float, float]]
    total_charge: float
    implicit_atoms: list[str]
    n_samples: int

    @classmethod
    def load(cls, fn_config: PathLike) -> "SimulateConfig":
        _, _, config, common = cls._load_common(fn_config)

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
    fn_specs: Path
    fn_samples: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "ValidateConfig":
        _, base_dir, config, common = cls._load_common(fn_config)

        if "fn_specs" not in config:
            raise ValueError("Validation mode requires 'fn_specs'.")
        sample_keys = [key for key in ("samples", "inputs") if key in config]
        if not sample_keys:
            raise ValueError("Validation mode requires 'samples' or legacy 'inputs'.")
        if len(sample_keys) > 1:
            raise ValueError(
                "Validation mode accepts only one of 'samples' or legacy 'inputs'."
            )
        sample_key = sample_keys[0]

        return cls(
            **common,
            fn_specs=_resolve_path(base_dir, config["fn_specs"], kind="specs file"),
            fn_samples=_resolve_path(
                base_dir,
                config[sample_key],
                kind="parameter samples file",
            ),
        )


@dataclass(frozen=True)
class PrepareSystemConfig:
    fn_topol: Path
    fn_mol: Path
    charge: int
    mult: int
    box: list[float] | None
    bias: BiasSpec
    nsteps_npt: int
    nsteps_nvt: int
    fn_mdp_em: Path
    fn_mdp_npt: Path
    fn_mdp_nvt: Path


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
        mdp_defaults = gromacs.get("mdp", {})
        if mdp_defaults is None:
            mdp_defaults = {}
        if not isinstance(mdp_defaults, dict):
            raise ValueError("'gromacs.mdp' must be a mapping.")

        fn_mdp_em_default = _resolve_path(
            base_dir,
            mdp_defaults.get("em", _default_mdp_path("em.mdp")),
            kind="gromacs.mdp.em file",
        )
        fn_mdp_npt_default = _resolve_path(
            base_dir,
            mdp_defaults.get("npt", _default_mdp_path("npt.mdp")),
            kind="gromacs.mdp.npt file",
        )
        fn_mdp_nvt_default = _resolve_path(
            base_dir,
            mdp_defaults.get("nvt", _default_mdp_path("nvt.mdp")),
            kind="gromacs.mdp.nvt file",
        )

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
        nsteps_nvt_default = int(default_steps.get("nvt", 100000))
        if nsteps_npt_default < 0 or nsteps_nvt_default < 0:
            raise ValueError("defaults.nsteps values must be non-negative.")

        systems_raw = config["systems"]
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'systems' must be a non-empty list.")

        systems: list[PrepareSystemConfig] = []
        for i, system in enumerate(systems_raw):
            if not isinstance(system, dict):
                raise ValueError(f"System {i} must be a mapping.")
            for key in ("topology", "coordinates", "charge", "multiplicity"):
                if key not in system:
                    raise ValueError(f"System {i} is missing required key {key!r}.")

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
            nsteps_nvt = int(steps.get("nvt", nsteps_nvt_default))
            if nsteps_npt < 0 or nsteps_nvt < 0:
                raise ValueError(f"System {i} nsteps values must be non-negative.")
            if (
                box_values is not None
                and len(box_values) == 6
                and box_values[3:] == [90.0, 90.0, 90.0]
                and len(system.get("box", [])) == 3
            ):
                nsteps_npt = 0

            mdp = system.get("mdp", {})
            if mdp is None:
                mdp = {}
            if not isinstance(mdp, dict):
                raise ValueError(f"System {i} mdp must be a mapping.")

            systems.append(
                PrepareSystemConfig(
                    fn_topol=_resolve_path(
                        base_dir,
                        system["topology"],
                        kind=f"system {i} topology file",
                    ),
                    fn_mol=_resolve_path(
                        base_dir,
                        system["coordinates"],
                        kind=f"system {i} coordinate file",
                    ),
                    charge=int(system["charge"]),
                    mult=int(system["multiplicity"]),
                    box=box_values,
                    bias=BiasSpec.from_any(system.get("bias"), base_dir=base_dir),
                    nsteps_npt=nsteps_npt,
                    nsteps_nvt=nsteps_nvt,
                    fn_mdp_em=_resolve_path(
                        base_dir,
                        mdp.get("em", fn_mdp_em_default),
                        kind=f"system {i} em mdp file",
                    ),
                    fn_mdp_npt=_resolve_path(
                        base_dir,
                        mdp.get("npt", fn_mdp_npt_default),
                        kind=f"system {i} npt mdp file",
                    ),
                    fn_mdp_nvt=_resolve_path(
                        base_dir,
                        mdp.get("nvt", fn_mdp_nvt_default),
                        kind=f"system {i} nvt mdp file",
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
        fn_mdp_em_values = {str(system.fn_mdp_em) for system in systems}
        if len(fn_mdp_em_values) != 1:
            raise ValueError(
                "prepare currently exports one shared training/em.mdp, so all "
                "systems must use the same EM MDP file."
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
    routines: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class AnalyzeConfig:
    fn_config: Path
    trainset_dir: Path
    systems: list[AnalyzeSystemConfig]
    training_start: int = 1
    training_stop: Optional[int] = None
    training_step: int = 1
    training_workers: int = -1
    training_progress_stride: int = 10
    reference_start: int = 0
    reference_stop: int = -1
    reference_step: int = 1
    analysis: AnalysisRoutinesConfig | None = None
    base_name: Path = Path("./qoi")
    fn_log: Path = Path("./out.log")
    write_raw_qoi: bool = False

    @classmethod
    def load(cls, fn_config: PathLike) -> "AnalyzeConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ("systems", "training"):
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key!r}.")

        training = config["training"]
        if not isinstance(training, Mapping):
            raise ValueError("'training' must be a mapping.")
        trainset_dir = training.get("trainset_dir")
        if trainset_dir is None:
            raise ValueError("Missing 'trainset_dir' in training configuration.")

        systems_raw = config["systems"]
        if not isinstance(systems_raw, list) or not systems_raw:
            raise ValueError("'systems' must be a non-empty list.")

        routine_systems: list[dict[str, Any]] = []
        systems: list[AnalyzeSystemConfig] = []
        for i, system in enumerate(systems_raw):
            if not isinstance(system, Mapping):
                raise ValueError(f"systems[{i}] must be a mapping.")
            reference = system.get("reference")
            if not isinstance(reference, Mapping):
                raise ValueError(f"systems[{i}].reference must be a mapping.")
            for key in ("coordinates", "topology", "trajectory"):
                if key not in reference:
                    raise ValueError(
                        f"systems[{i}].reference is missing required key {key!r}."
                    )
            routines = system.get("routines")
            if not isinstance(routines, list) or not routines:
                raise ValueError(f"systems[{i}].routines must be a non-empty list.")

            systems.append(
                AnalyzeSystemConfig(
                    fn_coord=_resolve_path(
                        base_dir,
                        reference["coordinates"],
                        kind=f"systems[{i}] reference coordinates file",
                    ),
                    fn_topol=_resolve_path(
                        base_dir,
                        reference["topology"],
                        kind=f"systems[{i}] reference topology file",
                    ),
                    fn_trj=_resolve_path(
                        base_dir,
                        reference["trajectory"],
                        kind=f"systems[{i}] reference trajectory file",
                    ),
                    routines=tuple(routines),
                )
            )
            routine_systems.append({"routines": routines})

        analysis = normalize_analysis_config(
            {"systems": routine_systems, **dict(config.get("analysis", {}))},
            n_systems=len(systems),
            base_dir=base_dir,
        )

        reference = config.get("reference", {})
        if not isinstance(reference, Mapping):
            raise ValueError("'reference' must be a mapping.")

        return cls(
            fn_config=fn_config,
            trainset_dir=_resolve_path(
                base_dir,
                trainset_dir,
                kind="trainset directory",
            ),
            systems=systems,
            training_start=int(training.get("start", 1)),
            training_stop=training.get("stop"),
            training_step=int(training.get("step", 1)),
            training_workers=int(training.get("workers", -1)),
            training_progress_stride=int(training.get("progress_stride", 10)),
            reference_start=int(reference.get("start", 0)),
            reference_stop=reference.get("stop", -1),
            reference_step=int(reference.get("step", 1)),
            analysis=analysis,
            base_name=_resolve_path(
                base_dir,
                config.get("base_name", "./qoi"),
                must_exist=False,
                kind="QoI output base",
            ),
            fn_log=_resolve_path(
                base_dir,
                config.get("fn_log", "./out.log"),
                must_exist=False,
                kind="log file",
            ),
            write_raw_qoi=bool(config.get("write_raw_qoi", False)),
        )


@dataclass(frozen=True)
class LearnDatasetConfig:
    name: str
    fn_data: Path
    mean: Any = 0
    nuisance: float | None = None
    observation_scale: float = 1.0
    fn_model: Path | None = None


@dataclass(frozen=True)
class LearnTrainingConfig:
    model_dir: Path
    reuse_models: bool = True
    n_hyper_max: int = 200
    committee_size: int = 1
    test_fraction: float = 0.2
    device: str = "cuda:0"
    opt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LearnMCMCConfig:
    priors_disttype: str = "normal"
    total_steps: int = 1500
    warmup: int = 500
    thin: int = 1
    progress_stride: int = 100
    n_walkers: int | None = None
    fn_checkpoint: Path | None = None
    fn_posterior: Path = Path("./posterior.pt")
    fn_priors: Path | None = Path("./priors.pt")
    restart: bool = True
    device: str = "cuda:0"
    rhat_tol: float = 1.01
    ess_min: int = 100


@dataclass(frozen=True)
class LearnConfig:
    fn_config: Path
    fn_specs: Path
    datasets: tuple[LearnDatasetConfig, ...]
    qoi: tuple[str, ...] | None
    training: LearnTrainingConfig
    mcmc: LearnMCMCConfig
    fn_log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> "LearnConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ("fn_specs", "datasets", "training"):
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
        training_config = LearnTrainingConfig(
            model_dir=model_dir,
            reuse_models=bool(training.get("reuse_models", True)),
            n_hyper_max=int(training.get("n_hyper_max", 200)),
            committee_size=int(training.get("committee_size", 1)),
            test_fraction=float(training.get("test_fraction", 0.2)),
            device=str(training.get("device", "cuda:0")),
            opt_kwargs=opt_kwargs,
        )

        if not (0 < training_config.test_fraction < 1):
            raise ValueError("'training.test_fraction' must be between 0 and 1.")
        if training_config.n_hyper_max < 1:
            raise ValueError("'training.n_hyper_max' must be positive.")
        if training_config.committee_size < 1:
            raise ValueError("'training.committee_size' must be positive.")

        datasets: list[LearnDatasetConfig] = []
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
                LearnDatasetConfig(
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

        qoi_raw = config.get("qoi")
        qoi = None if qoi_raw is None else tuple(qoi_raw)
        if qoi is not None:
            missing = set(qoi) - {dataset.name for dataset in datasets}
            if missing:
                raise ValueError(
                    "Selected QoI(s) are missing from datasets: "
                    + ", ".join(sorted(missing))
                )

        mcmc = config.get("mcmc", {})
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
            fn_checkpoint=_resolve_optional_path(
                base_dir,
                mcmc.get("fn_checkpoint", "./mcmc-checkpoint.pt"),
                must_exist=False,
                kind="MCMC checkpoint file",
            ),
            fn_posterior=_resolve_path(
                base_dir,
                mcmc.get("fn_posterior", "./posterior.pt"),
                must_exist=False,
                kind="posterior output file",
            ),
            fn_priors=_resolve_optional_path(
                base_dir,
                mcmc.get("fn_priors", "./priors.pt"),
                must_exist=False,
                kind="priors output file",
            ),
            restart=bool(mcmc.get("restart", True)),
            device=str(mcmc.get("device", "cuda:0")),
            rhat_tol=float(mcmc.get("rhat_tol", 1.01)),
            ess_min=int(mcmc.get("ess_min", 100)),
        )

        fn_log = _resolve_path(
            base_dir,
            config.get("fn_log", "./out.log"),
            must_exist=False,
            kind="log file",
        )

        return cls(
            fn_config=fn_config,
            fn_specs=_resolve_path(base_dir, config["fn_specs"], kind="specs file"),
            datasets=tuple(datasets),
            qoi=qoi,
            training=training_config,
            mcmc=mcmc_config,
            fn_log=fn_log,
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
