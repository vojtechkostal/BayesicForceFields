from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

from ..domain.bias import BiasSpec
from ..io.utils import load_yaml
from ..qoi.routines import AnalysisRoutineConfig, normalize_analysis_config


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
    kind: str = "file",
) -> Path | None:
    if path is None:
        return None
    return _resolve_path(base_dir, path, kind=kind)


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
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ValueError("'implicit_atoms' must be a string or a list of strings.")


@dataclass(frozen=True)
class RunsimsGromacsConfig:
    fn_topol: list[Path]
    fn_coordinates: list[Path]
    fn_mdp_em: list[Path | None]
    fn_mdp_prod: list[Path]
    fn_ndx: list[Path]
    fn_bias: list[Path | None]
    n_steps: list[int]

    @property
    def n_systems(self) -> int:
        return len(self.fn_topol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fn_topol": [str(path) for path in self.fn_topol],
            "fn_coordinates": [str(path) for path in self.fn_coordinates],
            "fn_mdp_em": [None if path is None else str(path) for path in self.fn_mdp_em],
            "fn_mdp_prod": [str(path) for path in self.fn_mdp_prod],
            "fn_ndx": [str(path) for path in self.fn_ndx],
            "fn_bias": [
                None if path is None else str(path) for path in self.fn_bias
            ],
            "n_steps": list(self.n_steps),
        }


@dataclass(frozen=True)
class RunsimsConfig:
    fn_config: Path
    data_dir: Path
    gmx_cmd: str
    job_scheduler: SchedulerName
    gromacs: RunsimsGromacsConfig
    fn_specs: Optional[Path] = None
    inputs: Optional[Path] = None
    mol_resname: Optional[str] = None
    bounds: Optional[dict[str, tuple[float, float]]] = None
    total_charge: Optional[float] = None
    implicit_atoms: Optional[list[str]] = None
    n_samples: Optional[int] = None
    compress: bool = False
    cleanup: bool = False
    store: tuple[str, ...] = ()
    slurm: Optional[dict[str, Any]] = None

    @property
    def validate(self) -> bool:
        return self.inputs is not None

    @classmethod
    def load(cls, fn_config: PathLike) -> "RunsimsConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        required = ["data_dir", "gromacs", "job_scheduler", "gmx_cmd"]
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

        gromacs = config["gromacs"]
        gmx_required = [
            "fn_topol",
            "fn_coordinates",
            "fn_mdp_em",
            "fn_mdp_prod",
            "fn_ndx",
            "fn_bias",
            "n_steps",
        ]
        missing = [key for key in gmx_required if key not in gromacs]
        if missing:
            raise ValueError(
                "Missing required GROMACS option(s): "
                + ", ".join(repr(key) for key in missing)
            )

        n_systems = _sequence_length(gromacs["fn_topol"])
        gromacs["fn_topol"] = _normalize_sequence(
            gromacs["fn_topol"],
            n_systems,
            key="gromacs.fn_topol",
        )
        for key in (
            "fn_coordinates",
            "fn_mdp_em",
            "fn_mdp_prod",
            "fn_ndx",
            "n_steps",
        ):
            gromacs[key] = _normalize_sequence(
                gromacs[key],
                n_systems,
                key=f"gromacs.{key}",
            )
        for key in ("fn_bias",):
            gromacs[key] = _normalize_sequence(
                gromacs.get(key),
                n_systems,
                key=f"gromacs.{key}",
            )

        gmx_config = RunsimsGromacsConfig(
            fn_topol=_resolve_paths(base_dir, gromacs["fn_topol"], key="gromacs.fn_topol"),
            fn_coordinates=_resolve_paths(
                base_dir,
                gromacs["fn_coordinates"],
                key="gromacs.fn_coordinates",
            ),
            fn_mdp_em=[
                _resolve_optional_path(base_dir, value, kind="gromacs.fn_mdp_em file")
                for value in gromacs["fn_mdp_em"]
            ],
            fn_mdp_prod=_resolve_paths(
                base_dir,
                gromacs["fn_mdp_prod"],
                key="gromacs.fn_mdp_prod",
            ),
            fn_ndx=_resolve_paths(base_dir, gromacs["fn_ndx"], key="gromacs.fn_ndx"),
            fn_bias=[
                _resolve_optional_path(base_dir, value, kind="gromacs.fn_bias file")
                for value in gromacs["fn_bias"]
            ],
            n_steps=[int(value) for value in gromacs["n_steps"]],
        )

        if any(step <= 0 for step in gmx_config.n_steps):
            raise ValueError("All entries in gromacs.n_steps must be positive.")

        inputs = config.get("inputs")
        if inputs is not None:
            fn_specs = config.get("fn_specs")
            if fn_specs is None:
                raise ValueError("Validation mode requires 'fn_specs'.")
            resolved_specs = _resolve_path(base_dir, fn_specs, kind="specs file")
            resolved_inputs = _resolve_path(base_dir, inputs, kind="inputs file")

            bounds = None
            total_charge = None
            implicit_atoms = None
            mol_resname = None
            n_samples = None
        else:
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
                    "Training mode requires configuration key(s): "
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

            resolved_specs = None
            resolved_inputs = None
            total_charge = float(config["total_charge"])
            mol_resname = str(config["mol_resname"])

        slurm = None
        if scheduler == "slurm":
            slurm = config.get("slurm")
            if not isinstance(slurm, dict):
                raise ValueError("Missing 'slurm' configuration for slurm scheduler.")
            for key in ("preamble", "commands"):
                if key not in slurm:
                    raise ValueError(f"Scheduler 'slurm' must define {key!r}.")
            if not isinstance(slurm["commands"], list) or not all(
                isinstance(cmd, str) for cmd in slurm["commands"]
            ):
                raise ValueError("slurm.commands must be a list of shell commands.")

        return cls(
            fn_config=fn_config,
            data_dir=_resolve_path(base_dir, config["data_dir"], must_exist=False, kind="data directory"),
            gmx_cmd=str(config["gmx_cmd"]),
            job_scheduler=scheduler,
            gromacs=gmx_config,
            fn_specs=resolved_specs,
            inputs=resolved_inputs,
            mol_resname=mol_resname,
            bounds=bounds,
            total_charge=total_charge,
            implicit_atoms=implicit_atoms,
            n_samples=n_samples,
            compress=bool(config.get("compress", False)),
            cleanup=bool(config.get("cleanup", False)),
            store=tuple(_normalize_store(config.get("store"))),
            slurm=slurm,
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
class AnalyzeConfig:
    fn_config: Path
    aimd_fn_coord: list[Path]
    aimd_fn_topol: list[Path]
    aimd_fn_trj: list[Path]
    aimd_start: int = 0
    aimd_stop: int = -1
    aimd_step: int = 1
    trainset_dir: Path = Path(".")
    ffmd_start: int = 1
    ffmd_stop: Optional[int] = None
    ffmd_step: int = 1
    ffmd_workers: int = -1
    ffmd_progress_stride: int = 10
    analysis: AnalysisRoutineConfig | None = None
    base_name: Path = Path("./qoi")
    fn_log: Path = Path("./out.log")
    write_raw_qoi: bool = False

    @classmethod
    def load(cls, fn_config: PathLike) -> "AnalyzeConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ("aimd", "ffmd"):
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key!r}.")

        aimd = config["aimd"]
        aimd_keys = ("fn_coord", "fn_topol", "fn_trj")
        missing = [key for key in aimd_keys if key not in aimd]
        if missing:
            raise ValueError(
                "Missing required AIMD option(s): "
                + ", ".join(repr(key) for key in missing)
            )

        n_systems = _sequence_length(aimd["fn_coord"])
        aimd["fn_coord"] = _normalize_sequence(
            aimd["fn_coord"],
            n_systems,
            key="aimd.fn_coord",
        )
        for key in aimd_keys[1:]:
            aimd[key] = _normalize_sequence(aimd[key], n_systems, key=f"aimd.{key}")

        ffmd = config["ffmd"]
        trainset_dir = ffmd.get("trainset_dir")
        if trainset_dir is None:
            raise ValueError("Missing 'trainset_dir' in FFMD configuration.")

        if "analysis" not in config:
            raise ValueError("Missing required configuration section: 'analysis'.")

        analysis = normalize_analysis_config(config["analysis"])

        return cls(
            fn_config=fn_config,
            aimd_fn_coord=_resolve_paths(base_dir, aimd["fn_coord"], key="aimd.fn_coord"),
            aimd_fn_topol=_resolve_paths(base_dir, aimd["fn_topol"], key="aimd.fn_topol"),
            aimd_fn_trj=_resolve_paths(base_dir, aimd["fn_trj"], key="aimd.fn_trj"),
            aimd_start=int(aimd.get("start", 0)),
            aimd_stop=aimd.get("stop", -1),
            aimd_step=int(aimd.get("step", 1)),
            trainset_dir=_resolve_path(base_dir, trainset_dir, kind="trainset directory"),
            ffmd_start=int(ffmd.get("start", 1)),
            ffmd_stop=ffmd.get("stop"),
            ffmd_step=int(ffmd.get("step", 1)),
            ffmd_workers=int(ffmd.get("workers", -1)),
            ffmd_progress_stride=int(ffmd.get("progress_stride", 10)),
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
class MDJobConfig:
    fn_config: Path
    hash: str
    params: list[float]
    data_dir: Path
    fn_specs: Optional[Path]
    gmx_cmd: str
    job_scheduler: SchedulerName
    store: tuple[str, ...]
    run: bool
    gromacs: RunsimsGromacsConfig

    @classmethod
    def load(cls, fn_config: PathLike) -> "MDJobConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        required = ["hash", "params", "data_dir", "gmx_cmd", "job_scheduler", "gromacs"]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                "Missing required MD job option(s): "
                + ", ".join(repr(key) for key in missing)
            )

        gromacs = config["gromacs"]
        gmx_required = [
            "fn_topol",
            "fn_coordinates",
            "fn_mdp_em",
            "fn_mdp_prod",
            "fn_ndx",
            "fn_bias",
            "n_steps",
        ]
        missing = [key for key in gmx_required if key not in gromacs]
        if missing:
            raise ValueError(
                "Missing required GROMACS option(s) in MD job config: "
                + ", ".join(repr(key) for key in missing)
            )

        n_systems = _sequence_length(gromacs["fn_topol"])
        gromacs["fn_topol"] = _normalize_sequence(
            gromacs["fn_topol"],
            n_systems,
            key="gromacs.fn_topol",
        )
        for key in (
            "fn_coordinates",
            "fn_mdp_em",
            "fn_mdp_prod",
            "fn_ndx",
            "n_steps",
        ):
            gromacs[key] = _normalize_sequence(
                gromacs[key],
                n_systems,
                key=f"gromacs.{key}",
            )
        for key in ("fn_bias",):
            gromacs[key] = _normalize_sequence(
                gromacs.get(key),
                n_systems,
                key=f"gromacs.{key}",
            )

        job = cls(
            fn_config=fn_config,
            hash=str(config["hash"]),
            params=[float(value) for value in config["params"]],
            data_dir=_resolve_path(base_dir, config["data_dir"], kind="data directory"),
            fn_specs=_resolve_optional_path(base_dir, config.get("fn_specs"), kind="specs file"),
            gmx_cmd=str(config["gmx_cmd"]),
            job_scheduler=config["job_scheduler"],
            store=tuple(_normalize_store(config.get("store"))),
            run=bool(config.get("run", True)),
            gromacs=RunsimsGromacsConfig(
                fn_topol=_resolve_paths(base_dir, gromacs["fn_topol"], key="gromacs.fn_topol"),
                fn_coordinates=_resolve_paths(
                    base_dir,
                    gromacs["fn_coordinates"],
                    key="gromacs.fn_coordinates",
                ),
                fn_mdp_em=[
                    _resolve_optional_path(base_dir, value, kind="gromacs.fn_mdp_em file")
                    for value in gromacs["fn_mdp_em"]
                ],
                fn_mdp_prod=_resolve_paths(
                    base_dir,
                    gromacs["fn_mdp_prod"],
                    key="gromacs.fn_mdp_prod",
                ),
                fn_ndx=_resolve_paths(base_dir, gromacs["fn_ndx"], key="gromacs.fn_ndx"),
                fn_bias=[
                    _resolve_optional_path(
                        base_dir,
                        value,
                        kind="gromacs.fn_bias file",
                    )
                    for value in gromacs["fn_bias"]
                ],
                n_steps=[int(value) for value in gromacs["n_steps"]],
            ),
        )
        if any(step <= 0 for step in job.gromacs.n_steps):
            raise ValueError("All entries in gromacs.n_steps must be positive.")
        return job
    
