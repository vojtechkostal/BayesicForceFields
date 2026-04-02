import importlib
import importlib.util
import hashlib
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .data import QoI
from .hbonds import compute_all_hbonds
from .rdf import compute_all_rdfs


def _split_routine_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            "Imported analysis routines must be specified as "
            "'module:function' or 'path/to/file.py:function'."
        )
    return spec.split(":", maxsplit=1)


def _builtin_callable(name: str) -> Callable[..., Any]:
    try:
        return BUILTIN_ROUTINES[name]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_ROUTINES))
        raise ValueError(
            f"Unknown builtin analysis routine {name!r}. "
            f"Known routines are: {known}."
        ) from exc


def _user_module_name(module_path: Path) -> str:
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    return f"bff_user_routine_{module_path.stem}_{digest}"


def _load_module_from_path(module_path: Path) -> Any:
    if not module_path.exists():
        raise ValueError(f"Analysis routine file not found: {module_path}")

    module_name = _user_module_name(module_path)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Could not load analysis routine module {module_path}.")

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _import_module(module_name: str) -> Any:
    if module_name.endswith(".py") or "/" in module_name:
        return _load_module_from_path(Path(module_name).resolve())
    return importlib.import_module(module_name)


@lru_cache(maxsize=None)
def _load_routine(spec: str) -> Callable[..., Any]:
    """Resolve a builtin or import-string routine specification."""
    if ":" not in spec:
        return _builtin_callable(spec)

    module_name, attr_name = _split_routine_spec(spec)
    module = _import_module(module_name)
    try:
        fn = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Callable {attr_name!r} was not found in module {module_name!r}."
        ) from exc
    if not callable(fn):
        raise ValueError(f"Resolved object {spec!r} is not callable.")
    return fn


BUILTIN_ROUTINES: dict[str, Callable[..., QoI]] = {
    "rdf": compute_all_rdfs,
    "hb": compute_all_hbonds,
}


@dataclass(frozen=True, slots=True)
class AnalysisRoutineConfig:
    """Configuration for a single-trajectory analysis routine.

    Attributes
    ----------
    routine
        Builtin selector or importable routine specification.
    options
        Keyword arguments forwarded to the analysis routine.
    """
    routine: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnalysisRuntimeConfig:
    """Runtime settings for the QoI analysis workflow."""

    in_memory: bool = False
    gc_collect: bool = False
    maxtasksperchild: int = 100


def _normalize_qoi_mapping(result: Mapping[str, Any]) -> dict[str, QoI]:
    """Validate and normalize a mapping of QoI outputs."""
    normalized: dict[str, QoI] = {}
    for name, value in result.items():
        if not isinstance(value, QoI):
            raise TypeError(
                "Mapping-valued analysis routines must return QoI objects."
            )
        if value.name != name:
            raise ValueError(
                f"QoI mapping key {name!r} does not match QoI.name "
                f"{value.name!r}."
            )
        normalized[name] = value
    return normalized


@dataclass(frozen=True, slots=True)
class RuntimeRoutine:
    fn: Callable[..., Any]
    options: dict[str, Any] = field(default_factory=dict)


def _normalize_routine_config(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> AnalysisRoutineConfig:
    """Validate one routine entry from the QoI configuration."""
    if not isinstance(config, Mapping):
        raise ValueError(
            "Each analysis routine specification must be a mapping with a "
            "'routine' field."
        )
    if "routine" not in config:
        raise ValueError("Each analysis routine must define 'routine'.")
    routine = config["routine"]
    if not isinstance(routine, str):
        raise ValueError("Analysis routine 'routine' must be a string.")
    if "kwargs" in config:
        raise ValueError(
            "Routine options must be provided as flat keys. "
            "Nested 'kwargs' are no longer supported."
        )
    extra_options = {
        key: value
        for key, value in config.items()
        if key != "routine"
    }
    options = dict(extra_options)

    if ":" in routine:
        module_name, attr_name = _split_routine_spec(routine)
        if module_name.endswith(".py") or "/" in module_name:
            if base_dir is None:
                module_path = Path(module_name).resolve()
            else:
                module_path = (base_dir / module_name).resolve()
            routine = f"{module_path}:{attr_name}"

    return AnalysisRoutineConfig(
        routine=routine,
        options=options,
    )


def normalize_routine_list(
    routines: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path | None = None,
) -> tuple[AnalysisRoutineConfig, ...]:
    if not isinstance(routines, Sequence) or isinstance(routines, (str, bytes)):
        raise ValueError("Analysis routines must be provided as a non-empty list.")
    if not routines:
        raise ValueError("Analysis routines must be provided as a non-empty list.")
    return tuple(
        _normalize_routine_config(routine, base_dir=base_dir)
        for routine in routines
    )


def normalize_analysis_runtime_config(
    config: Mapping[str, Any] | None,
) -> AnalysisRuntimeConfig:
    """Validate and normalize the runtime-only QoI ``run`` section."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("'run' must be a mapping.")
    if "routines" in config or "systems" in config:
        raise ValueError(
            "'run' only accepts runtime settings. "
            "Define routines under each entry in 'refset.systems'."
        )

    maxtasksperchild = int(config.get("maxtasksperchild", 100))
    if maxtasksperchild <= 0:
        raise ValueError("'run.maxtasksperchild' must be a positive integer.")

    return AnalysisRuntimeConfig(
        in_memory=bool(config.get("in_memory", False)),
        gc_collect=bool(config.get("gc_collect", False)),
        maxtasksperchild=maxtasksperchild,
    )


def run_analysis_routines(
    routines: Sequence[RuntimeRoutine],
    *,
    universe: Any,
    start: int,
    stop: int | None,
    step: int,
) -> dict[str, QoI]:
    """Run all configured routines for one trajectory and merge QoI outputs."""
    result: dict[str, QoI] = {}
    for routine in routines:
        qoi_result = routine.fn(
            universe=universe,
            start=start,
            stop=stop,
            step=step,
            **routine.options,
        )
        if isinstance(qoi_result, QoI):
            qoi_mapping = {qoi_result.name: qoi_result}
        elif isinstance(qoi_result, Mapping):
            qoi_mapping = _normalize_qoi_mapping(qoi_result)
        else:
            raise TypeError(
                "Analysis routine must return a QoI object or a mapping of "
                "QoI objects."
            )
        duplicates = set(result) & set(qoi_mapping)
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise ValueError(
                f"Analysis routines produced duplicate QoI output(s): {names}."
            )
        result.update(qoi_mapping)
    return result


def build_analysis_routines(
    systems: Sequence[Sequence[AnalysisRoutineConfig]],
) -> list[tuple[RuntimeRoutine, ...]]:
    """Build runtime analysis routines for all trajectory positions."""
    routines_by_system: list[tuple[RuntimeRoutine, ...]] = []
    for system in systems:
        runtime_routines: list[RuntimeRoutine] = []
        for routine in system:
            runtime_routines.append(
                RuntimeRoutine(
                    fn=_load_routine(routine.routine),
                    options=dict(routine.options),
                )
            )
        routines_by_system.append(tuple(runtime_routines))
    return routines_by_system
