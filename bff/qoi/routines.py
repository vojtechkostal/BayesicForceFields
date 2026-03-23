import importlib.util
import importlib
import inspect
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, TypeAlias

from .data import QoI
from .hbonds import compute_all_hbonds
from .rdf import compute_all_rdfs


FIXED_SIGNATURE = ("universe", "mol_resname", "start", "stop", "step")
FIXED_SIGNATURE_SET = set(FIXED_SIGNATURE)


def _extract_defaults(fn: Callable[..., Any]) -> dict[str, Any]:
    """Extract default keyword arguments from a callable signature.

    Parameters
    ----------
    fn
        Callable whose defaults should be extracted.

    Returns
    -------
    dict
        Mapping of parameter names to their default values.
    """
    sig = inspect.signature(fn)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def _extract_defaults_without(
    fn: Callable[..., Any],
    ignored: set[str],
) -> dict[str, Any]:
    """Extract callable defaults while omitting ignored parameters."""
    return {
        name: value
        for name, value in _extract_defaults(fn).items()
        if name not in ignored
    }


@lru_cache(maxsize=None)
def _load_callable_from_string(spec: str) -> Callable[..., Any]:
    """Resolve a builtin or import-string routine specification.

    Parameters
    ----------
    spec
        ``builtin:<name>`` selector or an import string in the form
        ``module:function``.

    Returns
    -------
    callable
        Resolved trajectory analysis routine.

    Raises
    ------
    ValueError
        If the specification cannot be resolved to a callable.
    """
    if spec.startswith("builtin:"):
        name = spec.split(":", maxsplit=1)[1]
        if name not in BUILTIN_ROUTINES:
            known = ", ".join(sorted(BUILTIN_ROUTINES))
            raise ValueError(
                f"Unknown builtin analysis routine {name!r}. "
                f"Known routines are: {known}."
            )
        return BUILTIN_ROUTINES[name]
    if ":" not in spec:
        raise ValueError(
            "Analysis routine must be a builtin name or 'module:function'."
        )

    module_name, attr_name = spec.split(":", maxsplit=1)
    if module_name.endswith(".py") or "/" in module_name:
        module_path = Path(module_name).resolve()
        if not module_path.exists():
            raise ValueError(f"Analysis routine file not found: {module_path}")
        spec_name = f"bff_user_routine_{module_path.stem}_{abs(hash(module_path))}"
        module_spec = importlib.util.spec_from_file_location(spec_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"Could not load analysis routine module {module_path}.")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    try:
        fn = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Callable {attr_name!r} was not found in module {module_name!r}."
        ) from exc
    if not callable(fn):
        raise ValueError(f"Resolved object {spec!r} is not callable.")
    return fn


def _load_callable(spec: str | Callable[..., Any]) -> Callable[..., Any]:
    """Resolve a callable while keeping string-based imports cacheable."""
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise ValueError(f"Invalid analysis routine specification: {spec!r}")
    return _load_callable_from_string(spec)


def _validate_signature(fn: Callable[..., Any], name: str) -> None:
    """Validate that a routine accepts the fixed trajectory-analysis signature.

    Parameters
    ----------
    fn
        Callable to validate.
    name
        Human-readable routine name used in error messages.

    Raises
    ------
    ValueError
        If the routine is missing one or more required arguments.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in params.values()
    )
    missing = [
        arg
        for arg in FIXED_SIGNATURE
        if arg not in params and not accepts_var_kwargs
    ]
    if missing:
        missing_text = ", ".join(repr(arg) for arg in missing)
        raise ValueError(
            f"Analysis routine {name!r} must accept the fixed arguments "
            f"{', '.join(FIXED_SIGNATURE)}. Missing: {missing_text}."
        )


def _validate_kwargs(
    fn: Callable[..., Any],
    values: Mapping[str, Any],
    *,
    ignored: set[str],
    name: str,
) -> dict[str, Any]:
    """Validate keyword arguments against an underlying callable signature.

    Parameters
    ----------
    fn
        Callable whose accepted keyword arguments should be respected.
    values
        User-supplied keyword arguments.
    ignored
        Parameter names that are intentionally supplied elsewhere and should
        be excluded from validation.
    name
        Human-readable section name for error reporting.

    Returns
    -------
    dict
        Validated keyword arguments.
    """
    sig = inspect.signature(fn)
    accepts_var_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return dict(values)

    supported = set(sig.parameters) - ignored
    unknown = set(values) - supported
    if unknown:
        unknown_text = ", ".join(sorted(unknown))
        supported_text = ", ".join(sorted(supported))
        raise ValueError(
            f"Unsupported option(s) in {name}: {unknown_text}. "
            f"Supported options are: {supported_text}."
        )
    return dict(values)


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call a function with only the keyword arguments it accepts.

    Parameters
    ----------
    fn
        Callable to invoke.
    **kwargs
        Candidate keyword arguments.

    Returns
    -------
    Any
        Callable return value.
    """
    sig = inspect.signature(fn)
    accepts_var_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return fn(**kwargs)

    supported_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name in sig.parameters
    }
    return fn(**supported_kwargs)


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
    kwargs
        Keyword arguments forwarded to the analysis routine.
    """
    routine: str | Callable[..., Any]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SystemAnalysisConfig:
    """Analysis routine list for one trajectory position within a sample."""

    routines: tuple[AnalysisRoutineConfig, ...]


@dataclass(frozen=True, slots=True)
class AnalysisRoutinesConfig:
    """Normalized analysis-routines configuration for the full workflow."""

    systems: tuple[SystemAnalysisConfig, ...]
    in_memory: bool = False
    gc_collect: bool = True
    chunksize: int = 1
    maxtasksperchild: int = 1


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


RuntimeRoutine: TypeAlias = tuple[str | Callable[..., Any], dict[str, Any]]


def _normalize_routine_config(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> AnalysisRoutineConfig:
    """Validate one routine entry from the ``analysis`` configuration.

    Parameters
    ----------
    config
        Raw routine mapping loaded from YAML.

    Returns
    -------
    AnalysisRoutineConfig
        Normalized routine configuration object.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Each analysis routine specification must be a mapping.")

    if "routine" not in config:
        raise ValueError("Each analysis routine must define 'routine'.")
    routine = config["routine"]
    kwargs = config.get("kwargs", {})

    if not isinstance(kwargs, Mapping):
        raise ValueError("Routine 'kwargs' must be a mapping.")

    if (
        isinstance(routine, str)
        and ":" in routine
        and not routine.startswith("builtin:")
    ):
        module_name, attr_name = routine.split(":", maxsplit=1)
        if module_name.endswith(".py") or "/" in module_name:
            if base_dir is None:
                module_path = Path(module_name).resolve()
            else:
                module_path = (base_dir / module_name).resolve()
            routine = f"{module_path}:{attr_name}"

    if isinstance(routine, str) and routine.startswith("builtin:"):
        builtin_name = routine.split(":", maxsplit=1)[1]
        if builtin_name not in BUILTIN_ROUTINES:
            known = ", ".join(sorted(BUILTIN_ROUTINES))
            raise ValueError(
                f"Unknown builtin analysis routine {builtin_name!r}. "
                f"Known routines are: {known}."
            )
        fn = BUILTIN_ROUTINES[builtin_name]
        kwargs = _extract_defaults_without(fn, FIXED_SIGNATURE_SET) | _validate_kwargs(
            fn,
            kwargs,
            ignored=FIXED_SIGNATURE_SET,
            name="analysis routine kwargs",
        )

    return AnalysisRoutineConfig(
        routine=routine,
        kwargs=dict(kwargs),
    )


def normalize_analysis_config(
    config: Mapping[str, Any],
    *,
    n_systems: int | None = None,
    base_dir: Path | None = None,
) -> AnalysisRoutinesConfig:
    """Validate and normalize the full ``analysis`` workflow configuration."""
    if not isinstance(config, Mapping):
        raise ValueError("'analysis' must be a mapping.")
    if "systems" not in config:
        raise ValueError("Missing required configuration section: 'analysis.systems'.")

    systems_raw = config["systems"]
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError("'analysis.systems' must be a non-empty list.")
    if n_systems is not None and len(systems_raw) != n_systems:
        raise ValueError(
            f"'analysis.systems' must contain exactly {n_systems} entries, "
            f"got {len(systems_raw)}."
        )

    systems: list[SystemAnalysisConfig] = []
    for i, system in enumerate(systems_raw):
        if not isinstance(system, Mapping):
            raise ValueError(f"analysis.systems[{i}] must be a mapping.")
        routines_raw = system.get("routines")
        if not isinstance(routines_raw, list) or not routines_raw:
            raise ValueError(
                f"analysis.systems[{i}].routines must be a non-empty list."
            )
        systems.append(
            SystemAnalysisConfig(
                routines=tuple(
                    _normalize_routine_config(routine, base_dir=base_dir)
                    for routine in routines_raw
                )
            )
        )

    chunksize = int(config.get("chunksize", 1))
    maxtasksperchild = int(config.get("maxtasksperchild", 1))
    if chunksize <= 0:
        raise ValueError("'analysis.chunksize' must be a positive integer.")
    if maxtasksperchild <= 0:
        raise ValueError("'analysis.maxtasksperchild' must be a positive integer.")

    return AnalysisRoutinesConfig(
        systems=tuple(systems),
        in_memory=bool(config.get("in_memory", False)),
        gc_collect=bool(config.get("gc_collect", True)),
        chunksize=chunksize,
        maxtasksperchild=maxtasksperchild,
    )


def run_analysis_routines(
    routines: tuple[RuntimeRoutine, ...],
    *,
    universe: Any,
    mol_resname: str,
    start: int,
    stop: int | None,
    step: int,
) -> dict[str, QoI]:
    """Run all configured routines for one trajectory and merge QoI outputs."""
    result: dict[str, QoI] = {}
    for routine_spec, kwargs in routines:
        fn = _load_callable(routine_spec)
        qoi_result = _call_with_supported_kwargs(
            fn,
            universe=universe,
            mol_resname=mol_resname,
            start=start,
            stop=stop,
            step=step,
            **kwargs,
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
    config: AnalysisRoutinesConfig,
) -> list[tuple[RuntimeRoutine, ...]]:
    """Build runtime analysis routines for all trajectory positions."""
    routines_by_system: list[tuple[RuntimeRoutine, ...]] = []
    for system in config.systems:
        routines: list[RuntimeRoutine] = []
        for routine in system.routines:
            fn = _load_callable(routine.routine)
            routine_name = (
                routine.routine
                if isinstance(routine.routine, str)
                else getattr(routine.routine, "__name__", "routine")
            )
            _validate_signature(fn, str(routine_name))
            routines.append((routine.routine, dict(routine.kwargs)))
        routines_by_system.append(tuple(routines))
    return routines_by_system
