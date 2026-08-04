"""Validated built-in and custom QoI routine dispatch."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..domain.systems import validate_system_id, validate_unique_system_ids
from ..workflows._shared.config import _strict_bool
from .data import QoI
from .hbonds import compute_hydrogen_bond_qoi
from .rdf import compute_rdf_qoi

BUILTIN_ROUTINES: dict[str, Callable[..., QoI]] = {
    "rdf": compute_rdf_qoi,
    "hydrogen_bonds": compute_hydrogen_bond_qoi,
}
BUILTIN_SELECTIONS = {
    "rdf": {"group_a", "group_b"},
    "hydrogen_bonds": {"donors", "hydrogens", "acceptors"},
}
BUILTIN_OPTIONS = {
    "rdf": {"range", "bins", "pbc", "update_selections", "smooth"},
    "hydrogen_bonds": {
        "donor_acceptor_cutoff",
        "angle_cutoff",
        "pbc",
        "update_selections",
    },
}


def _split_routine_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError(
            "Custom analysis callables must use 'module:function' or "
            "'path/to/file.py:function'."
        )
    return spec.split(":", maxsplit=1)


def _user_module_name(module_path: Path) -> str:
    digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    return f"bff_user_routine_{module_path.stem}_{digest}"


def _load_module_from_path(module_path: Path) -> Any:
    if not module_path.is_file():
        raise ValueError(f"Analysis routine file not found: {module_path}")
    module_name = _user_module_name(module_path)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load analysis routine module {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def load_custom_routine(spec: str) -> Callable[..., Any]:
    module_name, attr_name = _split_routine_spec(spec)
    module = (
        _load_module_from_path(Path(module_name).resolve())
        if module_name.endswith(".py") or "/" in module_name
        else importlib.import_module(module_name)
    )
    try:
        routine = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Callable {attr_name!r} was not found in module {module_name!r}."
        ) from exc
    if not callable(routine):
        raise ValueError(f"Resolved object {spec!r} is not callable.")
    return routine


@dataclass(frozen=True, slots=True)
class AnalysisRoutineConfig:
    name: str
    routine: str
    loader: str
    systems: tuple[str, ...]
    selections: dict[str, str] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)
    builtin: bool = False


RuntimeRoutine = AnalysisRoutineConfig


@dataclass(frozen=True, slots=True)
class AnalysisRuntimeConfig:
    in_memory: bool = True
    gc_collect: bool = False
    maxtasksperchild: int = 100


def _normalize_routine_config(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None = None,
) -> AnalysisRoutineConfig:
    if not isinstance(config, Mapping):
        raise ValueError("Each routines entry must be a mapping.")
    unknown = set(config) - {
        "name",
        "type",
        "callable",
        "loader",
        "systems",
        "selections",
        "inputs",
        "options",
    }
    if unknown:
        raise ValueError(
            "Routine contains unsupported key(s): " + ", ".join(sorted(unknown))
        )
    name = validate_system_id(config.get("name"), field="routines[].name")
    has_type = "type" in config
    has_callable = "callable" in config
    if has_type == has_callable:
        raise ValueError(
            f"Routine {config['name']!r} must define exactly one of type or callable."
        )
    systems_raw = config.get("systems")
    if not isinstance(systems_raw, list) or not systems_raw or not all(
        isinstance(value, str) for value in systems_raw
    ):
        raise ValueError(
            f"Routine {config['name']!r}.systems must be a non-empty ID list."
        )
    systems = [
        validate_system_id(
            system_id, field=f"routines.{name}.systems[{index}]"
        )
        for index, system_id in enumerate(systems_raw)
    ]
    validate_unique_system_ids(systems, field=f"routines.{name}.systems")
    options = config.get("options", {})
    selections = config.get("selections", {})
    inputs = config.get("inputs", [])
    if not isinstance(options, Mapping):
        raise ValueError(f"Routine {config['name']!r}.options must be a mapping.")
    if not isinstance(selections, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in selections.items()
    ):
        raise ValueError(
            f"Routine {config['name']!r}.selections must map names to selections."
        )
    if not isinstance(inputs, list) or not all(
        isinstance(role, str) for role in inputs
    ):
        raise ValueError(f"Routine {config['name']!r}.inputs must be a role list.")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"Routine {config['name']!r}.inputs contains duplicates.")

    if has_type:
        routine = str(config["type"])
        if routine not in BUILTIN_ROUTINES:
            raise ValueError(
                f"Routine {config['name']!r} has unknown type {routine!r}; "
                f"expected one of {sorted(BUILTIN_ROUTINES)}."
            )
        loader = config.get("loader", "mdanalysis")
        if loader != "mdanalysis":
            raise ValueError(
                f"Built-in routine {config['name']!r} requires mdanalysis."
            )
        required = BUILTIN_SELECTIONS[routine]
        if set(selections) != required:
            raise ValueError(
                f"Routine {config['name']!r}.selections requires exactly "
                f"{sorted(required)}, got {sorted(selections)}."
            )
        if inputs:
            raise ValueError(
                f"Built-in routine {config['name']!r} cannot define inputs."
            )
        unsupported_options = set(options) - BUILTIN_OPTIONS[routine]
        if unsupported_options:
            raise ValueError(
                f"Routine {name!r}.options contains unsupported key(s): "
                + ", ".join(sorted(unsupported_options))
            )
        for boolean_option in {"pbc", "update_selections", "smooth"} & set(
            options
        ):
            _strict_bool(
                options[boolean_option],
                field=f"routines.{name}.options.{boolean_option}",
            )
        if routine == "rdf":
            distance_range = options.get("range", (0.0, 10.0))
            if not (
                isinstance(distance_range, (list, tuple))
                and len(distance_range) == 2
                and all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in distance_range
                )
                and 0 <= distance_range[0] < distance_range[1]
            ):
                raise ValueError(
                    f"routines.{name}.options.range must contain two increasing "
                    f"non-negative numbers, got {distance_range!r}."
                )
            bins = options.get("bins", 200)
            if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
                raise ValueError(
                    f"routines.{name}.options.bins must be a positive integer, "
                    f"got {bins!r}."
                )
        else:
            distance_cutoff = options.get("donor_acceptor_cutoff", 3.5)
            angle_cutoff = options.get("angle_cutoff", 150.0)
            if (
                not isinstance(distance_cutoff, (int, float))
                or isinstance(distance_cutoff, bool)
                or distance_cutoff <= 0
            ):
                raise ValueError(
                    f"routines.{name}.options.donor_acceptor_cutoff must be "
                    f"positive, got {distance_cutoff!r}."
                )
            if (
                not isinstance(angle_cutoff, (int, float))
                or isinstance(angle_cutoff, bool)
                or not 0 < angle_cutoff <= 180
            ):
                raise ValueError(
                    f"routines.{name}.options.angle_cutoff must be in (0, 180], "
                    f"got {angle_cutoff!r}."
                )
        builtin = True
    else:
        routine = str(config["callable"])
        module_name, attr_name = _split_routine_spec(routine)
        if module_name.endswith(".py") or "/" in module_name:
            module_path = (
                Path(module_name).resolve()
                if base_dir is None
                else (base_dir / module_name).resolve()
            )
            routine = f"{module_path}:{attr_name}"
        load_custom_routine(routine)
        loader = config.get("loader")
        if loader not in {"files", "mdanalysis"}:
            raise ValueError(
                f"Custom routine {config['name']!r}.loader must be files or mdanalysis."
            )
        if loader == "files" and not inputs:
            raise ValueError(
                f"File routine {config['name']!r} must declare at least one input role."
            )
        if loader == "files" and selections:
            raise ValueError(
                f"File routine {config['name']!r} cannot define selections."
            )
        builtin = False

    return AnalysisRoutineConfig(
        name=name,
        routine=routine,
        loader=loader,
        systems=tuple(systems),
        selections=dict(selections),
        inputs=tuple(inputs),
        options=dict(options),
        builtin=builtin,
    )


def normalize_routine_list(
    routines: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path | None = None,
) -> tuple[AnalysisRoutineConfig, ...]:
    if not isinstance(routines, Sequence) or isinstance(routines, (str, bytes)):
        raise ValueError("routines must be a non-empty list.")
    normalized = tuple(
        _normalize_routine_config(routine, base_dir=base_dir) for routine in routines
    )
    if not normalized:
        raise ValueError("routines must be a non-empty list.")
    names = [routine.name for routine in normalized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Routine names must be unique: " + ", ".join(duplicates))
    return normalized


def normalize_analysis_runtime_config(
    config: Mapping[str, Any] | None,
) -> AnalysisRuntimeConfig:
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("run must be a mapping.")
    unknown = set(config) - {"in_memory", "gc_collect", "maxtasksperchild"}
    if unknown:
        raise ValueError(
            "run contains unsupported key(s): " + ", ".join(sorted(unknown))
        )
    maxtasksperchild = int(config.get("maxtasksperchild", 100))
    if maxtasksperchild <= 0:
        raise ValueError("run.maxtasksperchild must be positive.")
    return AnalysisRuntimeConfig(
        in_memory=_strict_bool(config.get("in_memory", True), field="run.in_memory"),
        gc_collect=_strict_bool(
            config.get("gc_collect", False), field="run.gc_collect"
        ),
        maxtasksperchild=maxtasksperchild,
    )


def run_analysis_routine(
    routine: RuntimeRoutine,
    *,
    universe: Any | None,
    inputs: Mapping[str, Any],
    system_id: str,
    sample_id: str,
    start: int,
    stop: int | None,
    step: int,
) -> QoI:
    context = (
        f"Routine {routine.name!r}, system {system_id!r}, sample {sample_id!r}"
    )
    if routine.loader == "files":
        selected_inputs = {}
        for role in routine.inputs:
            if role not in inputs or inputs[role] is None:
                raise ValueError(
                    f"Routine {routine.name!r}, system {system_id!r}, sample "
                    f"{sample_id!r}: required input role {role!r} is missing."
                )
            selected_inputs[role] = inputs[role]
        fn = load_custom_routine(routine.routine)
        try:
            result = fn(
                inputs=selected_inputs,
                system_id=system_id,
                sample_id=sample_id,
                options=dict(routine.options),
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{context}: {exc}") from exc
    else:
        if universe is None:
            raise ValueError(
                f"Routine {routine.name!r} requires an MDAnalysis Universe."
            )
        if routine.builtin:
            fn = BUILTIN_ROUTINES[routine.routine]
            try:
                result = fn(
                    universe=universe,
                    start=start,
                    stop=stop,
                    step=step,
                    **routine.selections,
                    **routine.options,
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{context}: {exc}") from exc
        else:
            fn = load_custom_routine(routine.routine)
            try:
                result = fn(
                    universe=universe,
                    frames=slice(start, stop, step),
                    system_id=system_id,
                    sample_id=sample_id,
                    options=dict(routine.options),
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"{context}: {exc}") from exc
    if not isinstance(result, QoI):
        raise TypeError(
            f"Routine {routine.name!r}, system {system_id!r}, sample "
            f"{sample_id!r} must return exactly one QoI, got "
            f"{type(result).__name__}."
        )
    result.name = routine.name
    return result


def build_analysis_routines(
    routines: Sequence[AnalysisRoutineConfig],
) -> tuple[RuntimeRoutine, ...]:
    return tuple(routines)
