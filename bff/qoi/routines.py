"""Configuration and dispatch for built-in and custom QoI routines."""

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
from .data import QoI
from .hbonds import compute_hydrogen_bond_qoi, validate_hydrogen_bond_options
from .rdf import compute_rdf_qoi, validate_rdf_options


@dataclass(frozen=True, slots=True)
class BuiltinRoutine:
    function: Callable[..., QoI]
    selections: frozenset[str]
    options: frozenset[str]
    validate: Callable[..., Any]


BUILTIN_ROUTINES = {
    "rdf": BuiltinRoutine(
        function=compute_rdf_qoi,
        selections=frozenset({"group_a", "group_b"}),
        options=frozenset(
            {"range", "bins", "pbc", "update_selections", "smooth"}
        ),
        validate=validate_rdf_options,
    ),
    "hydrogen_bonds": BuiltinRoutine(
        function=compute_hydrogen_bond_qoi,
        selections=frozenset({"selection", "water_selection"}),
        options=frozenset(
            {
                "donor_acceptor_cutoff",
                "angle_cutoff",
                "elements",
                "pbc",
                "update_selections",
            }
        ),
        validate=validate_hydrogen_bond_options,
    ),
}


@dataclass(frozen=True, slots=True)
class AnalysisRoutineConfig:
    name: str
    systems: tuple[str, ...]
    type: str | None = None
    callable: str | None = None
    selections: dict[str, str] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def uses_trajectory(self) -> bool:
        """Built-ins and custom routines without file inputs use a trajectory."""
        return self.type is not None or not self.inputs


@lru_cache(maxsize=None)
def resolve_custom_routine(
    specification: str,
    base_dir: Path | None = None,
) -> tuple[str, Callable[..., Any]]:
    """Normalize and load one custom routine specification."""
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "Custom analysis callables must use 'module:function' or "
            "'path/to/file.py:function'."
        )

    if module_name.endswith(".py") or "/" in module_name:
        module_path = Path(module_name)
        if base_dir is not None and not module_path.is_absolute():
            module_path = base_dir / module_path
        module_path = module_path.resolve()
        if not module_path.is_file():
            raise ValueError(f"Analysis routine file not found: {module_path}")
        specification = f"{module_path}:{attribute}"
        digest = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
        imported_name = f"bff_user_routine_{module_path.stem}_{digest}"
        module = sys.modules.get(imported_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(imported_name, module_path)
            if spec is None or spec.loader is None:
                raise ValueError(
                    f"Could not load analysis routine module {module_path}."
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[imported_name] = module
            spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    try:
        routine = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"Callable {attribute!r} was not found in module {module_name!r}."
        ) from exc
    if not callable(routine):
        raise ValueError(f"Resolved object {specification!r} is not callable.")
    return specification, routine


def _normalize_routine_config(
    config: Mapping[str, Any],
    *,
    base_dir: Path | None,
) -> AnalysisRoutineConfig:
    if not isinstance(config, Mapping):
        raise ValueError("Each routines entry must be a mapping.")
    unknown = set(config) - {
        "name",
        "type",
        "callable",
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
            f"Routine {name!r} must define exactly one of type or callable."
        )

    systems_raw = config.get("systems")
    if not isinstance(systems_raw, list) or not systems_raw:
        raise ValueError(f"Routine {name!r}.systems must be a non-empty ID list.")
    systems = [
        validate_system_id(value, field=f"routines.{name}.systems[{index}]")
        for index, value in enumerate(systems_raw)
    ]
    validate_unique_system_ids(systems, field=f"routines.{name}.systems")

    selections = config.get("selections", {})
    inputs = config.get("inputs", [])
    options = config.get("options", {})
    if not isinstance(selections, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in selections.items()
    ):
        raise ValueError(
            f"Routine {name!r}.selections must map names to selections."
        )
    if not isinstance(inputs, list) or not all(
        isinstance(role, str) and role for role in inputs
    ):
        raise ValueError(f"Routine {name!r}.inputs must be a role list.")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"Routine {name!r}.inputs contains duplicates.")
    if not isinstance(options, Mapping):
        raise ValueError(f"Routine {name!r}.options must be a mapping.")

    routine_type: str | None = None
    callable_spec: str | None = None
    if has_type:
        routine_type = str(config["type"])
        builtin = BUILTIN_ROUTINES.get(routine_type)
        if builtin is None:
            raise ValueError(
                f"Routine {name!r} has unknown type {routine_type!r}; "
                f"expected one of {sorted(BUILTIN_ROUTINES)}."
            )
        if set(selections) != builtin.selections:
            raise ValueError(
                f"Routine {name!r}.selections requires exactly "
                f"{sorted(builtin.selections)}, got {sorted(selections)}."
            )
        if inputs:
            raise ValueError(f"Built-in routine {name!r} cannot define inputs.")
        unsupported_options = set(options) - builtin.options
        if unsupported_options:
            raise ValueError(
                f"Routine {name!r}.options contains unsupported key(s): "
                + ", ".join(sorted(unsupported_options))
            )
        builtin.validate(options, context=f"routines.{name}.options")
    else:
        if selections:
            raise ValueError(
                f"Custom routine {name!r} cannot define selections; pass custom "
                "settings through options."
            )
        callable_spec, _ = resolve_custom_routine(
            str(config["callable"]), base_dir
        )

    return AnalysisRoutineConfig(
        name=name,
        systems=tuple(systems),
        type=routine_type,
        callable=callable_spec,
        selections=dict(selections),
        inputs=tuple(inputs),
        options=dict(options),
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


def run_analysis_routine(
    routine: AnalysisRoutineConfig,
    *,
    universe: Any | None,
    inputs: Mapping[str, Any],
    system_id: str,
    sample_id: str,
    start: int,
    stop: int | None,
    step: int,
) -> QoI:
    """Execute one validated routine and enforce its output contract."""
    context = f"Routine {routine.name!r}, system {system_id!r}, sample {sample_id!r}"
    try:
        if routine.type is not None:
            if universe is None:
                raise ValueError("requires an MDAnalysis Universe")
            result = BUILTIN_ROUTINES[routine.type].function(
                universe=universe,
                start=start,
                stop=stop,
                step=step,
                **routine.selections,
                **routine.options,
            )
        elif routine.inputs:
            selected_inputs: dict[str, Any] = {}
            for role in routine.inputs:
                value = inputs.get(role)
                if value is None:
                    raise ValueError(f"required input role {role!r} is missing")
                selected_inputs[role] = value
            _, function = resolve_custom_routine(routine.callable or "")
            result = function(
                inputs=selected_inputs,
                system_id=system_id,
                sample_id=sample_id,
                options=dict(routine.options),
            )
        else:
            if universe is None:
                raise ValueError("requires an MDAnalysis Universe")
            _, function = resolve_custom_routine(routine.callable or "")
            result = function(
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
            f"{context} must return exactly one QoI, got "
            f"{type(result).__name__}."
        )
    result.name = routine.name
    return result
