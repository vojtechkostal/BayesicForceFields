"""Routine configuration, loading, and execution.

Every routine, built-in or custom, follows one of two signatures and returns
exactly one :class:`~bff.qoi.dataset.QoI`::

    routine(universe, *, frames, system_id, sample_id, options) -> QoI
    routine(*, inputs, system_id, sample_id, options) -> QoI

Routines that declare ``inputs`` read files; all others analyze a trajectory.
Built-in routines receive their ``selections`` merged into ``options`` and
validate their own options.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from ..domain.systems import validate_system_id, validate_unique_system_ids
from .dataset import QoI
from .hbonds import hydrogen_bonds
from .rdf import rdf

BUILTIN_ROUTINES: dict[str, Callable[..., QoI]] = {
    "rdf": rdf,
    "hydrogen_bonds": hydrogen_bonds,
}


@dataclass(frozen=True, slots=True)
class AnalysisRoutineConfig:
    name: str
    systems: tuple[str, ...]
    type: str | None = None
    callable: str | None = None
    inputs: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def uses_trajectory(self) -> bool:
        """Routines without file inputs analyze a trajectory."""
        return not self.inputs

    @property
    def function(self) -> Callable[..., QoI]:
        if self.type is not None:
            return BUILTIN_ROUTINES[self.type]
        return load_custom_routine(self.callable or "")[1]


@lru_cache(maxsize=None)
def load_custom_routine(
    specification: str,
    base_dir: Path | None = None,
) -> tuple[str, Callable[..., Any]]:
    """Import ``module:function`` or ``path/to/file.py:function``.

    Returns the specification with an absolute file path and the callable.
    """
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

    routine = getattr(module, attribute, None)
    if routine is None:
        raise ValueError(
            f"Callable {attribute!r} was not found in module {module_name!r}."
        )
    if not callable(routine):
        raise ValueError(f"Resolved object {specification!r} is not callable.")
    return specification, routine


def _load_routine_config(
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
    if ("type" in config) == ("callable" in config):
        raise ValueError(
            f"Routine {name!r} must define exactly one of type or callable."
        )

    systems = config.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ValueError(f"Routine {name!r}.systems must be a non-empty ID list.")
    systems = [
        validate_system_id(value, field=f"routines.{name}.systems[{index}]")
        for index, value in enumerate(systems)
    ]
    validate_unique_system_ids(systems, field=f"routines.{name}.systems")

    selections = config.get("selections", {})
    inputs = config.get("inputs", [])
    options = config.get("options", {})
    if not isinstance(selections, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in selections.items()
    ):
        raise ValueError(f"Routine {name!r}.selections must map names to selections.")
    if not isinstance(inputs, list) or not all(
        isinstance(role, str) and role for role in inputs
    ):
        raise ValueError(f"Routine {name!r}.inputs must be a role list.")
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"Routine {name!r}.inputs contains duplicates.")
    if not isinstance(options, Mapping):
        raise ValueError(f"Routine {name!r}.options must be a mapping.")
    duplicated = set(selections) & set(options)
    if duplicated:
        raise ValueError(
            f"Routine {name!r} defines {sorted(duplicated)} in both selections "
            "and options."
        )

    routine_type = None
    callable_spec = None
    if "type" in config:
        routine_type = str(config["type"])
        if routine_type not in BUILTIN_ROUTINES:
            raise ValueError(
                f"Routine {name!r} has unknown type {routine_type!r}; "
                f"expected one of {sorted(BUILTIN_ROUTINES)}."
            )
        if inputs:
            raise ValueError(f"Built-in routine {name!r} cannot define inputs.")
    else:
        if selections:
            raise ValueError(
                f"Custom routine {name!r} cannot define selections; pass custom "
                "settings through options."
            )
        callable_spec, _ = load_custom_routine(str(config["callable"]), base_dir)

    return AnalysisRoutineConfig(
        name=name,
        systems=tuple(systems),
        type=routine_type,
        callable=callable_spec,
        inputs=tuple(inputs),
        options={**selections, **options},
    )


def load_routine_configs(
    routines: Sequence[Mapping[str, Any]],
    *,
    base_dir: Path | None = None,
) -> tuple[AnalysisRoutineConfig, ...]:
    if not isinstance(routines, Sequence) or isinstance(routines, (str, bytes)):
        raise ValueError("routines must be a non-empty list.")
    loaded = tuple(
        _load_routine_config(routine, base_dir=base_dir) for routine in routines
    )
    if not loaded:
        raise ValueError("routines must be a non-empty list.")
    names = [routine.name for routine in loaded]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Routine names must be unique: " + ", ".join(duplicates))
    return loaded


def run_routine(
    routine: AnalysisRoutineConfig,
    *,
    universe: Any | None,
    frames: slice,
    inputs: Mapping[str, Any],
    system_id: str,
    sample_id: str,
) -> QoI:
    """Call one routine and enforce that it returns exactly one QoI."""
    context = f"Routine {routine.name!r}, system {system_id!r}, sample {sample_id!r}"
    try:
        if routine.uses_trajectory:
            result = routine.function(
                universe=universe,
                frames=frames,
                system_id=system_id,
                sample_id=sample_id,
                options=dict(routine.options),
            )
        else:
            missing = [role for role in routine.inputs if inputs.get(role) is None]
            if missing:
                raise ValueError(f"required input role(s) missing: {missing}")
            result = routine.function(
                inputs={role: inputs[role] for role in routine.inputs},
                system_id=system_id,
                sample_id=sample_id,
                options=dict(routine.options),
            )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {exc}") from exc

    if not isinstance(result, QoI):
        raise TypeError(
            f"{context} must return exactly one QoI, got {type(result).__name__}."
        )
    result.name = routine.name
    return result
