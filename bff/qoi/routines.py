"""Routine configuration, loading, and execution.

Every routine, built-in or custom, follows one of two signatures and returns
exactly one :class:`~bff.qoi.dataset.QoI`::

    routine(universe, *, frames, options) -> QoI
    routine(*, inputs, options) -> QoI

Routines that declare ``inputs`` read files; all others analyze a trajectory.
Built-in routines receive their ``selections`` merged into ``options`` and
validate their own options.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

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
                options=dict(routine.options),
            )
        else:
            missing = [role for role in routine.inputs if inputs.get(role) is None]
            if missing:
                raise ValueError(f"required input role(s) missing: {missing}")
            result = routine.function(
                inputs={role: inputs[role] for role in routine.inputs},
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
