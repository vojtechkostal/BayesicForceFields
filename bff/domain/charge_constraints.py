"""Compile charge constraints into linear equations for topologies.

A charge constraint fixes the total charge of a group of atoms given by an
MDAnalysis selection: of each selected residue (``scope: residue``) or of all
selected atoms of a system together (``scope: system``). Counting the atoms
each ``charge ...`` parameter controls in the group turns it into one linear
equation; atoms no parameter controls keep their topology charge. The
implicit parameters are solved from these equations (see ``Specs``), and
``specs.yaml`` stores them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .specs import Specs


class ConstraintDefinition(Protocol):
    selection: str
    target: float
    scope: str
    implicit: str


def compile_specs(
    bounds: Mapping[str, Sequence[float]],
    charge_constraints: Sequence[ConstraintDefinition],
    topology_paths: Sequence[Path],
) -> Specs:
    """Turn bounds and charge constraints into linear charge equations.

    Every constraint must give the same equation in every topology (and
    residue, for residue scope), and its implicit parameter must control an
    atom in every group. Every charge token must match an atom name or type.
    """
    compiled = _compile(bounds, charge_constraints, topology_paths, complete=True)
    return Specs({"bounds": dict(bounds), "charge_constraints": compiled})


def check_specs_topologies(specs: Specs, topology_paths: Sequence[Path]) -> None:
    """Require that ``specs`` reconstructs charges correctly in these topologies.

    The topologies may differ from those the specs were compiled for, for
    example in validation: a parameter or constraint may be absent from them,
    but wherever a constraint selects atoms, it must give the stored equation.
    """
    compiled = _compile(
        specs.bounds, specs.charge_constraints, topology_paths, complete=False
    )
    names = list(specs.names)
    for stored, found in zip(specs.charge_constraints, compiled):
        if found is None:
            continue
        expected = _normalized(stored.to_dict(), names)
        actual = _normalized(found, names)
        if not (
            np.allclose(expected[0], actual[0]) and np.isclose(expected[1], actual[1])
        ):
            raise ValueError(
                f"Charge constraint {stored.selection!r} gives a different charge "
                "equation in these systems than in the specifications they were "
                "compiled for; the learned charges cannot be reconstructed here."
            )


def _normalized(
    constraint: Mapping[str, Any], names: Sequence[str]
) -> tuple[np.ndarray, float]:
    """Equation divided by the implicit parameter's coefficient."""
    coefficients = constraint["coefficients"]
    implicit = coefficients[constraint["implicit"]]
    row = np.asarray([coefficients.get(name, 0.0) / implicit for name in names])
    return row, (constraint["target"] - constraint["fixed_charge"]) / implicit


def _compile(
    bounds: Mapping[str, Sequence[float]],
    charge_constraints: Sequence[ConstraintDefinition],
    topology_paths: Sequence[Path],
    *,
    complete: bool,
) -> list[dict[str, Any] | None]:
    """Compiled equation per constraint; ``None`` where none of its atoms exist.

    With ``complete``, every charge token and constraint must match atoms.
    """
    from ..topology import TopologyModifier

    modifiers = [TopologyModifier(path) for path in topology_paths]
    charge_params = [name for name in bounds if name.startswith("charge ")]

    # Atoms controlled by each charge parameter, per topology.
    parameter_indices: list[dict[str, set[int]]] = []
    resolved_tokens: dict[str, set[str]] = {name: set() for name in charge_params}
    for modifier in modifiers:
        by_parameter: dict[str, set[int]] = {}
        owner: dict[int, str] = {}
        for parameter in charge_params:
            matches = modifier.charge_parameter_matches(parameter)
            resolved_tokens[parameter].update(matches)
            by_parameter[parameter] = set().union(*matches.values())
            for index in by_parameter[parameter]:
                if index in owner:
                    raise ValueError(
                        f"Charge parameters {owner[index]!r} and {parameter!r} "
                        f"both modify atom {modifier.atoms[index].name!r} in "
                        f"{modifier.source}."
                    )
                owner[index] = parameter
        parameter_indices.append(by_parameter)
    for parameter, tokens in resolved_tokens.items():
        missing = set(parameter.split()[1:]) - tokens
        if complete and missing:
            raise ValueError(
                f"Charge parameter {parameter!r} references atom name or type "
                "token(s) not found in any configured system: "
                + ", ".join(sorted(repr(name) for name in missing))
                + "."
            )

    # One equation per selected group: sum(coefficient * charge) + fixed = target.
    compiled: list[dict[str, Any] | None] = []
    for constraint in charge_constraints:
        equations: list[tuple[dict[str, float], float]] = []
        for modifier, by_parameter in zip(modifiers, parameter_indices):
            try:
                groups = modifier.selected_groups(
                    constraint.selection, constraint.scope
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid MDAnalysis selection {constraint.selection!r}."
                ) from exc
            for group in groups:
                coefficients = {
                    parameter: float(len(group & indices))
                    for parameter, indices in by_parameter.items()
                    if group & indices
                }
                controlled = set().union(
                    *(group & indices for indices in by_parameter.values())
                )
                fixed_charge = float(
                    sum(modifier.atoms[index].charge for index in group - controlled)
                )
                equations.append((coefficients, fixed_charge))
        if not equations:
            if complete:
                raise ValueError(
                    f"Charge-constraint selection {constraint.selection!r} does "
                    "not match any atoms in the configured systems."
                )
            compiled.append(None)
            continue

        records = [
            {
                "selection": constraint.selection,
                "target": float(constraint.target),
                "scope": constraint.scope,
                "implicit": constraint.implicit,
                "coefficients": coefficients,
                "fixed_charge": fixed_charge,
            }
            for coefficients, fixed_charge in equations
        ]
        for record in records:
            if np.isclose(record["coefficients"].get(constraint.implicit, 0.0), 0.0):
                raise ValueError(
                    f"Implicit parameter {constraint.implicit!r} is not selected "
                    f"by its owning constraint {constraint.selection!r} in every "
                    "system."
                )
        reference_row, reference_target = _normalized(records[0], list(bounds))
        for record in records[1:]:
            row, target = _normalized(record, list(bounds))
            if not np.allclose(row, reference_row) or not np.isclose(
                target, reference_target
            ):
                raise ValueError(
                    f"Charge constraint {constraint.selection!r} does not define "
                    "one consistent equation across the configured systems and "
                    f"{constraint.scope} groups."
                )
        compiled.append(records[0])

    return compiled
