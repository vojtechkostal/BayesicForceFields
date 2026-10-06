"""Draw force-field parameter samples and run one MD campaign over them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...domain.specs import ChargeConstraint, RandomParamsGenerator, Specs
from ...io.logs import Logger
from ...topology import TopologyModifier
from ..campaign.run import log_campaign_summary, run_campaign
from .config import SampleParametersConfig


def compile_specs(config: SampleParametersConfig) -> Specs:
    """Turn bounds and charge constraints into reconstructable equations.

    Every constraint selection must yield the same linear equation in every
    configured system (and residue, for residue scope). Constraints must be
    disjoint or nested, and a parent's implicit parameter must not appear in
    a child constraint.
    """
    modifiers = [TopologyModifier(system.topology_path) for system in config.systems]
    charge_params = [name for name in config.bounds if name.startswith("charge ")]

    # Atoms controlled by each charge parameter, per system.
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
        if missing:
            raise ValueError(
                f"Charge parameter {parameter!r} references atom name or type "
                "token(s) not found in any configured system: "
                + ", ".join(sorted(repr(name) for name in missing))
                + "."
            )

    # One equation per selected group: sum(coefficient * charge) + fixed = target.
    compiled: list[dict[str, Any]] = []
    selected_atoms: list[list[set[int]]] = []
    for constraint in config.charge_constraints:
        equations: list[tuple[dict[str, float], float]] = []
        selections: list[set[int]] = []
        for modifier, by_parameter in zip(modifiers, parameter_indices):
            try:
                groups = modifier.selected_groups(
                    constraint.selection, constraint.scope
                )
            except Exception as exc:
                raise ValueError(
                    f"Invalid MDAnalysis selection {constraint.selection!r}."
                ) from exc
            selections.append(set().union(*groups))
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
            raise ValueError(
                f"Charge-constraint selection {constraint.selection!r} does not "
                "match any atoms in the configured systems."
            )

        normalized = []
        for coefficients, fixed_charge in equations:
            implicit = coefficients.get(constraint.implicit, 0.0)
            if np.isclose(implicit, 0.0):
                raise ValueError(
                    f"Implicit parameter {constraint.implicit!r} is not selected "
                    f"by its owning constraint {constraint.selection!r} in every "
                    "system."
                )
            row = np.asarray(
                [coefficients.get(name, 0.0) / implicit for name in config.bounds]
            )
            normalized.append((row, (constraint.target - fixed_charge) / implicit))
        reference_row, reference_target = normalized[0]
        for row, target in normalized[1:]:
            if not np.allclose(row, reference_row) or not np.isclose(
                target, reference_target
            ):
                raise ValueError(
                    f"Charge constraint {constraint.selection!r} does not define "
                    "one consistent equation across the configured systems and "
                    f"{constraint.scope} groups."
                )
        coefficients, fixed_charge = equations[0]
        compiled.append(
            {
                "selection": constraint.selection,
                "target": constraint.target,
                "scope": constraint.scope,
                "implicit": constraint.implicit,
                "coefficients": coefficients,
                "fixed_charge": fixed_charge,
            }
        )
        selected_atoms.append(selections)

    for first in range(len(compiled)):
        for second in range(first + 1, len(compiled)):
            names = (
                f"{compiled[first]['selection']!r} and "
                f"{compiled[second]['selection']!r}"
            )
            relations: set[str] = set()
            for atoms_first, atoms_second in zip(
                selected_atoms[first], selected_atoms[second]
            ):
                if not atoms_first or not atoms_second:
                    continue
                if not atoms_first & atoms_second:
                    relations.add("disjoint")
                elif atoms_first == atoms_second:
                    raise ValueError(
                        "Charge constraints must not select exactly the same "
                        f"atoms: {names}."
                    )
                elif atoms_first < atoms_second:
                    relations.add("first-child")
                elif atoms_second < atoms_first:
                    relations.add("second-child")
                else:
                    raise ValueError(
                        "Charge constraints may be disjoint or nested, but must not "
                        f"partially overlap: {names}."
                    )
            if len(relations) > 1:
                raise ValueError(
                    "Charge-constraint hierarchy changes between configured "
                    f"systems: {names}."
                )
            if relations == {"first-child"}:
                child, parent = compiled[first], compiled[second]
            elif relations == {"second-child"}:
                child, parent = compiled[second], compiled[first]
            else:
                continue
            if child["coefficients"].get(parent["implicit"], 0.0):
                raise ValueError(
                    f"Implicit parameter {parent['implicit']!r} owned by parent "
                    f"constraint {parent['selection']!r} must not appear in "
                    f"descendant constraint {child['selection']!r}."
                )

    return Specs({"bounds": config.bounds, "charge_constraints": compiled})


def main(fn_config: str | Path) -> None:
    config = SampleParametersConfig.load(fn_config)
    config.campaign_dir.mkdir(parents=True, exist_ok=True)
    fn_specs = config.campaign_dir / "specs.yaml"
    specs = compile_specs(config)
    specs.write(fn_specs)
    constraint = ChargeConstraint(specs)
    parameter_samples = RandomParamsGenerator(constraint.explicit_bounds, constraint)(
        config.n_samples
    )

    logger = Logger("sample-parameters", str(config.log), mode="w")
    log_campaign_summary(config, config.n_samples, logger, title="Sampling Campaign")
    logger.info("parameters (full array order):", level=1)
    for index, name in enumerate(specs.parameter_names()):
        role = "implicit" if name in specs.implicit_params else "explicit"
        logger.info(f"{index}: {name}: {specs.bounds.get(name)} ({role})", level=2)
    logger.kv(
        "Sampled parameter order",
        ", ".join(specs.parameter_names(explicit_only=True)) or "none",
    )
    logger.kv("Charge constraints", len(specs.charge_constraints))
    logger.blank()
    run_campaign(
        config,
        fn_specs=fn_specs,
        parameter_samples=parameter_samples,
        logger=logger,
    )
