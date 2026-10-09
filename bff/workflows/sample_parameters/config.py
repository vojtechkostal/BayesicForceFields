from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..campaign.config import SimulationCampaignConfig, load_campaign_config
from ..config import ConfigSection, PathLike


@dataclass(frozen=True)
class ChargeConstraintConfig:
    selection: str
    target: float
    scope: Literal["system", "residue"]
    implicit: str


def _load_bounds(config: ConfigSection) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for name, value in config.mapping("bounds").items():
        field = f"bounds.{name}"
        if not (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(x)
                for x in value
            )
            and value[0] < value[1]
        ):
            raise ValueError(
                f"{field} must be [lower, upper] with finite lower < upper, "
                f"got {value!r}."
            )
        bounds[str(name)] = (float(value[0]), float(value[1]))
    if not bounds:
        raise ValueError("bounds must define at least one parameter.")
    return bounds


@dataclass(frozen=True, kw_only=True)
class SampleParametersConfig(SimulationCampaignConfig):
    bounds: dict[str, tuple[float, float]]
    charge_constraints: tuple[ChargeConstraintConfig, ...]
    n_samples: int
    seed: int | None = None

    @classmethod
    def load(cls, fn_config: PathLike) -> SampleParametersConfig:
        config, common = load_campaign_config(
            fn_config,
            stage="sample-parameters",
            stage_keys={"bounds", "charge_constraints", "n_samples", "seed"},
            stage_required=("bounds", "n_samples"),
        )
        bounds = _load_bounds(config)
        constraints: list[ChargeConstraintConfig] = []
        keys = ("selection", "target", "scope", "implicit")
        raw_constraints = (
            config.sections("charge_constraints", allowed=keys, required=keys)
            if config.get("charge_constraints")
            else []
        )
        for raw in raw_constraints:
            # The implicit atom (name or type) selects the charge parameter
            # that is solved from the constraint instead of being sampled.
            atom = raw.string("implicit")
            labels = [
                label
                for label in bounds
                if label.startswith("charge ") and atom in label.split()[1:]
            ]
            if len(labels) != 1:
                raise ValueError(
                    f"{raw.field('implicit')} must be an atom name or type of "
                    f"exactly one 'charge ...' parameter in bounds, got {atom!r}."
                )
            implicit = labels[0]
            constraints.append(
                ChargeConstraintConfig(
                    selection=raw.string("selection"),
                    target=raw.number("target"),
                    scope=raw.string("scope", choices=("system", "residue")),
                    implicit=implicit,
                )
            )
        implicit_params = [constraint.implicit for constraint in constraints]
        if len(implicit_params) != len(set(implicit_params)):
            raise ValueError(
                "charge_constraints must each solve a different charge parameter; "
                f"implicit atoms resolve to {implicit_params}."
            )
        return cls(
            **common,
            bounds=bounds,
            charge_constraints=tuple(constraints),
            n_samples=config.integer("n_samples", minimum=1),
            seed=config.integer("seed", None, minimum=0),
        )
