from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..campaign.config import SimulationCampaignConfig, load_campaign_config
from ..config import PathLike, check_keys


@dataclass(frozen=True)
class ChargeConstraintConfig:
    selection: str
    target: float
    scope: Literal["system", "residue"]
    implicit: str


def _load_bounds(bounds: Any) -> dict[str, tuple[float, float]]:
    if not isinstance(bounds, dict):
        raise ValueError("'bounds' must be a mapping of parameter names to bounds.")
    loaded: dict[str, tuple[float, float]] = {}
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
        loaded[name] = (lower, upper)
    return loaded


@dataclass(frozen=True, kw_only=True)
class SampleParametersConfig(SimulationCampaignConfig):
    bounds: dict[str, tuple[float, float]]
    charge_constraints: tuple[ChargeConstraintConfig, ...]
    n_samples: int

    @classmethod
    def load(cls, fn_config: PathLike) -> SampleParametersConfig:
        _, config, common = load_campaign_config(
            fn_config,
            stage="sample-parameters",
            stage_keys={"bounds", "charge_constraints", "n_samples"},
            stage_required=("bounds", "charge_constraints", "n_samples"),
        )
        bounds = _load_bounds(config["bounds"])
        if not isinstance(config["charge_constraints"], list):
            raise ValueError("'charge_constraints' must be a list.")
        constraints: list[ChargeConstraintConfig] = []
        for index, raw in enumerate(config["charge_constraints"]):
            where = f"charge_constraints[{index}]"
            keys = ("selection", "target", "scope", "implicit")
            check_keys(raw, where=where, allowed=keys, required=keys)
            scope = str(raw["scope"])
            if scope not in {"system", "residue"}:
                raise ValueError(
                    f"{where}.scope must be 'system' or 'residue', got {scope!r}."
                )
            implicit = str(raw["implicit"])
            if implicit not in bounds:
                raise ValueError(
                    f"{where}.implicit ({implicit!r}) must match a parameter "
                    "defined in 'bounds'."
                )
            if not implicit.startswith("charge "):
                raise ValueError(
                    f"{where}.implicit must be a charge parameter, got {implicit!r}."
                )
            constraints.append(
                ChargeConstraintConfig(
                    selection=str(raw["selection"]),
                    target=float(raw["target"]),
                    scope=scope,
                    implicit=implicit,
                )
            )
        implicit_params = [constraint.implicit for constraint in constraints]
        if len(implicit_params) != len(set(implicit_params)):
            raise ValueError(
                "Each charge constraint must define a distinct implicit parameter."
            )
        n_samples = int(config["n_samples"])
        if n_samples <= 0:
            raise ValueError("'n_samples' must be a positive integer.")
        return cls(
            **common,
            bounds=bounds,
            charge_constraints=tuple(constraints),
            n_samples=n_samples,
        )
