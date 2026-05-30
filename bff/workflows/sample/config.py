from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .._shared.config import (
    PathLike,
    SimulationCampaignConfig,
    _load_campaign_common,
    _validate_bounds,
)


@dataclass(frozen=True)
class ChargeConstraintConfig:
    selection: str
    target: float
    scope: Literal["system", "residue"]
    implicit: str


@dataclass(frozen=True, kw_only=True)
class SampleConfig(SimulationCampaignConfig):
    bounds: dict[str, tuple[float, float]]
    charge_constraints: tuple[ChargeConstraintConfig, ...]
    n_samples: int

    @classmethod
    def load(cls, fn_config: PathLike) -> 'SampleConfig':
        _, _, config, common = _load_campaign_common(fn_config)

        required = [
            'bounds',
            'charge_constraints',
            'n_samples',
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                'Sample workflow requires configuration key(s): '
                + ', '.join(repr(key) for key in missing)
            )

        bounds = _validate_bounds(config['bounds'])
        raw_constraints = config['charge_constraints']
        if not isinstance(raw_constraints, list):
            raise ValueError("'charge_constraints' must be a list.")
        charge_constraints: list[ChargeConstraintConfig] = []
        for index, constraint in enumerate(raw_constraints):
            if not isinstance(constraint, dict):
                raise ValueError(f"charge_constraints[{index}] must be a mapping.")
            missing = [
                key
                for key in ('selection', 'target', 'scope', 'implicit')
                if key not in constraint
            ]
            if missing:
                raise ValueError(
                    f"charge_constraints[{index}] is missing required key(s): "
                    + ', '.join(repr(key) for key in missing)
                )
            scope = str(constraint['scope'])
            if scope not in {'system', 'residue'}:
                raise ValueError(
                    f"charge_constraints[{index}].scope must be 'system' or "
                    f"'residue', got {scope!r}."
                )
            implicit = str(constraint['implicit'])
            if implicit not in bounds:
                raise ValueError(
                    f"charge_constraints[{index}].implicit ({implicit!r}) must "
                    "match a parameter defined in 'bounds'."
                )
            if not implicit.startswith('charge '):
                raise ValueError(
                    f"charge_constraints[{index}].implicit must be a charge "
                    f"parameter, got {implicit!r}."
                )
            charge_constraints.append(
                ChargeConstraintConfig(
                    selection=str(constraint['selection']),
                    target=float(constraint['target']),
                    scope=scope,
                    implicit=implicit,
                )
            )
        implicit_params = [constraint.implicit for constraint in charge_constraints]
        if len(implicit_params) != len(set(implicit_params)):
            raise ValueError(
                "Each charge constraint must define a distinct implicit parameter."
            )

        n_samples = int(config['n_samples'])
        if n_samples <= 0:
            raise ValueError("'n_samples' must be a positive integer.")

        return cls(
            **common,
            bounds=bounds,
            charge_constraints=tuple(charge_constraints),
            n_samples=n_samples,
        )
