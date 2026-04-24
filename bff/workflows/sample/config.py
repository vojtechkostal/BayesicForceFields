from __future__ import annotations

from dataclasses import dataclass

from .._shared.config import (
    PathLike,
    SimulationCampaignConfig,
    _load_campaign_common,
    _normalize_implicit_atoms,
    _validate_bounds,
)


@dataclass(frozen=True, kw_only=True)
class SampleConfig(SimulationCampaignConfig):
    mol_resname: str
    bounds: dict[str, tuple[float, float]]
    total_charge: float
    implicit_atoms: list[str]
    n_samples: int

    @classmethod
    def load(cls, fn_config: PathLike) -> 'SampleConfig':
        _, _, config, common = _load_campaign_common(
            fn_config,
            asset_systems=True,
        )

        required = [
            'mol_resname',
            'bounds',
            'total_charge',
            'implicit_atoms',
            'n_samples',
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(
                'Sample workflow requires configuration key(s): '
                + ', '.join(repr(key) for key in missing)
            )

        bounds = _validate_bounds(config['bounds'])
        implicit_atoms = _normalize_implicit_atoms(config['implicit_atoms'])
        bound_charge_groups = {
            name.split(maxsplit=1)[1]
            for name in bounds
            if name.startswith('charge ')
        }
        implicit_group = ' '.join(implicit_atoms)
        if bound_charge_groups and implicit_group not in bound_charge_groups:
            raise ValueError(
                f"'implicit_atoms' ({implicit_group}) must match one of the "
                "charge parameters defined in 'bounds'."
            )

        n_samples = int(config['n_samples'])
        if n_samples <= 0:
            raise ValueError("'n_samples' must be a positive integer.")

        return cls(
            **common,
            mol_resname=str(config['mol_resname']),
            bounds=bounds,
            total_charge=float(config['total_charge']),
            implicit_atoms=implicit_atoms,
            n_samples=n_samples,
        )
