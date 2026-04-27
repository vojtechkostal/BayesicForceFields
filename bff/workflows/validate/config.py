from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .._shared.config import (
    PathLike,
    SimulationCampaignConfig,
    _load_campaign_common,
    _resolve_path,
)


@dataclass(frozen=True, kw_only=True)
class ValidateConfig(SimulationCampaignConfig):
    specs: Path
    parameters: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> 'ValidateConfig':
        _, base_dir, config, common = _load_campaign_common(fn_config)

        if 'specs' not in config:
            raise ValueError("Validation mode requires 'specs'.")
        if 'parameters' not in config:
            raise ValueError("Validation mode requires 'parameters'.")

        return cls(
            **common,
            specs=_resolve_path(base_dir, config['specs'], kind='specs file'),
            parameters=_resolve_path(
                base_dir,
                config['parameters'],
                kind='parameter samples file',
            ),
        )
