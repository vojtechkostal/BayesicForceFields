from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping

from ...io.utils import load_yaml
from .._shared.config import (
    PathLike,
    _resolve_optional_path,
    _resolve_path,
)


def _boolean(value: object, *, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} must be a boolean.")
    return value


@dataclass(frozen=True)
class LearnModelConfig:
    model_path: Path
    independent_observations: bool = False
    n_eff: float | None = None
    tolerance: float | None = None


@dataclass(frozen=True)
class LearnMCMCConfig:
    priors_disttype: str = 'normal'
    total_steps: int = 1500
    warmup: int = 500
    thin: int = 1
    progress_stride: int = 100
    n_walkers: int | None = None
    checkpoint: Path | None = None
    posterior: Path = Path('./posterior.pt')
    priors: Path | None = Path('./priors.pt')
    restart: bool = True
    device: str = 'cuda'
    rhat_tol: float = 1.01
    ess_min: int = 100
    include_implicit_charge: bool = False


@dataclass(frozen=True)
class LearnConfig:
    fn_config: Path
    specs: Path
    models: dict[str, LearnModelConfig]
    mcmc: LearnMCMCConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> 'LearnConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ('specs', 'models', 'mcmc'):
            if key not in config:
                raise ValueError(f'Missing required configuration section: {key!r}.')

        models_raw = config['models']
        if not isinstance(models_raw, Mapping) or not models_raw:
            raise ValueError("'models' must be a non-empty mapping.")

        mcmc = config['mcmc']
        if not isinstance(mcmc, Mapping):
            raise ValueError("'mcmc' must be a mapping.")

        total_steps = int(mcmc.get('total_steps', 1500))
        warmup = int(mcmc.get('warmup', 500))
        thin = int(mcmc.get('thin', 1))
        progress_stride = int(mcmc.get('progress_stride', 100))
        if total_steps < 1:
            raise ValueError("'mcmc.total_steps' must be positive.")
        if warmup < 0 or warmup >= total_steps:
            raise ValueError("'mcmc.warmup' must satisfy 0 <= warmup < total_steps.")
        if thin < 1:
            raise ValueError("'mcmc.thin' must be positive.")
        if progress_stride < 1:
            raise ValueError("'mcmc.progress_stride' must be positive.")

        mcmc_config = LearnMCMCConfig(
            priors_disttype=str(mcmc.get('priors_disttype', 'normal')),
            total_steps=total_steps,
            warmup=warmup,
            thin=thin,
            progress_stride=progress_stride,
            n_walkers=None if mcmc.get('n_walkers') is None else int(mcmc['n_walkers']),
            checkpoint=_resolve_optional_path(
                base_dir,
                mcmc.get('checkpoint', './mcmc-checkpoint.pt'),
                must_exist=False,
                kind='MCMC checkpoint file',
            ),
            posterior=_resolve_path(
                base_dir,
                mcmc.get('posterior', './posterior.pt'),
                must_exist=False,
                kind='posterior output file',
            ),
            priors=_resolve_optional_path(
                base_dir,
                mcmc.get('priors', './priors.pt'),
                must_exist=False,
                kind='priors output file',
            ),
            restart=_boolean(mcmc.get('restart', True), key='mcmc.restart'),
            device=str(mcmc.get('device', 'cuda')),
            rhat_tol=float(mcmc.get('rhat_tol', 1.01)),
            ess_min=int(mcmc.get('ess_min', 100)),
            include_implicit_charge=_boolean(
                mcmc.get('include_implicit_charge', False),
                key='mcmc.include_implicit_charge',
            ),
        )

        models: dict[str, LearnModelConfig] = {}
        for name, model in models_raw.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Model names must be non-empty strings.")
            if not isinstance(model, Mapping):
                raise ValueError(
                    f"Model {name!r} must be a mapping with a 'model_path' key."
                )
            if 'model_path' not in model:
                raise ValueError(
                    f"Model {name!r} is missing required key 'model_path'."
                )

            supported_keys = {
                'model_path',
                'independent_observations',
                'n_eff',
                'tolerance',
            }
            unsupported_keys = set(model) - supported_keys
            if unsupported_keys:
                keys = ', '.join(sorted(unsupported_keys))
                raise ValueError(
                    f"Model {name!r} has unsupported keys: {keys}."
                )

            independent_observations = _boolean(
                model.get('independent_observations', False),
                key=f"models.{name}.independent_observations",
            )
            n_eff = (
                None if model.get('n_eff') is None else float(model['n_eff'])
            )
            tolerance = (
                None
                if model.get('tolerance') is None
                else float(model['tolerance'])
            )

            if n_eff is not None:
                if not isfinite(n_eff) or n_eff <= 0.0:
                    raise ValueError(
                        f"Model {name!r} n_eff must be positive and finite."
                    )
                if 'independent_observations' in model or 'tolerance' in model:
                    raise ValueError(
                        f"Model {name!r} cannot combine 'n_eff' with "
                        "'independent_observations' or 'tolerance'."
                    )
            elif independent_observations:
                if tolerance is not None:
                    raise ValueError(
                        f"Independent model {name!r} does not use 'tolerance'."
                    )
            else:
                if tolerance is None:
                    raise ValueError(
                        f"Curve model {name!r} must define 'tolerance'."
                    )
                if not isfinite(tolerance) or tolerance <= 0.0:
                    raise ValueError(
                        f"Model {name!r} tolerance must be positive and finite."
                    )

            models[name] = LearnModelConfig(
                model_path=_resolve_path(
                    base_dir,
                    model['model_path'],
                    kind=f'model {name!r} file',
                ),
                independent_observations=independent_observations,
                n_eff=n_eff,
                tolerance=tolerance,
            )

        log = _resolve_path(
            base_dir,
            config.get('log', './out.log'),
            must_exist=False,
            kind='log file',
        )

        return cls(
            fn_config=fn_config,
            specs=_resolve_path(base_dir, config['specs'], kind='specs file'),
            models=models,
            mcmc=mcmc_config,
            log=log,
        )
