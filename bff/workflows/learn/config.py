from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...io.utils import load_yaml
from .._shared.config import (
    PathLike,
    _load_model_paths,
    _resolve_optional_path,
    _resolve_path,
)


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
    models: dict[str, Path]
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
            restart=bool(mcmc.get('restart', True)),
            device=str(mcmc.get('device', 'cuda')),
            rhat_tol=float(mcmc.get('rhat_tol', 1.01)),
            ess_min=int(mcmc.get('ess_min', 100)),
            include_implicit_charge=bool(mcmc.get('include_implicit_charge', False)),
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
            models=_load_model_paths(base_dir, config['models']),
            mcmc=mcmc_config,
            log=log,
        )
