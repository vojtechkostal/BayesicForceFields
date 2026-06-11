from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ...io.utils import load_yaml
from .._shared.config import PathLike, _resolve_path


@dataclass(frozen=True)
class FitDatasetConfig:
    name: str
    fn_data: Path
    mean: Any = 0
    nuisance: float | None = None
    fn_model: Path | None = None


@dataclass(frozen=True)
class FitOptionsConfig:
    model_dir: Path
    reuse_models: bool = True
    n_hyper_max: int = 200
    committee_size: int = 1
    test_fraction: float = 0.2
    device: str = 'cuda'
    opt_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitConfig:
    fn_config: Path
    datasets: tuple[FitDatasetConfig, ...]
    fit: FitOptionsConfig
    log: Path

    @classmethod
    def load(cls, fn_config: PathLike) -> 'FitConfig':
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)

        for key in ('datasets', 'fit'):
            if key not in config:
                raise ValueError(f'Missing required configuration section: {key!r}.')

        datasets_raw = config['datasets']
        if not isinstance(datasets_raw, Mapping) or not datasets_raw:
            raise ValueError("'datasets' must be a non-empty mapping.")

        fit = config['fit']
        if not isinstance(fit, Mapping):
            raise ValueError("'fit' must be a mapping.")

        model_dir = _resolve_path(
            base_dir,
            fit.get('model_dir', './models'),
            must_exist=False,
            kind='model directory',
        )

        fit_known_keys = {
            'model_dir',
            'reuse_models',
            'n_hyper_max',
            'committee_size',
            'test_fraction',
            'device',
        }
        opt_kwargs = {
            key: value
            for key, value in fit.items()
            if key not in fit_known_keys
        }
        fit_config = FitOptionsConfig(
            model_dir=model_dir,
            reuse_models=bool(fit.get('reuse_models', True)),
            n_hyper_max=int(fit.get('n_hyper_max', 200)),
            committee_size=int(fit.get('committee_size', 1)),
            test_fraction=float(fit.get('test_fraction', 0.2)),
            device=str(fit.get('device', 'cuda')),
            opt_kwargs=opt_kwargs,
        )

        if not (0 < fit_config.test_fraction < 1):
            raise ValueError("'fit.test_fraction' must be between 0 and 1.")
        if fit_config.n_hyper_max < 1:
            raise ValueError("'fit.n_hyper_max' must be positive.")
        if fit_config.committee_size < 1:
            raise ValueError("'fit.committee_size' must be positive.")

        datasets: list[FitDatasetConfig] = []
        for name, dataset in datasets_raw.items():
            if not isinstance(dataset, Mapping):
                raise ValueError(f'Dataset {name!r} must be a mapping.')
            if 'data' not in dataset:
                raise ValueError(f"Dataset {name!r} is missing required key 'data'.")
            if 'observation_scale' in dataset:
                raise ValueError(
                    "'observation_scale' has been removed from 'bff fit'. "
                    "Configure effective observations under 'models' in "
                    "'bff learn' instead."
                )
            nuisance = dataset.get('nuisance')
            if nuisance is not None:
                nuisance = float(nuisance)
                if nuisance <= 0:
                    raise ValueError(
                        'Dataset '
                        f'{name!r} nuisance must be a positive standard deviation.'
                    )
            fn_model = dataset.get('model')
            if fn_model is None:
                fn_model_resolved = model_dir / f'{name}.lgp'
            else:
                fn_model_resolved = _resolve_path(
                    base_dir,
                    fn_model,
                    must_exist=False,
                    kind=f'dataset {name!r} model file',
                )
            datasets.append(
                FitDatasetConfig(
                    name=str(name),
                    fn_data=_resolve_path(
                        base_dir,
                        dataset['data'],
                        kind=f'dataset {name!r} data file',
                    ),
                    mean=dataset.get('mean', 0),
                    nuisance=nuisance,
                    fn_model=fn_model_resolved,
                )
            )

        log = _resolve_path(
            base_dir,
            config.get('log', './out.log'),
            must_exist=False,
            kind='log file',
        )

        return cls(
            fn_config=fn_config,
            datasets=tuple(datasets),
            fit=fit_config,
            log=log,
        )
