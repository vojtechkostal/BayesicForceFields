"""Workflow entry point for surrogate fitting."""

import time
from pathlib import Path
from typing import Union

from ...io.logs import Logger
from ...qoi.data import QoIDataset
from .config import FitConfig

PathLike = Union[str, Path]


def _load_datasets(config: FitConfig) -> tuple[QoIDataset, ...]:
    datasets: list[QoIDataset] = []
    for dataset_config in config.datasets:
        dataset = QoIDataset.load(dataset_config.fn_data)
        dataset.nuisance = dataset_config.nuisance
        datasets.append(dataset)
    return tuple(datasets)


def main(fn_config: PathLike) -> None:
    workflow_start = time.perf_counter()
    try:
        from ...bayes.learning import fit_surrogates
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            raise RuntimeError(
                "PyTorch is required for 'bff fit'. Install a CPU or CUDA "
                'build of PyTorch first.'
            ) from exc
        raise

    config = FitConfig.load(fn_config)
    logger = Logger('fit', str(config.log), mode='w')
    datasets = _load_datasets(config)
    model_paths = {dataset.name: dataset.fn_model for dataset in config.datasets}
    y_means = {dataset.name: dataset.mean for dataset in config.datasets}
    observation_scales = {
        dataset.name: dataset.observation_scale for dataset in config.datasets
    }

    config.fit.model_dir.mkdir(parents=True, exist_ok=True)

    logger.section('Surrogate Fitting')
    logger.kv('Config', Path(fn_config).resolve())
    logger.kv('Log file', config.log.resolve())
    logger.kv('Datasets', len(datasets))
    logger.kv('Model directory', config.fit.model_dir.resolve())
    logger.kv('Device', config.fit.device)
    if config.fit.reuse_models and any(
        path is not None and path.exists()
        for path in model_paths.values()
    ):
        logger.warn(
            'Existing surrogate models may be reused if matching files are present.'
        )
    logger.blank()

    fit_surrogates(
        datasets,
        y_means=y_means,
        observation_scales=observation_scales,
        model_paths=model_paths,
        reuse_models=config.fit.reuse_models,
        n_hyper_max=config.fit.n_hyper_max,
        committee_size=config.fit.committee_size,
        test_fraction=config.fit.test_fraction,
        device=config.fit.device,
        logger=logger,
        **config.fit.opt_kwargs,
    )
    elapsed = time.perf_counter() - workflow_start
    logger.done('Surrogate fitting', detail=f'finished in {elapsed:.2f}s', level=1)
