"""Workflow entry point for surrogate training."""

import sys
from pathlib import Path
from typing import Union

from ..io.logs import Logger
from ..qoi.data import QoIDataset
from .configs import TrainConfig

PathLike = Union[str, Path]


def _load_datasets(config: TrainConfig) -> tuple[QoIDataset, ...]:
    datasets: list[QoIDataset] = []
    for dataset_config in config.datasets:
        dataset = QoIDataset.load(dataset_config.fn_data)
        dataset.nuisance = dataset_config.nuisance
        datasets.append(dataset)
    return tuple(datasets)


def _dataset_options(
    config: TrainConfig,
) -> tuple[dict[str, Path | None], dict[str, object], dict[str, float]]:
    model_paths = {dataset.name: dataset.fn_model for dataset in config.datasets}
    y_means = {dataset.name: dataset.mean for dataset in config.datasets}
    observation_scales = {
        dataset.name: dataset.observation_scale for dataset in config.datasets
    }
    return model_paths, y_means, observation_scales


def main(fn_config: PathLike) -> None:
    try:
        from ..bayes.learning import train_surrogates
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for 'bff train'. Install a CPU or CUDA "
                "build of PyTorch first."
            ) from exc
        raise

    config = TrainConfig.load(fn_config)
    logger = Logger("BFF", str(config.log), mode="w")
    datasets = _load_datasets(config)
    model_paths, y_means, observation_scales = _dataset_options(config)

    config.training.model_dir.mkdir(parents=True, exist_ok=True)

    train_surrogates(
        datasets,
        y_means=y_means,
        observation_scales=observation_scales,
        model_paths=model_paths,
        reuse_models=config.training.reuse_models,
        n_hyper_max=config.training.n_hyper_max,
        committee_size=config.training.committee_size,
        test_fraction=config.training.test_fraction,
        device=config.training.device,
        logger=logger,
        **config.training.opt_kwargs,
    )


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
