import sys
from pathlib import Path
from typing import Union

from ..bayes.learning import InferenceProblem, train_surrogates
from ..domain.specs import ChargeConstraint
from ..io.logs import Logger
from ..qoi.data import QoIDataset
from .configs import LearnConfig


PathLike = Union[str, Path]


def _load_datasets(config: LearnConfig) -> tuple[QoIDataset, ...]:
    datasets: list[QoIDataset] = []
    for dataset_config in config.datasets:
        dataset = QoIDataset.load(dataset_config.fn_data)
        dataset.nuisance = dataset_config.nuisance
        datasets.append(dataset)
    return tuple(datasets)


def main(fn_config: PathLike) -> None:
    config = LearnConfig.load(fn_config)
    logger = Logger("BFF", str(config.fn_log), mode="w")
    constraint = ChargeConstraint(config.fn_specs)
    datasets = _load_datasets(config)

    config.training.model_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {
        dataset.name: dataset.fn_model
        for dataset in config.datasets
    }
    y_means = {
        dataset.name: dataset.mean
        for dataset in config.datasets
    }

    models = train_surrogates(
        datasets,
        y_means=y_means,
        model_paths=model_paths,
        reuse_models=config.training.reuse_models,
        n_hyper_max=config.training.n_hyper_max,
        committee_size=config.training.committee_size,
        test_fraction=config.training.test_fraction,
        device=config.training.device,
        logger=logger,
        **config.training.opt_kwargs,
    )

    problem = InferenceProblem.from_datasets(
        models,
        datasets,
        qoi=config.qoi,
        constraint=constraint,
    )
    problem.infer(
        priors_disttype=config.mcmc.priors_disttype,
        total_steps=config.mcmc.total_steps,
        warmup=config.mcmc.warmup,
        thin=config.mcmc.thin,
        progress_stride=config.mcmc.progress_stride,
        n_walkers=config.mcmc.n_walkers,
        fn_posterior=config.mcmc.fn_posterior,
        fn_checkpoint=config.mcmc.fn_checkpoint,
        fn_priors=config.mcmc.fn_priors,
        restart=config.mcmc.restart,
        device=config.mcmc.device,
        logger=logger,
        rhat_tol=config.mcmc.rhat_tol,
        ess_min=config.mcmc.ess_min,
    )


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
