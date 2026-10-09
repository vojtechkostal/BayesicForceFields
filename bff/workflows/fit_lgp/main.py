"""Workflow entry point for local-GP surrogate fitting."""

import time
from pathlib import Path

from ...io.logs import Logger
from ...qoi.dataset import QoIDataset
from .config import FitLGPConfig


def main(fn_config: str | Path) -> None:
    workflow_start = time.perf_counter()
    try:
        from ...bayes.fit import fit_surrogates
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for 'bff fit-lgp'. Install a CPU or CUDA "
                "build of PyTorch first."
            ) from exc
        raise

    config = FitLGPConfig.load(fn_config)
    logger = Logger("fit-lgp", str(config.log), mode="w")
    datasets = []
    for dataset_config in config.datasets:
        dataset = QoIDataset.load(dataset_config.fn_data)
        if dataset.name != dataset_config.name:
            raise ValueError(
                f"datasets.{dataset_config.name}.data contains QoI name "
                f"{dataset.name!r}; expected {dataset_config.name!r}."
            )
        dataset.nuisance = dataset_config.nuisance
        datasets.append(dataset)
    model_paths = {dataset.name: dataset.fn_model for dataset in config.datasets}
    means = {dataset.name: dataset.mean for dataset in config.datasets}
    config.fit.model_dir.mkdir(parents=True, exist_ok=True)

    logger.section("Fit LGP")
    logger.kv("Config", Path(fn_config).resolve())
    logger.kv(
        "Datasets",
        ", ".join(f"{data.name} ({data.n_samples} samples)" for data in datasets),
    )
    logger.kv("Models", config.fit.model_dir.resolve())
    logger.kv("Device", "cpu (float64)")
    logger.blank()
    fit_surrogates(
        datasets,
        means=means,
        model_paths=model_paths,
        reuse_models=config.fit.reuse_models,
        n_hyper_max=config.fit.n_hyper_max,
        committee_size=config.fit.committee_size,
        test_fraction=config.fit.test_fraction,
        logger=logger,
        **config.fit.opt_kwargs,
    )
    elapsed = time.perf_counter() - workflow_start
    logger.done(
        "Fit LGP",
        detail=f"{len(datasets)} model(s) | {elapsed:.1f} s | {config.fit.model_dir}",
    )
