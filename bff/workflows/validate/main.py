"""Rerun explicit or posterior-drawn parameter samples as a new campaign."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...domain.specs import Specs
from ...io.logs import Logger
from ...io.utils import load_yaml
from ..campaign.run import log_campaign_summary, run_campaign
from .config import ValidateConfig


def load_parameter_samples(fn_samples: Path, specs: Specs) -> np.ndarray:
    """Load samples from a YAML mapping of explicit parameter name to values."""
    raw = load_yaml(fn_samples)
    names = specs.parameter_names(explicit_only=True)
    if not isinstance(raw, dict) or not all(name in raw for name in names):
        raise ValueError(
            f"Parameter sample file {fn_samples} must map every explicit "
            f"parameter name ({', '.join(names)}) to a list of values."
        )
    if len({len(raw[name]) for name in names}) != 1:
        raise ValueError(
            "Column-oriented YAML sample lists must all have the same length."
        )
    samples = np.column_stack([np.asarray(raw[name], dtype=float) for name in names])
    if samples.shape[0] == 0:
        raise ValueError("No validation parameter samples were found.")
    return samples


def main(fn_config: str | Path) -> None:
    config = ValidateConfig.load(fn_config)
    config.campaign_dir.mkdir(parents=True, exist_ok=True)

    if config.posterior is None:
        fn_specs = config.specs
        parameter_samples = load_parameter_samples(config.parameters, Specs(fn_specs))
    else:
        from ...bayes.results import PosteriorResults

        posterior = PosteriorResults.load(config.posterior.file)
        if posterior.specs is None:
            raise ValueError(
                f"Posterior '{config.posterior.file}' does not contain embedded "
                "parameter specifications."
            )
        parameter_samples = posterior.sample_posterior(
            n_samples=config.posterior.n_samples,
            include_mean=config.posterior.include_mean,
            distribution=config.posterior.distribution,
            confidence=config.posterior.confidence,
            random_state=config.posterior.seed,
        )
        fn_specs = config.campaign_dir / "specs.yaml"
        posterior.specs.write(fn_specs)

    logger = Logger("validate", str(config.log), mode="w")
    log_campaign_summary(
        config, len(parameter_samples), logger, title="Validation Campaign"
    )
    if config.posterior is None:
        logger.kv("Parameter source", config.parameters)
    else:
        logger.kv("Posterior source", config.posterior.file)
        logger.kv("Random draws", config.posterior.n_samples)
        logger.kv("Distribution", config.posterior.distribution)
        logger.kv("Confidence", config.posterior.confidence)
        logger.kv(
            "Seed",
            "fresh random seed"
            if config.posterior.seed is None
            else config.posterior.seed,
        )
        logger.kv(
            "Posterior mean sample",
            "first sample" if config.posterior.include_mean else "not included",
        )
    logger.blank()
    run_campaign(
        config,
        fn_specs=fn_specs,
        parameter_samples=parameter_samples,
        logger=logger,
    )
