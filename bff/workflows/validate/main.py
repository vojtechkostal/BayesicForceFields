"""Workflow entry point for posterior-sample validation campaigns."""

from ...io.logs import Logger
from .._shared.campaign import (
    load_parameter_samples,
    print_validate_summary,
    run_campaign,
    stage_campaign,
)
from .config import ValidateConfig


def main(fn_config: str) -> None:
    """Run a validation campaign from explicit parameters or a posterior."""
    config = ValidateConfig.load(fn_config)

    if config.posterior is None:
        assert config.parameters is not None
        assert config.specs is not None
        parameter_samples = load_parameter_samples(config.parameters, config.specs)
        source_specs = config.specs
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
        config.campaign_dir.mkdir(parents=True, exist_ok=True)
        source_specs = config.campaign_dir.resolve() / "specs.yaml"
        posterior.specs.write(source_specs)

    fn_specs, systems = stage_campaign(config, fn_specs=source_specs)
    assert fn_specs is not None

    logger = Logger("validate", str(config.log), mode="w")
    print_validate_summary(config, len(parameter_samples), logger)
    run_campaign(
        config=config,
        fn_specs=fn_specs,
        systems=systems,
        parameter_samples=parameter_samples,
        logger=logger,
    )
