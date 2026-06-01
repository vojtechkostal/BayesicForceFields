"""Workflow entry point for sampled FFMD campaigns."""

from ...io.logs import Logger
from .._shared.campaign import (
    build_parameter_samples,
    print_sample_summary,
    run_campaign,
    stage_campaign,
)
from .config import SampleConfig


def main(fn_config: str) -> None:
    """Run a sampled MD campaign for force-field parameter exploration."""
    config = SampleConfig.load(fn_config)
    fn_specs, parameter_samples = build_parameter_samples(config)
    resolved_specs, systems = stage_campaign(config, fn_specs=fn_specs)
    assert resolved_specs is not None

    logger = Logger('sample', str(config.log), mode='w')
    print_sample_summary(config, resolved_specs, logger)
    run_campaign(
        config=config,
        fn_specs=resolved_specs,
        systems=systems,
        parameter_samples=parameter_samples,
        logger=logger,
    )
