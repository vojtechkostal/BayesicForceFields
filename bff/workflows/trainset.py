"""Workflow entry point for sampled training-set generation."""

from ..io.logs import Logger
from .configs import TrainsetConfig
from .runsims import (
    build_parameter_samples,
    print_trainset_summary,
    run_campaign,
    stage_campaign,
)


def main(fn_config: str) -> None:
    """Run a sampled MD campaign for training-set generation."""
    config = TrainsetConfig.load(fn_config)
    fn_specs, parameter_samples = build_parameter_samples(config)
    resolved_specs, systems = stage_campaign(config, fn_specs=fn_specs)
    assert resolved_specs is not None

    logger = Logger("trainset")
    print_trainset_summary(config, resolved_specs, logger)
    run_campaign(
        config=config,
        fn_specs=resolved_specs,
        systems=systems,
        parameter_samples=parameter_samples,
        logger=logger,
    )
