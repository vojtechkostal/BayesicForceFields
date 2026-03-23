from .configs import SimulateConfig
from .runsims import (
    build_parameter_samples,
    print_simulate_summary,
    run_campaign,
    stage_campaign,
)
from ..io.logs import Logger


def main(fn_config: str) -> None:
    """Run a sampled simulation campaign for training-set generation."""
    config = SimulateConfig.load(fn_config)
    fn_specs, parameter_samples = build_parameter_samples(config)
    resolved_specs, systems = stage_campaign(config, fn_specs=fn_specs)
    assert resolved_specs is not None

    logger = Logger("simulate")
    print_simulate_summary(config, resolved_specs, logger)
    run_campaign(
        config=config,
        fn_specs=resolved_specs,
        systems=systems,
        parameter_samples=parameter_samples,
        logger=logger,
    )
