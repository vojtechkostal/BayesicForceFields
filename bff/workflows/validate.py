from ..io.logs import Logger
from .configs import ValidateConfig
from .runsims import (
    load_parameter_samples,
    print_validate_summary,
    run_campaign,
    stage_campaign,
)


def main(fn_config: str) -> None:
    """Run a validation campaign for provided force-field parameter samples."""
    config = ValidateConfig.load(fn_config)
    parameter_samples = load_parameter_samples(config.fn_samples, config.fn_specs)

    fn_specs, systems = stage_campaign(config, fn_specs=config.fn_specs)
    assert fn_specs is not None

    logger = Logger("validate")
    print_validate_summary(config, fn_specs, len(parameter_samples), logger)
    run_campaign(
        config=config,
        fn_specs=fn_specs,
        systems=systems,
        parameter_samples=parameter_samples,
        logger=logger,
    )
