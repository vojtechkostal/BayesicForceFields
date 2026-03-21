import numpy as np

from ..io.logs import Logger
from .configs import ValidateConfig
from .runsims import (
    print_validate_summary,
    run_campaign,
    stage_campaign,
)


def main(fn_config: str) -> None:
    """Run a validation campaign for a provided set of parameter vectors."""
    config = ValidateConfig.load(fn_config)
    inputs = np.load(config.inputs)
    if inputs.ndim != 2:
        raise ValueError("'inputs' must contain a 2D array of parameter samples.")

    fn_specs, systems = stage_campaign(config, fn_specs=config.fn_specs)
    assert fn_specs is not None

    logger = Logger("validate")
    print_validate_summary(config, fn_specs, len(inputs), logger)
    run_campaign(
        config=config,
        fn_specs=fn_specs,
        systems=systems,
        parameter_samples=np.asarray(inputs, dtype=float),
        logger=logger,
    )
