"""Draw force-field parameter samples and run one MD campaign over them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...domain.charge_constraints import compile_specs
from ...domain.specs import latin_hypercube
from ..campaign.run import fresh_seed, run_campaign
from .config import SampleParametersConfig


def main(fn_config: str | Path) -> None:
    config = SampleParametersConfig.load(fn_config)
    specs = compile_specs(
        config.bounds,
        config.charge_constraints,
        [system.topology_path for system in config.systems],
    )

    def draw() -> tuple[np.ndarray, dict[str, Any]]:
        seed = fresh_seed() if config.seed is None else config.seed
        provenance = {
            "source": "latin_hypercube",
            "n_samples": config.n_samples,
            "seed": seed,
        }
        return latin_hypercube(specs, config.n_samples, seed), provenance

    run_campaign(
        config,
        stage="sample-parameters",
        title="Sample Parameters",
        specs=specs,
        draw=draw,
    )
