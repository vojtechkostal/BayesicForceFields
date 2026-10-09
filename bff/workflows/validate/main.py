"""Rerun explicit or posterior-drawn parameter samples as a new campaign."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...domain.charge_constraints import check_specs_topologies
from ...domain.specs import Specs
from ...io.utils import file_sha256, load_yaml
from ..campaign.run import fresh_seed, run_campaign
from .config import ValidateConfig


def load_parameter_samples(fn_samples: Path, specs: Specs) -> np.ndarray:
    """Load samples from a YAML mapping of parameter name to a list of values.

    Every explicit parameter needs a column. Columns of implicit charges, as
    written by ``PosteriorResults.sample_posterior(include_implicit_charge=
    True)``, are accepted and ignored: they are reconstructed from the
    constraints.
    """
    raw = load_yaml(fn_samples)
    names = specs.explicit_names
    if not isinstance(raw, dict):
        raise ValueError(
            f"{fn_samples} must map every explicit parameter name "
            f"({', '.join(names)}) to a list of values."
        )
    unknown = sorted(set(raw) - set(specs.names))
    missing = [name for name in names if name not in raw]
    if unknown or missing:
        raise ValueError(
            f"{fn_samples} does not match the specifications: "
            f"missing {missing}, unknown {unknown}."
        )
    columns = [raw[name] for name in names]
    if not all(isinstance(column, list) for column in columns) or (
        len({len(column) for column in columns}) > 1
    ):
        raise ValueError(f"{fn_samples}: every column must be a list of one length.")
    samples = np.column_stack([np.asarray(column, dtype=float) for column in columns])
    if samples.shape[0] == 0:
        raise ValueError(f"{fn_samples} contains no parameter samples.")
    return samples


def main(fn_config: str | Path) -> None:
    config = ValidateConfig.load(fn_config)
    if config.posterior is None:
        specs = Specs(config.specs)
    else:
        from ...bayes.results import Results

        results = Results.load(config.posterior.file)
        specs = results.specs
    # Validation systems may differ from the sampled ones; the learned
    # charges must still be reconstructable in them.
    check_specs_topologies(specs, [system.topology_path for system in config.systems])

    def draw() -> tuple[np.ndarray, dict[str, Any]]:
        if config.posterior is None:
            provenance = {
                "source": "parameters",
                "parameters": str(config.parameters),
                "parameters_sha256": file_sha256(config.parameters),
                "specs": str(config.specs),
            }
            return load_parameter_samples(config.parameters, specs), provenance
        settings = config.posterior
        seed = fresh_seed() if settings.seed is None else settings.seed
        draws = results.draw(
            settings.n_samples,
            distribution=settings.distribution,
            confidence=settings.confidence,
            seed=seed,
            include_mean=settings.include_mean,
            include_map=settings.include_map,
        )
        samples = np.column_stack([draws[name] for name in specs.explicit_names])
        provenance = {
            "source": "posterior",
            "results": str(settings.file),
            "results_sha256": file_sha256(settings.file),
            "distribution": settings.distribution,
            "confidence": settings.confidence,
            "n_samples": settings.n_samples,
            "include_mean": settings.include_mean,
            "include_map": settings.include_map,
            "seed": seed,
        }
        return samples, provenance

    run_campaign(
        config,
        stage="validate",
        title="Validate",
        specs=specs,
        draw=draw,
    )
