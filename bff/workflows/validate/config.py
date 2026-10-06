from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..campaign.config import SimulationCampaignConfig, load_campaign_config
from ..config import PathLike, check_keys, resolve_path, strict_bool

PosteriorDistribution = Literal["empirical", "normal", "uniform", "kde"]


@dataclass(frozen=True)
class ValidatePosteriorConfig:
    file: Path
    n_samples: int = 10
    include_mean: bool = False
    distribution: PosteriorDistribution = "normal"
    confidence: float = 0.9
    seed: int | None = None


@dataclass(frozen=True, kw_only=True)
class ValidateConfig(SimulationCampaignConfig):
    specs: Path | None
    parameters: Path | None
    posterior: ValidatePosteriorConfig | None

    @classmethod
    def load(cls, fn_config: PathLike) -> "ValidateConfig":
        base_dir, config, common = load_campaign_config(
            fn_config,
            stage="validate",
            stage_keys={"specs", "parameters", "posterior"},
        )

        has_parameters = "parameters" in config
        has_posterior = "posterior" in config
        if has_parameters == has_posterior:
            raise ValueError(
                "Validation requires exactly one of 'parameters' or 'posterior'."
            )

        if has_parameters:
            if "specs" not in config:
                raise ValueError("Explicit validation mode requires 'specs'.")
            return cls(
                **common,
                specs=resolve_path(base_dir, config["specs"], kind="specs file"),
                parameters=resolve_path(
                    base_dir,
                    config["parameters"],
                    kind="parameter samples file",
                ),
                posterior=None,
            )

        if "specs" in config:
            raise ValueError(
                "Posterior validation uses embedded specifications; remove 'specs'."
            )
        posterior = check_keys(
            config["posterior"],
            where="posterior",
            allowed=(
                "file",
                "n_samples",
                "include_mean",
                "distribution",
                "confidence",
                "seed",
            ),
            required=("file",),
        )

        n_samples = posterior.get("n_samples", 10)
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise ValueError("posterior.n_samples must be a non-negative integer.")
        if n_samples < 0:
            raise ValueError("posterior.n_samples must be a non-negative integer.")

        include_mean = strict_bool(
            posterior.get("include_mean", False),
            field="posterior.include_mean",
        )
        if n_samples == 0 and not include_mean:
            raise ValueError(
                "Posterior validation must produce at least one sample; set "
                "'n_samples' above zero or enable 'include_mean'."
            )

        distribution = posterior.get("distribution", "normal")
        supported = {"empirical", "normal", "uniform", "kde"}
        if not isinstance(distribution, str) or distribution not in supported:
            raise ValueError(
                'posterior.distribution must be "empirical", "normal", '
                '"uniform", or "kde".'
            )

        confidence = posterior.get("confidence", 0.9)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("posterior.confidence must be between 0 and 1.")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 < confidence < 1:
            raise ValueError("posterior.confidence must be between 0 and 1.")

        seed = posterior.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("posterior.seed must be an integer or null.")

        return cls(
            **common,
            specs=None,
            parameters=None,
            posterior=ValidatePosteriorConfig(
                file=resolve_path(
                    base_dir,
                    posterior["file"],
                    kind="posterior file",
                ),
                n_samples=n_samples,
                include_mean=include_mean,
                distribution=distribution,
                confidence=confidence,
                seed=seed,
            ),
        )
