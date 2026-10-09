from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..campaign.config import SimulationCampaignConfig, load_campaign_config
from ..config import PathLike

PosteriorDistribution = Literal["empirical", "normal", "uniform", "kde"]


@dataclass(frozen=True)
class ValidatePosteriorConfig:
    file: Path
    n_samples: int = 10
    include_mean: bool = False
    include_map: bool = False
    distribution: PosteriorDistribution = "normal"
    confidence: float = 0.9
    seed: int | None = None


@dataclass(frozen=True, kw_only=True)
class ValidateConfig(SimulationCampaignConfig):
    specs: Path | None
    parameters: Path | None
    posterior: ValidatePosteriorConfig | None

    @classmethod
    def load(cls, fn_config: PathLike) -> ValidateConfig:
        config, common = load_campaign_config(
            fn_config,
            stage="validate",
            stage_keys={"specs", "parameters", "posterior"},
        )
        if ("parameters" in config) == ("posterior" in config):
            raise ValueError(
                "validate requires exactly one of 'parameters' (with 'specs') or "
                "'posterior'."
            )

        if "parameters" in config:
            return cls(
                **common,
                specs=config.path("specs"),
                parameters=config.path("parameters"),
                posterior=None,
            )

        if "specs" in config:
            raise ValueError(
                "specs is only used with 'parameters'; posterior validation reads "
                "the specifications stored in the posterior."
            )
        posterior = config.section(
            "posterior",
            allowed=(
                "file",
                "n_samples",
                "include_mean",
                "include_map",
                "distribution",
                "confidence",
                "seed",
            ),
            required=("file",),
        )
        n_samples = posterior.integer("n_samples", 10, minimum=0)
        include_mean = posterior.boolean("include_mean", False)
        include_map = posterior.boolean("include_map", False)
        if n_samples == 0 and not (include_mean or include_map):
            raise ValueError(
                "posterior must produce at least one sample; set n_samples above "
                "zero, include_mean: true, or include_map: true."
            )
        return cls(
            **common,
            specs=None,
            parameters=None,
            posterior=ValidatePosteriorConfig(
                file=posterior.path("file"),
                n_samples=n_samples,
                include_mean=include_mean,
                include_map=include_map,
                distribution=posterior.string(
                    "distribution",
                    "normal",
                    choices=("empirical", "normal", "uniform", "kde"),
                ),
                confidence=posterior.number(
                    "confidence", 0.9, minimum=0, maximum=1, exclusive=True
                ),
                seed=posterior.integer("seed", None, minimum=0),
            ),
        )
