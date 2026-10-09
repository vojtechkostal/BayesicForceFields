from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ...domain.systems import validate_system_id
from ..config import ConfigSection, PathLike, load_config


@dataclass(frozen=True, slots=True)
class LearnModelConfig:
    model_path: Path
    # Accepted deviation from the reference, in the QoI's units.
    tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class LearnMCMCConfig:
    priors_disttype: str = "normal"
    total_steps: int = 1500
    warmup: int = 500
    thin: int = 1
    progress_stride: int = 100
    n_walkers: int | None = None
    resume: bool = False
    device: str = "auto"
    rhat_tol: float = 1.01
    ess_min: int = 400


@dataclass(frozen=True, slots=True)
class LearnPlotsConfig:
    max_corner_samples: int = 2_000
    max_marginal_samples: int | None = 10_000
    max_qoi_samples: int = 10_000
    qoi_batch_size: int = 256
    plot_metadata: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LearnOutputConfig:
    directory: Path
    overwrite: bool
    log: Path
    plots_dir: Path
    outputs_dir: Path
    results: Path
    checkpoint: Path
    specs: Path
    marginals: Path
    qoi_marginals: Path
    corner: Path

    @property
    def stage_owned_files(self) -> tuple[Path, ...]:
        return (
            self.log,
            self.results,
            self.checkpoint,
            self.specs,
            self.marginals,
            self.qoi_marginals,
            self.corner,
        )


@dataclass(frozen=True, slots=True)
class LearnConfig:
    fn_config: Path
    specs: Path
    models: dict[str, LearnModelConfig]
    mcmc: LearnMCMCConfig
    plots: LearnPlotsConfig
    output: LearnOutputConfig

    @classmethod
    def load(cls, fn_config: PathLike) -> LearnConfig:
        config = load_config(
            fn_config,
            stage="learn",
            allowed=("specs", "models", "mcmc", "plots", "output"),
            required=("specs", "models"),
        )
        mcmc = _load_mcmc(config)
        output = config.section("output", allowed=("directory", "overwrite"))
        overwrite = output.boolean("overwrite", False)
        if mcmc.resume and overwrite:
            raise ValueError("mcmc.resume and output.overwrite cannot both be true.")
        output_dir = output.path("directory", "./", must_exist=False)
        plots_dir = output_dir / "plots"
        outputs_dir = output_dir / "outputs"
        return cls(
            fn_config=Path(fn_config).resolve(),
            specs=config.path("specs"),
            models=_load_models(config),
            mcmc=mcmc,
            plots=_load_plots(config),
            output=LearnOutputConfig(
                directory=output_dir,
                overwrite=overwrite,
                log=output_dir / "learn.log",
                plots_dir=plots_dir,
                outputs_dir=outputs_dir,
                results=outputs_dir / "results.pt",
                checkpoint=outputs_dir / "mcmc.ckpt",
                specs=outputs_dir / "specs.yaml",
                marginals=plots_dir / "marginals.pdf",
                qoi_marginals=plots_dir / "qoi-marginals.pdf",
                corner=plots_dir / "corner.pdf",
            ),
        )


def _load_mcmc(config: ConfigSection) -> LearnMCMCConfig:
    mcmc = config.section(
        "mcmc",
        allowed=(
            "priors_disttype",
            "total_steps",
            "warmup",
            "thin",
            "progress_stride",
            "n_walkers",
            "resume",
            "device",
            "rhat_tol",
            "ess_min",
        ),
    )
    total_steps = mcmc.integer("total_steps", 1500, minimum=1)
    warmup = mcmc.integer("warmup", 500, minimum=0)
    if warmup >= total_steps:
        raise ValueError(
            f"mcmc.warmup ({warmup}) must be smaller than mcmc.total_steps "
            f"({total_steps})."
        )
    return LearnMCMCConfig(
        priors_disttype=mcmc.string(
            "priors_disttype", "normal", choices=("normal", "uniform")
        ),
        total_steps=total_steps,
        warmup=warmup,
        thin=mcmc.integer("thin", 1, minimum=1),
        progress_stride=mcmc.integer("progress_stride", 100, minimum=1),
        n_walkers=mcmc.integer("n_walkers", None, minimum=2),
        resume=mcmc.boolean("resume", False),
        device=mcmc.device("device", "auto"),
        rhat_tol=mcmc.number("rhat_tol", 1.01, minimum=1, exclusive=True),
        ess_min=mcmc.integer("ess_min", 400, minimum=1),
    )


def _load_plots(config: ConfigSection) -> LearnPlotsConfig:
    plots = config.section(
        "plots",
        allowed=(
            "max_corner_samples",
            "max_marginal_samples",
            "max_qoi_samples",
            "qoi_batch_size",
            "plot_metadata",
        ),
    )
    max_marginal_samples = plots.integer(
        "max_marginal_samples", 10_000, minimum=1, special=(-1,)
    )
    plot_metadata: dict[str, dict[str, str]] = {}
    if "plot_metadata" in plots:
        for parameter, labels in plots.named_sections(
            "plot_metadata", allowed=("xlabel", "ylabel")
        ).items():
            plot_metadata[parameter] = {
                key: labels.string(key).strip() for key in labels.raw
            }
    return LearnPlotsConfig(
        max_corner_samples=plots.integer("max_corner_samples", 2_000, minimum=1),
        # -1 means every posterior sample.
        max_marginal_samples=(
            None if max_marginal_samples == -1 else max_marginal_samples
        ),
        max_qoi_samples=plots.integer("max_qoi_samples", 10_000, minimum=1),
        qoi_batch_size=plots.integer("qoi_batch_size", 256, minimum=1),
        plot_metadata=plot_metadata,
    )


def _load_models(config: ConfigSection) -> dict[str, LearnModelConfig]:
    """Read ``models``: each surrogate and its accepted deviation."""
    models: dict[str, LearnModelConfig] = {}
    for name, model in config.named_sections(
        "models", allowed=("model_path", "tolerance"), required=("model_path",)
    ).items():
        validate_system_id(name, field=model.where)
        models[name] = LearnModelConfig(
            model_path=model.path("model_path"),
            tolerance=model.number("tolerance", 0.0, minimum=0),
        )
    return models
