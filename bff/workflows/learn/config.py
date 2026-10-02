from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Mapping

from ...domain.systems import validate_system_id
from ...io.utils import load_yaml
from .._shared.config import PathLike, _resolve_path, _strict_bool


@dataclass(frozen=True, slots=True)
class LearnModelConfig:
    model_path: Path
    independent_observations: bool = False
    n_eff: float | None = None
    tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class LearnMCMCConfig:
    priors_disttype: str = "normal"
    total_steps: int = 1500
    warmup: int = 500
    thin: int = 1
    progress_stride: int = 100
    n_walkers: int | None = None
    resume: bool = False
    device: str = "cuda"
    rhat_tol: float = 1.01
    ess_min: int = 100
    include_implicit_charge: bool = False


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
    prior: Path
    posterior: Path
    checkpoint: Path
    specs: Path
    marginals: Path
    qoi_marginals: Path
    corner: Path

    @property
    def stage_owned_files(self) -> tuple[Path, ...]:
        return (
            self.log,
            self.prior,
            self.posterior,
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
    def load(cls, fn_config: PathLike) -> "LearnConfig":
        fn_config = Path(fn_config).resolve()
        base_dir = fn_config.parent
        config = load_yaml(fn_config)
        if not isinstance(config, Mapping):
            raise ValueError("Learn configuration must contain a mapping.")
        unknown_top = set(config) - {"specs", "models", "mcmc", "plots", "output"}
        if unknown_top:
            raise ValueError(
                "Learn configuration contains unsupported key(s): "
                + ", ".join(sorted(unknown_top))
            )
        missing = [key for key in ("specs", "models", "mcmc") if key not in config]
        if missing:
            raise ValueError(
                "Missing required learn section(s): "
                + ", ".join(repr(key) for key in missing)
            )

        mcmc_raw = config["mcmc"]
        if not isinstance(mcmc_raw, Mapping):
            raise ValueError("mcmc must be a mapping.")
        allowed_mcmc = {
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
            "include_implicit_charge",
        }
        unknown_mcmc = set(mcmc_raw) - allowed_mcmc
        if unknown_mcmc:
            raise ValueError(
                "mcmc contains unsupported key(s): "
                + ", ".join(sorted(unknown_mcmc))
            )
        total_steps = int(mcmc_raw.get("total_steps", 1500))
        warmup = int(mcmc_raw.get("warmup", 500))
        thin = int(mcmc_raw.get("thin", 1))
        progress_stride = int(mcmc_raw.get("progress_stride", 100))
        if total_steps < 1:
            raise ValueError("mcmc.total_steps must be positive.")
        if warmup < 0 or warmup >= total_steps:
            raise ValueError("mcmc.warmup must satisfy 0 <= warmup < total_steps.")
        if thin < 1 or progress_stride < 1:
            raise ValueError("mcmc.thin and progress_stride must be positive.")
        n_walkers = mcmc_raw.get("n_walkers")
        if n_walkers is not None and int(n_walkers) < 2:
            raise ValueError("mcmc.n_walkers must be at least 2.")
        mcmc = LearnMCMCConfig(
            priors_disttype=str(mcmc_raw.get("priors_disttype", "normal")),
            total_steps=total_steps,
            warmup=warmup,
            thin=thin,
            progress_stride=progress_stride,
            n_walkers=None if n_walkers is None else int(n_walkers),
            resume=_strict_bool(mcmc_raw.get("resume", False), field="mcmc.resume"),
            device=str(mcmc_raw.get("device", "cuda")),
            rhat_tol=float(mcmc_raw.get("rhat_tol", 1.01)),
            ess_min=int(mcmc_raw.get("ess_min", 100)),
            include_implicit_charge=_strict_bool(
                mcmc_raw.get("include_implicit_charge", False),
                field="mcmc.include_implicit_charge",
            ),
        )

        plots_raw = config.get("plots", {})
        if not isinstance(plots_raw, Mapping):
            raise ValueError("plots must be a mapping.")
        allowed_plots = {
            "max_corner_samples",
            "max_marginal_samples",
            "max_qoi_samples",
            "qoi_batch_size",
            "plot_metadata",
        }
        unknown_plots = set(plots_raw) - allowed_plots
        if unknown_plots:
            raise ValueError(
                "plots contains unsupported key(s): "
                + ", ".join(sorted(unknown_plots))
            )
        plot_values = {
            "max_corner_samples": int(plots_raw.get("max_corner_samples", 2_000)),
            "max_qoi_samples": int(plots_raw.get("max_qoi_samples", 10_000)),
            "qoi_batch_size": int(plots_raw.get("qoi_batch_size", 256)),
        }
        for name, value in plot_values.items():
            if value < 1:
                raise ValueError(f"plots.{name} must be positive.")
        max_marginal_samples_raw = plots_raw.get("max_marginal_samples", 10_000)
        if max_marginal_samples_raw is None:
            max_marginal_samples = None
        else:
            max_marginal_samples = int(max_marginal_samples_raw)
            if max_marginal_samples == -1:
                max_marginal_samples = None
            elif max_marginal_samples < 1:
                raise ValueError(
                    "plots.max_marginal_samples must be positive, -1, or null."
                )
        plot_metadata_raw = plots_raw.get("plot_metadata", {})
        if not isinstance(plot_metadata_raw, Mapping):
            raise ValueError("plots.plot_metadata must be a mapping.")
        plot_metadata: dict[str, dict[str, str]] = {}
        for parameter, metadata in plot_metadata_raw.items():
            if not isinstance(parameter, str) or not parameter:
                raise ValueError(
                    "plots.plot_metadata keys must be non-empty parameter names."
                )
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    f"plots.plot_metadata.{parameter} must be a mapping."
                )
            unknown_metadata = set(metadata) - {"xlabel", "ylabel"}
            if unknown_metadata:
                raise ValueError(
                    f"plots.plot_metadata.{parameter} contains unsupported key(s): "
                    + ", ".join(sorted(unknown_metadata))
                )
            values = {}
            for key, value in metadata.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"plots.plot_metadata.{parameter}.{key} must be a "
                        "non-empty string."
                    )
                values[key] = value.strip()
            plot_metadata[parameter] = values
        plots = LearnPlotsConfig(
            **plot_values,
            max_marginal_samples=max_marginal_samples,
            plot_metadata=plot_metadata,
        )

        models_raw = config["models"]
        if not isinstance(models_raw, Mapping) or not models_raw:
            raise ValueError("models must be a non-empty mapping.")
        models: dict[str, LearnModelConfig] = {}
        for raw_name, model in models_raw.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("Model names must be non-empty strings.")
            name = validate_system_id(raw_name, field="models key")
            if not isinstance(model, Mapping) or "model_path" not in model:
                raise ValueError(f"models.{name} must define model_path.")
            unknown = set(model) - {
                "model_path",
                "independent_observations",
                "n_eff",
                "tolerance",
            }
            if unknown:
                raise ValueError(
                    f"models.{name} contains unsupported key(s): "
                    + ", ".join(sorted(unknown))
                )
            independent = _strict_bool(
                model.get("independent_observations", False),
                field=f"models.{name}.independent_observations",
            )
            n_eff = None if model.get("n_eff") is None else float(model["n_eff"])
            tolerance = (
                None if model.get("tolerance") is None else float(model["tolerance"])
            )
            if n_eff is not None:
                if not isfinite(n_eff) or n_eff <= 0:
                    raise ValueError(
                        f"models.{name}.n_eff must be positive and finite."
                    )
                if independent or tolerance is not None:
                    raise ValueError(
                        f"models.{name}.n_eff cannot be combined with "
                        "independent_observations or tolerance."
                    )
            elif independent:
                if tolerance is not None:
                    raise ValueError(
                        f"models.{name}.tolerance is invalid for independent "
                        "observations."
                    )
            elif tolerance is None or not isfinite(tolerance) or tolerance <= 0:
                raise ValueError(
                    f"Curve model {name!r} requires a positive finite tolerance."
                )
            models[name] = LearnModelConfig(
                model_path=_resolve_path(
                    base_dir, model["model_path"], kind=f"model {name!r} file"
                ),
                independent_observations=independent,
                n_eff=n_eff,
                tolerance=tolerance,
            )

        output_raw = config.get("output", {})
        if not isinstance(output_raw, Mapping):
            raise ValueError("output must be a mapping.")
        unknown_output = set(output_raw) - {"directory", "overwrite"}
        if unknown_output:
            raise ValueError(
                "output contains unsupported key(s): "
                + ", ".join(sorted(unknown_output))
            )
        output_dir = _resolve_path(
            base_dir,
            output_raw.get("directory", "./"),
            must_exist=False,
            kind="learn output directory",
        )
        overwrite = _strict_bool(
            output_raw.get("overwrite", False), field="output.overwrite"
        )
        if mcmc.resume and overwrite:
            raise ValueError("mcmc.resume and output.overwrite cannot both be true.")
        plots_dir = output_dir / "plots"
        outputs_dir = output_dir / "outputs"
        output = LearnOutputConfig(
            directory=output_dir,
            overwrite=overwrite,
            log=output_dir / "learn.log",
            plots_dir=plots_dir,
            outputs_dir=outputs_dir,
            prior=outputs_dir / "prior.pt",
            posterior=outputs_dir / "posterior.pt",
            checkpoint=outputs_dir / "mcmc.ckpt",
            specs=outputs_dir / "specs.yaml",
            marginals=plots_dir / "marginals.pdf",
            qoi_marginals=plots_dir / "qoi-marginals.pdf",
            corner=plots_dir / "corner.pdf",
        )
        return cls(
            fn_config=fn_config,
            specs=_resolve_path(base_dir, config["specs"], kind="specs file"),
            models=models,
            mcmc=mcmc,
            plots=plots,
            output=output,
        )
