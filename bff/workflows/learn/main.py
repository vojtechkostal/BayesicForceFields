"""Posterior learning with fixed, validated stage artifacts."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ...domain.specs import Specs
from ...io.logs import Logger
from ...io.utils import file_sha256
from .config import LearnConfig


def _prepare_output(config: LearnConfig) -> None:
    output = config.output
    existing = [path for path in output.stage_owned_files if path.exists()]
    if config.mcmc.resume:
        if not output.checkpoint.is_file():
            raise ValueError(
                f"mcmc.resume=true requires checkpoint {output.checkpoint}; "
                "set resume: false to start a new run."
            )
        if not output.specs.is_file():
            raise ValueError(
                "mcmc.resume=true requires the copied specifications at "
                f"{output.specs}; restore them or start a new run."
            )
        if file_sha256(config.specs) != file_sha256(output.specs):
            raise ValueError(
                "Configured specs do not match the specifications copied into "
                f"the learning outputs: {output.specs}."
            )
    elif output.overwrite:
        for path in existing:
            if path.is_file():
                path.unlink()
            else:
                raise ValueError(
                    f"Stage-owned output path is not a file: {path}. Remove or "
                    "rename it before learning."
                )
    elif existing:
        raise ValueError(
            "Learn stage-owned output(s) already exist: "
            + ", ".join(str(path) for path in existing)
            + ". Set output.overwrite: true for a fresh run or mcmc.resume: true."
        )
    output.directory.mkdir(parents=True, exist_ok=True)
    output.plots_dir.mkdir(parents=True, exist_ok=True)
    output.outputs_dir.mkdir(parents=True, exist_ok=True)
    if not config.mcmc.resume:
        shutil.copy2(config.specs, output.specs)


def _write_plots(results, config: LearnConfig) -> None:
    """Write the three default figures."""
    import matplotlib.pyplot as plt

    from ...plotting import plot_corner, plot_marginals, plot_qoi_marginals

    plots = config.plots
    for fn_out, figure in (
        (
            config.output.marginals,
            plot_marginals(
                results,
                max_samples=plots.max_marginal_samples,
                plot_metadata=plots.plot_metadata,
            ),
        ),
        (
            config.output.qoi_marginals,
            plot_qoi_marginals(results, plot_metadata=plots.plot_metadata),
        ),
        (
            config.output.corner,
            plot_corner(results, max_samples=plots.max_corner_samples),
        ),
    ):
        figure.savefig(fn_out, bbox_inches="tight")
        plt.close(figure)


def _log_posterior(results, logger: Logger) -> None:
    """One line per parameter: posterior mean, standard deviation, and MAP."""
    summary = results.summary()
    for name in results.names:
        entry = summary[name]
        logger.info(
            f"{name}: {entry['mean']:.4g} +- {entry['std']:.2g} "
            f"(MAP {entry['map']:.4g})",
            level=2,
        )
    converged = results.info["mcmc"]["converged"]
    logger.kv("Converged", "yes" if converged else "no (stopped at total_steps)")


def main(fn_config: str | Path):
    workflow_start = time.perf_counter()
    config = LearnConfig.load(fn_config)
    _prepare_output(config)
    logger = Logger(
        "learn",
        str(config.output.log),
        mode="a" if config.mcmc.resume else "w",
    )
    logger.section("Learn (resumed)" if config.mcmc.resume else "Learn")
    logger.kv("Config", config.fn_config)
    logger.kv("Specs", config.specs)
    logger.kv("Models", ", ".join(config.models))

    try:
        from ...bayes.gaussian_process import LGPCommittee
        from ...bayes.learning import LearningProblem
        from ...bayes.utils import resolve_device
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for 'bff learn'. Install a CPU or CUDA "
                "build of PyTorch first."
            ) from exc
        raise
    device = resolve_device(config.mcmc.device)
    logger.kv("Device", device)
    logger.kv("Output", config.output.directory)
    logger.blank()
    specs = Specs(config.specs)
    models = {}
    model_fingerprints = {}
    target_configuration = {}
    for name, model_config in config.models.items():
        model = LGPCommittee.load(model_config.model_path)
        expected = specs.explicit_names
        if model.parameter_names is not None and model.parameter_names != expected:
            raise ValueError(
                f"models.{name} was fitted to parameters "
                f"{list(model.parameter_names)}, but {config.specs} samples "
                f"{list(expected)}; use the specs.yaml "
                "of the campaign the model's QoI dataset came from."
            )
        model.tolerance = model_config.tolerance
        models[name] = model
        model_fingerprints[name] = file_sha256(model_config.model_path)
        target_configuration[name] = {
            "tolerance": model_config.tolerance,
            "effective_observations": float(model.n_eff),
        }
        logger.kv(
            name,
            f"{model.n_eff:.1f} effective observations, "
            f"tolerance {model_config.tolerance:g}",
        )

    logger.blank()
    problem = LearningProblem.from_models(models, specs=specs)
    results = problem.learn(
        priors_disttype=config.mcmc.priors_disttype,
        total_steps=config.mcmc.total_steps,
        warmup=config.mcmc.warmup,
        thin=config.mcmc.thin,
        progress_stride=config.mcmc.progress_stride,
        n_walkers=config.mcmc.n_walkers,
        fn_results=config.output.results,
        fn_checkpoint=config.output.checkpoint,
        resume=config.mcmc.resume,
        device=device,
        logger=logger,
        rhat_tol=config.mcmc.rhat_tol,
        ess_min=config.mcmc.ess_min,
        max_qoi_samples=config.plots.max_qoi_samples,
        qoi_batch_size=config.plots.qoi_batch_size,
        compatibility={
            "ordered_models": list(config.models),
            "model_fingerprints": model_fingerprints,
            "target_configuration": target_configuration,
        },
    )
    output = config.output
    for artifact in (output.specs, output.results, output.checkpoint):
        if not artifact.is_file():
            raise RuntimeError(
                f"Mandatory learning artifact was not written: {artifact}"
            )

    logger.blank()
    _log_posterior(results, logger)
    _write_plots(results, config)
    for plot in (
        config.output.marginals,
        config.output.qoi_marginals,
        config.output.corner,
    ):
        if not plot.is_file():
            raise RuntimeError(f"Mandatory learning plot was not written: {plot}")
    logger.done("Plots", detail=str(config.output.plots_dir))
    logger.blank()
    logger.done(
        "Learn",
        detail=f"{time.perf_counter() - workflow_start:.1f} s | "
        f"{config.output.results}",
    )
    return results
