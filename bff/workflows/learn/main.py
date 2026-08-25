"""Posterior learning with fixed, validated stage artifacts."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from ...domain.specs import ChargeConstraint
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


def _write_default_plots(results, config: LearnConfig, problem) -> None:
    import numpy as np
    import torch

    from ...bayes.likelihoods import gaussian_log_likelihood_by_qoi
    from ...plotting import plot_corner, plot_marginals, plot_qoi_marginals

    # The sampler checkpoint and posterior already exclude warmup states. Use
    # every saved production state so mandatory plots also work for short CPU
    # validation runs; users can apply diagnostic thinning when loading results.
    results.prepare_samples(discard=0, thin=1)
    plot_marginals(results, config.specs, fn_out=config.output.marginals)
    plot_corner(results, fn_out=config.output.corner)

    prepared = results.prepared_samples
    if results.include_implicit_charge:
        specs = results.specs
        all_names = specs.bounds.names.tolist()
        explicit_indices = [
            all_names.index(name) for name in specs.explicit_bounds.names
        ]
        raw_samples = np.column_stack(
            [
                prepared[:, explicit_indices],
                prepared[:, specs.bounds.n_params :],
            ]
        )
    else:
        raw_samples = prepared.copy()
    raw_samples[:, problem.n_params :] = np.log(
        raw_samples[:, problem.n_params :]
    )
    device = next(iter(problem.models.values())).lgps[0].X_train.device
    theta = torch.as_tensor(raw_samples, dtype=torch.float32, device=device)
    contributions = {
        qoi: values.detach().cpu().numpy()
        for qoi, values in gaussian_log_likelihood_by_qoi(
            theta, problem.to_torch(str(device))
        ).items()
    }
    plot_qoi_marginals(
        results,
        config.specs,
        contributions,
        fn_out=config.output.qoi_marginals,
    )


def main(fn_config: str | Path):
    workflow_start = time.perf_counter()
    config = LearnConfig.load(fn_config)
    _prepare_output(config)
    logger = Logger(
        "learn",
        str(config.output.log),
        mode="a" if config.mcmc.resume else "w",
    )
    logger.section(
        "Posterior Learning (resumed)"
        if config.mcmc.resume
        else "Posterior Learning"
    )
    logger.kv("Config", config.fn_config)
    logger.kv("Output directory", config.output.directory)
    logger.kv("Specs", config.specs)
    logger.kv("Models", len(config.models))
    logger.kv("Device", config.mcmc.device)
    logger.kv("Resume", config.mcmc.resume)
    logger.blank()

    try:
        from ...bayes.effective_observations import estimate_curve_n_eff
        from ...bayes.gaussian_process import LGPCommittee
        from ...bayes.learning import LearningProblem
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for 'bff learn'. Install a CPU or CUDA "
                "build of PyTorch first."
            ) from exc
        raise
    constraint = ChargeConstraint(config.specs)
    models = {}
    model_fingerprints = {}
    target_configuration = {}
    for name, model_config in config.models.items():
        model = LGPCommittee.load(model_config.model_path)
        if model_config.n_eff is not None:
            model.n_eff = model_config.n_eff
        elif model_config.independent_observations:
            model.n_eff = float(model.reference_values.size)
        else:
            curves = model.reference_values.reshape(model.n_curves, -1)
            model.n_eff = sum(
                estimate_curve_n_eff(curve, tolerance=model_config.tolerance)
                for curve in curves
            )
        models[name] = model
        model_fingerprints[name] = file_sha256(model_config.model_path)
        target_configuration[name] = {
            "independent_observations": model_config.independent_observations,
            "n_eff": model_config.n_eff,
            "tolerance": model_config.tolerance,
            "effective_observations": float(model.n_eff),
        }
        logger.kv(f"{name} effective observations", f"{model.n_eff:.3f}")

    logger.blank()
    problem = LearningProblem.from_models(models, constraint=constraint)
    results = problem.learn(
        priors_disttype=config.mcmc.priors_disttype,
        total_steps=config.mcmc.total_steps,
        warmup=config.mcmc.warmup,
        thin=config.mcmc.thin,
        progress_stride=config.mcmc.progress_stride,
        n_walkers=config.mcmc.n_walkers,
        fn_posterior=config.output.posterior,
        fn_checkpoint=config.output.checkpoint,
        fn_priors=config.output.prior,
        resume=config.mcmc.resume,
        device=config.mcmc.device,
        logger=logger,
        rhat_tol=config.mcmc.rhat_tol,
        ess_min=config.mcmc.ess_min,
        include_implicit_charge=config.mcmc.include_implicit_charge,
        compatibility={
            "ordered_models": list(config.models),
            "model_fingerprints": model_fingerprints,
            "target_configuration": target_configuration,
        },
    )
    for artifact in (
        config.output.specs,
        config.output.prior,
        config.output.posterior,
        config.output.checkpoint,
    ):
        if not artifact.is_file():
            raise RuntimeError(
                f"Mandatory learning artifact was not written: {artifact}"
            )

    logger.blank()
    _write_default_plots(results, config, problem)
    for plot in (
        config.output.marginals,
        config.output.qoi_marginals,
        config.output.corner,
    ):
        if not plot.is_file():
            raise RuntimeError(f"Mandatory learning plot was not written: {plot}")
        logger.kv("Plot", plot)
    logger.blank()
    logger.done(
        "Posterior learning",
        detail=f"finished in {time.perf_counter() - workflow_start:.2f}s",
        level=1,
    )
    return results
