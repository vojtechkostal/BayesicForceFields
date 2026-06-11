"""Workflow entry point for posterior learning from trained surrogates."""

import time
from pathlib import Path
from typing import Union

from ...domain.specs import ChargeConstraint
from ...io.logs import Logger
from .config import LearnConfig

PathLike = Union[str, Path]


def _import_learning_stack():
    try:
        from ...bayes.effective_observations import estimate_curve_n_eff
        from ...bayes.gaussian_process import LGPCommittee
        from ...bayes.learning import LearningProblem
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            raise RuntimeError(
                "PyTorch is required for 'bff learn'. Install a CPU or CUDA "
                'build of PyTorch first.'
            ) from exc
        raise
    return LGPCommittee, LearningProblem, estimate_curve_n_eff


def _write_default_plots(
    results,
    config: LearnConfig,
    problem,
    logger: Logger,
) -> None:
    try:
        import numpy as np
        import torch

        from ...bayes.likelihoods import gaussian_log_likelihood_by_qoi
        from ...plotting import (
            plot_corner,
            plot_marginals,
            plot_qoi_marginals,
        )
    except ModuleNotFoundError as exc:
        logger.warn(
            f"Skipping default posterior plots because {exc.name!r} is not installed.",
            level=1,
        )
        return

    fn_posterior = Path(config.mcmc.posterior).resolve()
    fn_marginals = fn_posterior.with_name('marginals.pdf')
    fn_qoi_marginals = fn_posterior.with_name('qoi-marginals.pdf')
    fn_corner = fn_posterior.with_name('corner.pdf')

    try:
        results.prepare_samples()
        plot_marginals(results, config.specs, fn_out=fn_marginals)
        plot_corner(results, fn_out=fn_corner)
    except Exception as exc:
        logger.warn(f"Skipping default posterior plots: {exc}", level=1)
        return

    logger.kv('Marginals plot', fn_marginals.resolve(), level=1)
    logger.kv('Corner plot', fn_corner.resolve(), level=1)

    try:
        prepared = results.prepared_samples
        if results.include_implicit_charge:
            specs = results.specs
            all_names = specs.bounds.names.tolist()
            explicit_indices = [
                all_names.index(name)
                for name in specs.explicit_bounds.names
            ]
            raw_samples = np.column_stack([
                prepared[:, explicit_indices],
                prepared[:, specs.bounds.n_params:],
            ])
        else:
            raw_samples = prepared.copy()
        raw_samples[:, problem.n_params:] = np.log(
            raw_samples[:, problem.n_params:]
        )

        device = next(iter(problem.models.values())).lgps[0].X_train.device
        theta = torch.as_tensor(
            raw_samples,
            dtype=torch.float32,
            device=device,
        )
        contributions = {
            qoi: values.detach().cpu().numpy()
            for qoi, values in gaussian_log_likelihood_by_qoi(
                theta,
                problem.to_torch(str(device)),
            ).items()
        }
        plot_qoi_marginals(
            results,
            config.specs,
            contributions,
            fn_out=fn_qoi_marginals,
        )
    except Exception as exc:
        logger.warn(f"Skipping QoI marginals plot: {exc}", level=1)
        return

    logger.kv('QoI marginals plot', fn_qoi_marginals.resolve(), level=1)


def main(fn_config: PathLike):
    workflow_start = time.perf_counter()
    config = LearnConfig.load(fn_config)
    logger = Logger('learn', str(config.log), mode='w')
    logger.section('Posterior Learning')
    logger.kv('Config', Path(fn_config).resolve())
    logger.kv('Log file', config.log.resolve())
    logger.kv('Specs', Path(config.specs).resolve())
    logger.kv('Models', len(config.models))
    logger.kv('Device', config.mcmc.device)
    if (
        config.mcmc.restart
        and config.mcmc.checkpoint is not None
        and Path(config.mcmc.checkpoint).resolve().exists()
    ):
        logger.warn(
            'Posterior learning is configured to reuse an existing checkpoint '
            'if one is found.'
        )
    logger.blank()
    (
        lgp_committee_type,
        learning_problem_type,
        estimate_curve_n_eff,
    ) = _import_learning_stack()
    constraint = ChargeConstraint(config.specs)
    models = {}
    for name, model_config in config.models.items():
        model = lgp_committee_type.load(model_config.model_path)
        if model_config.n_eff is not None:
            model.n_eff = model_config.n_eff
        elif model_config.independent_observations:
            model.n_eff = float(model.reference_values.size)
        else:
            tolerance = model_config.tolerance
            if tolerance is None:
                raise ValueError(f"Model {name!r} is missing 'tolerance'.")
            curves = model.reference_values.reshape(model.n_curves, -1)
            model.n_eff = sum(
                estimate_curve_n_eff(curve, tolerance=tolerance)
                for curve in curves
            )
        models[name] = model
        logger.kv(
            f'{name} effective observations',
            f'{model.n_eff:.3f}',
            level=1,
        )
    problem = learning_problem_type.from_models(models, constraint=constraint)
    results = problem.learn(
        priors_disttype=config.mcmc.priors_disttype,
        total_steps=config.mcmc.total_steps,
        warmup=config.mcmc.warmup,
        thin=config.mcmc.thin,
        progress_stride=config.mcmc.progress_stride,
        n_walkers=config.mcmc.n_walkers,
        fn_posterior=config.mcmc.posterior,
        fn_checkpoint=config.mcmc.checkpoint,
        fn_priors=config.mcmc.priors,
        restart=config.mcmc.restart,
        device=config.mcmc.device,
        logger=logger,
        rhat_tol=config.mcmc.rhat_tol,
        ess_min=config.mcmc.ess_min,
        include_implicit_charge=config.mcmc.include_implicit_charge,
    )
    _write_default_plots(results, config, problem, logger)
    elapsed = time.perf_counter() - workflow_start
    logger.done('Posterior learning', detail=f'finished in {elapsed:.2f}s', level=1)
    return results
