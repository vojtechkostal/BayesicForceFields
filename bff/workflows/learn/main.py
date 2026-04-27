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
        from ...bayes.gaussian_process import LGPCommittee
        from ...bayes.learning import LearningProblem
    except ModuleNotFoundError as exc:
        if exc.name == 'torch':
            raise RuntimeError(
                "PyTorch is required for 'bff learn'. Install a CPU or CUDA "
                'build of PyTorch first.'
            ) from exc
        raise
    return LGPCommittee, LearningProblem


def _write_default_plots(results, config: LearnConfig, logger: Logger) -> None:
    try:
        from ...plotting import plot_corner, plot_marginals
    except ModuleNotFoundError as exc:
        logger.warn(
            f"Skipping default posterior plots because {exc.name!r} is not installed.",
            level=1,
        )
        return

    fn_posterior = Path(config.mcmc.posterior).resolve()
    fn_marginals = fn_posterior.with_name('marginals.pdf')
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
    lgp_committee_type, learning_problem_type = _import_learning_stack()
    constraint = ChargeConstraint(config.specs)
    models = {
        name: lgp_committee_type.load(path)
        for name, path in config.models.items()
    }
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
    _write_default_plots(results, config, logger)
    elapsed = time.perf_counter() - workflow_start
    logger.done('Posterior learning', detail=f'finished in {elapsed:.2f}s', level=1)
    return results
