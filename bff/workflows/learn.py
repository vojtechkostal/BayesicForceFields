"""Workflow entry point for posterior learning from trained surrogates."""

import sys
from pathlib import Path
from typing import Union

from ..domain.specs import ChargeConstraint
from ..io.logs import Logger
from .configs import LearnConfig


PathLike = Union[str, Path]


def _load_models(
    config: LearnConfig,
    lgp_committee_type,
) -> dict[str, object]:
    return {
        name: lgp_committee_type.load(path)
        for name, path in config.models.items()
    }


def main(fn_config: PathLike) -> None:
    try:
        from ..bayes.gaussian_process import LGPCommittee
        from ..bayes.learning import InferenceProblem
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise RuntimeError(
                "PyTorch is required for 'bff learn'. Install a CPU or CUDA "
                "build of PyTorch first."
            ) from exc
        raise

    config = LearnConfig.load(fn_config)
    logger = Logger("BFF", str(config.log), mode="w")
    constraint = ChargeConstraint(config.specs)
    models = _load_models(config, LGPCommittee)
    problem = InferenceProblem.from_models(models, constraint=constraint)
    problem.infer(
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


if __name__ == "__main__":
    fn_config = sys.argv[1]
    main(fn_config)
