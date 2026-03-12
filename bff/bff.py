from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
)

from .bayes.inference import lgp_hyperopt, initialize_mcmc_sampler
from .bayes.gaussian_process import LGPCommittee
from .structures import (
    Specs, MCMCResults, TrainData
)
from .io.logs import Logger, print_progress_mcmc
from .tools import rdf_sigmoid_mean

import numpy as np
import torch


# ---- Type aliases (keep signatures readable) ----
PathLike = Union[str, Path]
SpecsLike = Union[str, Path, Dict[str, Any], Specs]
ArrayLike = Union[np.ndarray, torch.Tensor]


def train_lgp(
    *train_data: TrainData,
    qoi: Optional[Sequence[str]] = None,
    y_means: Optional[Dict[str, ArrayLike | float | str]] = None,
    n_hyper_max: int = 200,
    fn_hyperparams: Optional[Dict[str, PathLike]] = None,
    committee_size: int = 1,
    test_fraction: float = 0.2,
    device: str = "cuda",
    logger: Optional[Logger] = None,
    **kwargs,
) -> Dict[str, LGPCommittee]:
    """
    Train Local Gaussian Process surrogate models for each QoI.

    Parameters
    ----------
    *train_data : TrainData
        One or more training datasets, each containing QoI outputs.
    qoi : sequence of str, optional
        QoI names to train surrogates for. Defaults to all QoIs found in the data.
    y_means : dict, optional
        Per-QoI mean values for centering. Defaults to 0 for each QoI.
        if "sigmoid", a sigmoid function will be used for RDFs.
    n_hyper_max : int
        Maximum number of samples used for hyperparameter optimisation.
    fn_hyperparams : dict, optional
        Mapping from QoI name to path for loading/saving hyperparameters.
    committee_size : int
        Number of LGP models in each committee ensemble.
    test_fraction : float
        Fraction of data reserved for testing.
    device : str
        Torch device string (e.g. 'cpu', 'cuda', 'cuda:1').
    logger : Logger, optional
        Logger instance. If None, a default Logger is created.
    **kwargs
        Additional keyword arguments forwarded to the MAP optimiser.

    Returns
    -------
    dict[str, LGPCommittee]
        Trained surrogate committee for each QoI.

    Raises
    ------
    ValueError
        If requested QoIs are not present in any training dataset.
    """
    logger = logger or Logger("BFF")
    y_means = y_means or {}
    fn_hyperparams = fn_hyperparams or {}

    # index datasets by the QoIs they contain
    qoi_to_dataset: Dict[str, TrainData] = {
        name: dataset
        for dataset in train_data
        for name in dataset.qoi_names
    }

    available_qoi = sorted(qoi_to_dataset)
    requested_qoi = list(qoi) if qoi is not None else available_qoi

    missing = set(requested_qoi) - set(available_qoi)
    if missing:
        raise ValueError(
            f"QoI not found in any training dataset: {', '.join(sorted(missing))}. "
            f"Available: {', '.join(available_qoi)}"
        )

    logger.info("=== Optimizing LGP surrogates ===", level=0)
    logger.info("", level=0)

    models: Dict[str, LGPCommittee] = {}

    for name in requested_qoi:
        logger.info(f"QoI: {name}", level=1)

        dataset = qoi_to_dataset[name]

        mean = y_means.get(name, 0)
        if name == "rdf" and mean == "sigmoid":
            mean = rdf_sigmoid_mean(dataset.settings, dataset.outputs_ref[name])

        models[name] = lgp_hyperopt(
            X=dataset.inputs,
            y=dataset.outputs[name],
            y_mean=y_means.get(name, 0),
            fn_hyperparams=fn_hyperparams.get(name),
            test_fraction=test_fraction,
            n_hyper=n_hyper_max,
            committee=committee_size,
            observations=dataset.observations[name],
            nuisance=dataset.nuisances.get(name),
            device=device,
            logger=logger,
            opt_kwargs=kwargs,
        )

        logger.info("", level=0)

    return models


def learn(
    y_ref: Dict[str, ArrayLike],
    models: Dict[str, LGPCommittee],
    constraint: Optional[Callable] = None,
    priors_disttype: str = "normal",
    max_iter: int = 1000,
    n_walkers: Optional[int] = None,
    fn_chain: PathLike = "./mcmc.h5",
    fn_priors: Optional[PathLike] = "./priors.yaml",
    fn_tau: Optional[PathLike] = "./tau.yaml",
    restart: bool = True,
    device: str = "cuda",
    logger: Optional[Logger] = None,
    **kwargs,
) -> MCMCResults:
    """
    Run MCMC posterior sampling using pre-trained LGP surrogate models.

    Parameters
    ----------
    y_ref : dict[str, ArrayLike]
        Reference (target) values for each QoI.
    models : dict[str, LGPCommittee]
        Trained LGP surrogate committees, one per QoI.
    constraint : callable, optional
        Callable that returns a boolean mask for valid parameter samples
        (e.g. ChargeConstraint). If None, no constraint is applied.
    priors_disttype : str
        Prior distribution type, either 'normal' or 'uniform'.
    max_iter : int
        Maximum number of MCMC iterations.
    n_walkers : int, optional
        Number of ensemble walkers. Defaults to 5 x n_params if None.
    fn_chain : PathLike
        Path to the HDF5 backend file for storing the MCMC chain.
    fn_priors : PathLike, optional
        Path to write the prior distributions as YAML. Skipped if None.
    fn_tau : PathLike, optional
        Path to write autocorrelation times. Skipped if None.
    restart : bool
        If True, resume from the last saved state in fn_chain when available.
    device : str
        Torch device string (e.g. 'cpu', 'cuda', 'cuda:1').
    logger : Logger, optional
        Logger instance. A default Logger is created if None.
    **kwargs
        Additional keyword arguments forwarded to print_progress_mcmc.

    Returns
    -------
    MCMCResults
        Object containing the sampler chain, priors, and autocorrelation times.

    Raises
    ------
    ValueError
        If the QoI keys in y_ref and models do not match.
    """
    logger = logger or Logger("BFF")

    if set(y_ref) != set(models):
        missing_in_models = set(y_ref) - set(models)
        missing_in_ref = set(models) - set(y_ref)
        raise ValueError(
            "QoI keys in y_ref and models must match.\n"
            + (
                f"  Missing in models: {sorted(missing_in_models)}\n"
                if missing_in_models else ""
            )
            + (
                f"  Missing in y_ref:  {sorted(missing_in_ref)}\n"
                if missing_in_ref else ""
            )
        )

    logger.info("=== Parameter Learning ===", level=0)
    logger.info("", level=0)

    p0, priors, sampler = initialize_mcmc_sampler(
        surrogate=models,
        y_true=y_ref,
        constraint=constraint,
        n_walkers=n_walkers,
        priors_disttype=priors_disttype,
        fn_backend=fn_chain,
        restart=restart,
        device=device,
    )

    print_progress_mcmc(sampler, p0, max_iter, logger=logger, **kwargs)

    tau = sampler.get_autocorr_time(tol=0)
    mcmc = MCMCResults(chain_src=sampler, priors_src=priors, tau_src=tau)

    if fn_priors:
        mcmc.write_priors(fn_priors)
    if fn_tau:
        mcmc.write_tau(fn_tau)

    return mcmc
