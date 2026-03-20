import torch
import numpy as np
from functools import partial
from pathlib import Path

from .gaussian_process import LocalGaussianProcess, LGPCommittee
from .likelihoods import loo_log_likelihood, gaussian_log_likelihood
from .priors import Prior, Priors
from .posterior import log_posterior
from ..io.logs import Logger
from .utils import (
    initialize_walkers,
    check_device, check_tensor,
    train_test_split,
    find_map,
    laplace_approximation
)
from ..mcmc.proposal import AdaptiveGaussianProposal
from ..mcmc.sampler import Sampler

from typing import Union, List, Callable, Dict, Optional


PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, List[float], torch.Tensor]


def initialize_mcmc_sampler(
    surrogate: Dict[str, LGPCommittee],
    y_true: Dict[str, np.ndarray],
    constraint: Callable[[ArrayLike], ArrayLike] = None,
    n_walkers: Optional[int] = None,
    priors_disttype: str = 'normal',
    device: str = 'cuda:0'
) -> tuple[np.ndarray, Priors, Sampler]:
    """
    Initialize the MCMC sampler for parameter optimization.

    Parameters
    ----------
    surrogate : dict
        Dictionary of surrogate models used for predictions.
    y_true : dict
        True target values for the model for all QoI.
    constraint : Callable, optional
        Constraint function to apply to parameters.
        Signature should be `constraint(theta: np.ndarray) -> np.ndarray`.
        Defaults to None.
    n_walkers : int, optional
        Number of walkers in the MCMC sampler.
        If None, defaults to 5 times the number of parameters.
    priors_disttype : str, optional
        Type of distribution for the priors, either 'normal' or 'uniform'.
        Defaults to 'normal'.
    device : str, optional
        Device on which to perform computations (e.g., 'cuda:0' or 'cpu').
        Defaults to 'cuda:0'.

    Returns
    -------
    tuple[np.ndarray, Priors, Sampler]
        Initial walker positions, priors, and the initialized sampler.
    """

    # Determine parameter bounds
    if constraint is None:
        # TODO: define broad but meaningfull bounds
        n_params = [s.n_params for s in surrogate.values()][0]
        bounds = np.tile([-1e5, 1e5], (n_params, 1))
    else:
        bounds = constraint.explicit_bounds

    # priors
    param_names = None
    if constraint is not None and hasattr(constraint, "bounds") and hasattr(constraint, "implicit_param"):
        param_names = constraint.bounds.without(constraint.implicit_param).names.tolist()

    priors = Priors.from_bounds(
        bounds=bounds,
        dist_type=priors_disttype,
        n_nuisance=len(surrogate),
        names=param_names,
        nuisance_names=[f"log_sigma_{qoi}" for qoi in surrogate],
    )

    # Initialize backend
    n_dim = len(priors)
    n_walkers = 5 * n_dim if n_walkers is None else n_walkers
    p0 = initialize_walkers(priors.distributions, n_walkers, constraint)
    cov = check_tensor(
        torch.diag(torch.tensor(priors.scales, dtype=torch.float32) ** 2),
        device=device
    )
    proposal = AdaptiveGaussianProposal(cov, device=device)

    y_true = {qoi: check_tensor(y, device) for qoi, y in y_true.items()}

    # likelihood
    log_likelihood = partial(
        gaussian_log_likelihood,
        y_true=y_true,
        surrogate=surrogate,
        constraint=constraint
    )

    # posterior
    log_probability = partial(
        log_posterior,
        priors=priors,
        log_likelihood_fn=log_likelihood,
        device=device
    )

    # Initialize the sampler
    sampler = Sampler(
        log_prob=log_probability,
        proposal=proposal,
        device=device,
        dtype=torch.float32

    )

    return p0, priors, sampler


def lgp_hyperopt(
    X: torch.Tensor,
    y: torch.Tensor,
    y_mean: torch.Tensor,
    test_fraction: float,
    n_hyper: int,
    committee: int,
    observations: int,
    nuisance: float,
    fn_out: PathLike,
    device: str,
    logger: Logger,
    opt_kwargs: Dict[str, Union[int, float, str]]
) -> LGPCommittee:
    """
    Perform hyperparameter optimization for Local Gaussian Processes (LGPs)
    and construct a committee of LGP models.

    Parameters
    ----------
    X : torch.Tensor
        Input features of shape (n_samples, n_features).
    y : torch.Tensor
        Target values of shape (n_samples, output_dim).
    y_mean : torch.Tensor
        Mean of the target values (used for centering).
    test_fraction : float
        Fraction of the data to reserve for testing.
    n_hyper : int
        Maximum number of data points used for hyperparameter optimization.
    committee : int
        Number of LGP models in the ensemble.
    observations : int
        Number of observations.
    nuisance : float
        Nuisance parameter value.
    fn_out : PathLike
        Path to save the trained LGP committee model.
    device : str
        Device to perform computations on (e.g., 'cpu' or 'cuda').
    logger : callable
        Logger for reporting progress.
    opt_kwargs : dict
        Additional arguments passed to the MAP optimizer.

    Returns
    -------
    LGPCommittee
        Committee of Local Gaussian Process models.
    """

    check_device(device)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction)

    # Optimization of the hyperparameters
    n_hyper = min(n_hyper, len(X_train))

    X_hyper = check_tensor(X_train[:n_hyper], device='cpu')
    y_hyper = check_tensor(y_train[:n_hyper], device='cpu')
    y_mean = check_tensor(y_mean, device='cpu')

    n_params = X.shape[1]
    priors = Priors(
        [Prior("normal", -2.0, 2.0, name=f"length_{i}") for i in range(n_params)]
        + [
            Prior("normal", -2.0, 2.0, name="width"),
            Prior("normal", -2.0, 3.0, name="noise"),
        ]
    )
    p0 = torch.tensor(priors.means, dtype=torch.float32)

    log_likelihood = partial(
        loo_log_likelihood,
        X=X_hyper,
        y=y_hyper-y_mean)

    log_probability = partial(
        log_posterior,
        priors=priors,
        log_likelihood_fn=log_likelihood,
        device='cpu',
        numpy_output=False)

    map_theta = find_map(log_probability, p0, logger=logger, **opt_kwargs)

    if committee > 1:
        cov = laplace_approximation(log_probability, map_theta, device='cpu')
        hyper_dist = torch.distributions.MultivariateNormal(map_theta, cov)
        hyper_samples = hyper_dist.sample((committee,))
    else:
        hyper_samples = map_theta.unsqueeze(0)

    # Split hyperparameters
    hyper_samples = hyper_samples.exp()
    lengths = hyper_samples[:, :-2]
    widths = hyper_samples[:, -2]
    sigmas = hyper_samples[:, -1]

    # Create committee of LGPs
    logger.info(f'Committee: 0/{committee}', level=2, overwrite=True)
    lgps = []
    for i, (l, w, s) in enumerate(zip(lengths, widths, sigmas), start=1):
        lgps.append(LocalGaussianProcess(X_train, y_train, y_mean, l, w, s, device))
        logger.info(f'Committee: {i}/{committee}', level=2, overwrite=True)

    lgp_committee = LGPCommittee(lgps, observations, nuisance)

    # Validate the surrogate
    lgp_committee.validate(X_test, y_test)

    logger.info(
        f'Committee: {committee} (100%) | MAPE = {lgp_committee.error:.2f}%',
        level=2
    )

    # Save models
    if fn_out:
        lgp_committee.write(fn_out)

    return lgp_committee
