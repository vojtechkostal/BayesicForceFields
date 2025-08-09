import emcee
import torch
import numpy as np
from torch.distributions import Normal, Uniform
from functools import partial
from pathlib import Path

from .gaussian_process import LocalGaussianProcess, LGPCommittee
from .likelihoods import loo_log_likelihood, gaussian_log_likelihood
from ..io.utils import load_yaml, save_yaml
from .utils import (
    initialize_backend, initialize_walkers,
    check_device, check_tensor,
    train_test_split,
    find_map, laplace_approximation
)


def define_param_priors(
    param_bounds: dict[list], QoI: dict, dist_type: str
) -> dict[list]:
    """
    Define priors for the model parameters and nuisance parameters.

    Parameters
    ----------
    param_bounds : dict[list]
        Dict of tuples defining the bounds for each parameter.
    QoI : dict
        Dictionary of quantities of interest (QoI).
    dist_type : str
        Type of distribution for the priors, either 'normal' or 'uniform'.

    Returns
    -------
    dict
        Dict of prior distributions for the parameters and nuisance parameters.
    """

    if dist_type == 'normal':
        param_priors = {
            param: Normal(np.mean(bound), 1 / 5 * np.diff(bound)[0])
            for param, bound in param_bounds.bounds.items()
        }
    elif dist_type == 'uniform':
        param_priors = {
            param: Uniform(bound[0], bound[1], validate_args=False)
            for param, bound in param_bounds.bounds.items()
        }
    else:
        raise ValueError(
            f'Unknown prior type "{dist_type}". '
            'Options are "normal" or "uniform".'
        )

    nuisance_priors = {f'nuisance {q}': Normal(-2, 2) for q in QoI}

    return param_priors | nuisance_priors


def define_hyper_priors(n_params) -> list[Normal]:
    """
    Define Gaussian priors for model parameters:
    - `lengths` (per parameter): N(-2, 2)
    - `width`: N(-2, 2)
    - `noise`: N(-2, 3)

    Parameters
    ----------
        n_params : int
            Number of model parameters (length scales).

    Returns
    -------
        List[Normal]
            List of Normal distributions representing the priors.
    """

    lengths_priors = {f'length {i}': Normal(-2, 2) for i in range(n_params)}
    width_prior = {'width': Normal(-2, 2)}
    sigma_prior = {'sigma': Normal(-2, 3)}
    return lengths_priors | width_prior | sigma_prior


def log_prior(
    theta: torch.Tensor, priors: list[torch.distributions.Distribution]
) -> torch.Tensor:
    """
    Compute the log prior probabilities for the parameters.

    Parameters
    ----------
    theta : torch.Tensor
        Tensor of shape (n_samples, n_params) containing parameter values.
    priors : list of torch.distributions.Distribution
        List of prior distributions for each parameter.

    Returns
    -------
    torch.Tensor
        Log prior probabilities for each parameter set.
    """

    device = theta.device
    log_probabilities = [
        p.log_prob(theta[:, i]).to(device) for i, p in enumerate(priors)
    ]
    return torch.stack(log_probabilities, dim=1).sum(dim=1)


def log_posterior(
    theta: torch.Tensor,
    priors: list[torch.distributions.Distribution],
    log_likelihood_fn: callable,
    device: str,
    numpy_output: bool = True
) -> np.ndarray:

    """
    Computes the log-posterior = log-prior + log-likelihood.

    Parameters
    ----------
    theta : torch.Tensor
        Tensor of parameters for which to compute the log-posterior.
    priors : list[torch.distributions.Distribution]
        List of prior distributions for the parameters.
    log_likelihood_fn : callable
        Function that computes the log-likelihood given the parameters.
        It must have signature `log_likelihood_fn(theta: torch.Tensor) -> torch.Tensor`.
    device : str
        Device on which the computations should be performed (e.g., 'cuda:0' or 'cpu').
    numpy_output : bool, optional
        If True, returns the log-posterior as a NumPy array.
        If False, returns it as a PyTorch tensor. Defaults to True.

    Returns
    -------
    np.ndarray
        Log-posterior values for the given parameters.
    """

    theta = check_tensor(theta, device)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)

    log_likelihood = log_likelihood_fn(theta)
    log_likelihood = torch.where(
        torch.isnan(log_likelihood), -torch.inf, log_likelihood)

    log_prob = log_prior(theta, priors) + log_likelihood
    if log_prob.ndim == 1:
        log_prob = log_prob.squeeze(0)
    if numpy_output:
        return log_prob.detach().cpu().numpy()
    return log_prob


def initialize_mcmc_sampler(
    surrogate: object,
    specs: object,
    QoI: list[str],
    y_true: np.ndarray | torch.Tensor,
    sl_true: dict[slice],
    n_walkers: int = None,
    priors_disttype: str = 'normal',
    fn_backend: str = 'backend.h5',
    restart: bool = True,
    device: str = 'cuda:0'
) -> tuple[np.ndarray, list, emcee.EnsembleSampler]:
    """
    Initialize the MCMC sampler for parameter optimization.

    Parameters
    ----------
    surrogate : object
        Surrogate model used for predictions.
        Must have a `predict` method.
    specs : object
        Object that contains specifications of the system.
    QoI : list[str]
        List of quantities of interest (QoI) to be optimized.
    observations : list[int]
        List of number of observations used in the likelihood function.
    y_true : np.ndarray | torch.Tensor
        True target values for the model.
    sl_true : dict[slice]
        Dictionary mapping quantities of interest (QoI) to slices of `y_true`.
    n_walkers : int, optional
        Number of walkers in the MCMC sampler.
        If None, defaults to 5 times the number of parameters.
    priors_disttype : str, optional
        Type of distribution for the priors, either 'normal' or 'uniform'.
        Defaults to 'normal'.
    fn_backend : str, optional
        Path to the backend file for storing samples.
        Defaults to 'backend.h5'.
    restart : bool, optional
        Whether to restart the sampler from the last saved state.
        Defaults to True.
    device : str, optional
        Device on which to perform computations (e.g., 'cuda:0' or 'cpu').
        Defaults to 'cuda:0'.

    Returns
    -------
    tuple[np.ndarray, list, emcee.EnsembleSampler]
        Initial parameter values (p0), list of priors, and the initialized sampler.
    """

    # Define priors based on the implicit parameter bounds
    priors = define_param_priors(specs.bounds_implicit, QoI, priors_disttype)

    # Initialize backend
    n_dim = len(priors)
    n_walkers = 5 * n_dim if n_walkers is None else n_walkers
    backend = initialize_backend(fn_backend)
    if restart:
        try:
            p0 = backend.get_last_sample()
        except AttributeError:
            p0 = initialize_walkers(priors, n_walkers, specs)
    else:
        backend.reset(n_walkers, n_dim)
        p0 = initialize_walkers(priors, n_walkers, specs)

    y_true = check_tensor(y_true, device)

    log_likelihood = partial(
        gaussian_log_likelihood,
        y_true=y_true,
        sl_true=sl_true,
        surrogate=surrogate,
        specs=specs
    )

    log_probability = partial(
        log_posterior,
        priors=list(priors.values()),
        log_likelihood_fn=log_likelihood,
        device=device
    )

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_probability,
        backend=backend, vectorize=True)

    return p0, priors, sampler


def lgp_hyperopt(
    X: torch.Tensor,
    y: torch.Tensor,
    y_mean: torch.Tensor,
    fn_hyperparams: str | Path,
    test_fraction: float,
    n_hyper: int,
    committee: int,
    observations: int,
    device: str,
    logger: callable,
    opt_kwargs: dict
) -> tuple[list[LocalGaussianProcess], tuple]:
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
    fn_hyperparams : str or Path
        Path to file to load/save optimized hyperparameters.
    test_fraction : float
        Fraction of the data to reserve for testing.
    n_hyper : int
        Maximum number of data points used for hyperparameter optimization.
    committee : int
        Number of LGP models in the ensemble.
    observations : int
        Number of observations.
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

    # Check if the hyperparameters file exists
    fn_hyperparams = Path(fn_hyperparams) if fn_hyperparams else None
    reuse_hyper = fn_hyperparams and fn_hyperparams.exists()

    if reuse_hyper:
        # Read hyperparameters from the file
        hyperparams = load_yaml(fn_hyperparams)
        lengths = hyperparams['lengths']
        widths = hyperparams['widths']
        sigmas = hyperparams['sigmas']

        # Ensure that the hyperparameters from the file match the committee size
        if (
            len(lengths) != committee or
            len(widths) != committee or
            len(sigmas) != committee
        ):
            raise ValueError(
                f"Number of hyperparameters in {fn_hyperparams}"
                f" does not match the ensemble size {committee}."
            )

    else:
        # Optimization of the hyperparameters
        n_hyper = min(n_hyper, len(X_train))

        X_hyper = check_tensor(X_train[:n_hyper], device='cpu')
        y_hyper = check_tensor(y_train[:n_hyper], device='cpu')

        priors = define_hyper_priors(X.shape[1])
        p0 = initialize_walkers(priors, 1).squeeze(0)

        log_likelihood = partial(
            loo_log_likelihood,
            X=X_hyper,
            y=y_hyper-y_mean)

        log_probability = partial(
            log_posterior,
            priors=list(priors.values()),
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
    logger.info(f'  > LGP committee: {0}/{committee}', overwrite=True)
    lgps = []
    for i, (l, w, s) in enumerate(zip(lengths, widths, sigmas), start=1):
        lgps.append(LocalGaussianProcess(X_train, y_train, y_mean, l, w, s, device))
        logger.info(f'  > LGP committee: {i}/{committee}', overwrite=True)

    lgp_committee = LGPCommittee(lgps, observations)

    # Validate the surrogate
    lgp_committee.validate(X_test, y_test)

    logger.info(
        f'  > LGP committee: {committee} (100%) | MAPE = {lgp_committee.error:.2f}%'
    )

    # Save hyperparameters if a file is specified
    if fn_hyperparams and not reuse_hyper:
        committee_data = {
            'lengths': np.array([lgp.hyperparameters['lengths'] for lgp in lgps]),
            'widths':  np.array([lgp.hyperparameters['width'] for lgp in lgps]),
            'sigmas': np.array([lgp.hyperparameters['sigma'] for lgp in lgps]),
        }
        save_yaml(committee_data, fn_hyperparams)

    return lgp_committee
