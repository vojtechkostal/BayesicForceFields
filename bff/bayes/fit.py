"""Fit local Gaussian-process surrogate committees to QoI datasets.

Hyperparameters are found by MAP optimization of a leave-one-out likelihood;
committees draw further hyperparameter sets from a Laplace approximation.
"""

from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.autograd.functional import hessian

from ..io.logs import Logger
from ..qoi.dataset import QoIDataset
from .gaussian_process import (
    LGPCommittee,
    LocalGaussianProcess,
    MeanFunction,
    evaluate_mean,
)
from .likelihoods import loo_log_likelihood
from .means import rdf_sigmoid_mean
from .posterior import log_posterior
from .priors import Prior, Priors
from .utils import check_device, check_tensor, enable_manual_dist

PathLike = Union[str, Path]


def train_test_split(
    X: torch.Tensor, y: torch.Tensor, test_fraction: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the dataset into training and testing sets."""

    device = X.device if isinstance(X, torch.Tensor) else "cpu"
    X = check_tensor(X, device=device)
    y = check_tensor(y, device=device)
    n = len(X)
    if n != len(y):
        raise ValueError("X and y must have the same length.")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1.")
    if n < 2:
        raise ValueError("X and y must have at least 2 samples.")

    indices = torch.randperm(n, device=X.device)
    test_size = min(max(int(n * test_fraction), 1), n - 1)
    idx_train = indices[test_size:]
    idx_test = indices[:test_size]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]


def find_max_stable_lr(
    fn: Callable,
    p0: torch.Tensor,
    learning_rates: (
        Sequence[Union[float, torch.Tensor]] | float | torch.Tensor | None
    ) = None,
    max_iter: int = 100,
    param_bounds: Tuple[float, float] = (-7, 7),
) -> Union[float, None]:
    """Find the largest stable learning rate for gradient-based optimization.

    Parameters
    ----------
    fn : Callable
        Objective function returning a scalar tensor.
    p0 : torch.Tensor
        Initial parameter vector.
    learning_rates : iterable of float, optional
        Learning rates to test. Defaults to log-spaced values.
    max_iter : int
        Number of steps to test for each learning rate.
    param_bounds : tuple of float
        Bounds beyond which parameters are considered unstable.

    Returns
    -------
    float or None
        The largest stable learning rate found, or None if none were stable.
    """
    if learning_rates is None:
        learning_rates = 10 ** torch.linspace(-1, -6, 6)
    elif isinstance(learning_rates, torch.Tensor) and learning_rates.ndim == 0:
        learning_rates = [float(learning_rates.item())]
    elif isinstance(learning_rates, (int, float)):
        learning_rates = [float(learning_rates)]
    else:
        learning_rates = list(learning_rates)
    lower, upper = param_bounds
    # Log-scale hyperparameters can legitimately start outside the generic
    # stability window when the underlying data are very small or very large.
    # Preserve the explosion guard while always admitting the supplied start.
    lower = min(lower, float(p0.min()) - 1.0)
    upper = max(upper, float(p0.max()) + 1.0)
    for lr in learning_rates:
        x = p0.clone().detach().requires_grad_(True)
        opt = torch.optim.SGD([x], lr=lr)

        for i in range(max_iter):
            diverged = not torch.isfinite(x).all()
            if diverged or torch.any(x < lower) or torch.any(x > upper):
                break
            opt.zero_grad()
            loss = -fn(x)
            loss.backward()
            opt.step()
        else:
            # Only gets executed if inner loop did not break (i.e., stable)
            return lr


def find_map(
    fn: Callable,
    x0: torch.Tensor,
    lr: Union[float, torch.Tensor] = None,
    max_iter: int = 10000,
    tol_grad: float = 1e-2,
    device: str = 'cpu',
    logger: Callable = None
) -> torch.Tensor:

    """Find the maximum a posteriori (MAP) estimate
    using gradient-based optimization.

    Parameters
    ----------
    fn : Callable
        Objective function to maximize (log-posterior).
    x0 : torch.Tensor
        Initial parameter vector.
    lr : float or torch.Tensor, optional
        Learning rate for optimization. If None, it will be determined by a search.
    max_iter : int
        Maximum number of optimization iterations.
    tol_grad : float
        Gradient norm threshold for convergence.
    device : str
        Device to perform computations on.
    logger : Callable, optional
        Logger for progress updates. Should accept a string and a level argument.

    Returns
    -------
    torch.Tensor
        The MAP estimate found by optimization.
    """

    if logger is not None:
        logger.status(
            "Learning-rate search",
            "in progress...",
            level=2,
            overwrite=True,
        )

    lr_opt = find_max_stable_lr(fn, x0, learning_rates=lr)
    if lr_opt is not None:
        lr_opt = 0.5 * lr_opt
        if logger is not None:
            logger.done(
                "Learning-rate search",
                detail=f"lr = {lr_opt:.1e}",
                level=2,
            )
    else:
        raise ValueError("No stable learning rate found.")

    x0 = x0.clone().detach().to(device).requires_grad_(True)
    optimizer = torch.optim.SGD([x0], lr=lr_opt)
    iter_width = len(str(max_iter))
    tol_grad_text = f"{tol_grad:g}"
    best_value = -torch.inf
    best_x = x0.detach().clone()

    for i in range(max_iter):
        optimizer.zero_grad()
        loss = -fn(x0)
        value = -loss.detach()
        if torch.isfinite(value) and value > best_value:
            best_value = value
            best_x = x0.detach().clone()
        loss.backward()
        grad_norm = x0.grad.norm().item()
        if i % 100 == 0 and logger is not None:
            logger.status(
                "MAP search",
                (
                    f"it. {i:>{iter_width}d}/{max_iter:<{iter_width}d} | "
                    f"loss: {loss.item():>10.3f} | "
                    f"grad: {grad_norm:>8.3f}/{tol_grad_text}"
                ),
                level=2,
                overwrite=True,
            )
        optimizer.step()
        if grad_norm < tol_grad:
            if logger is not None:
                logger.done("MAP search", level=2)
            break

    else:
        if logger is not None:
            logger.warn(
                "MAP search reached the maximum iterations without convergence.",
                level=2,
            )

    return best_x


def laplace_approximation(
    fn: Callable,
    map_theta: torch.Tensor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Perform Laplace approximation around MAP estimate.

    Parameters
    ----------
    fn : Callable
        Objective function to approximate (log-posterior).
    map_theta : torch.Tensor
        MAP estimate around which to perform the approximation.
    device : str
        Device to perform computations on.

    Returns
    -------
    cov : torch.Tensor
        The approximate posterior covariance (inverse Hessian).
    """

    map_theta = check_tensor(map_theta, device=device)
    with enable_manual_dist():
        H = -hessian(lambda th: fn(th).sum(), map_theta)

    reg_eye = 1e-6 * torch.eye(H.shape[0], device=H.device)
    cov = torch.linalg.inv(H + reg_eye)

    # symmerize covariance matrix
    cov = (cov + cov.T) / 2

    return cov


def _resolve_mean(
    dataset: QoIDataset,
    mean: MeanFunction | str,
) -> MeanFunction:
    """Resolve a configured surrogate mean specification."""
    if isinstance(mean, str):
        if mean == "sigmoid":
            bins = dataset.settings.get("bins")
            distance_range = dataset.settings.get("range")
            if bins is None or distance_range is None:
                raise ValueError(
                    "RDF sigmoid mean requires shared RDF settings in the dataset. "
                    "Build the dataset with one consistent RDF routine definition "
                    "per QoI."
                )
            if bins != dataset.curve_length:
                raise ValueError(
                    "RDF dataset settings declare "
                    f"bins={bins!r}, but each RDF curve contains "
                    f"{dataset.curve_length} values."
                )
            return rdf_sigmoid_mean(bins, distance_range, dataset.outputs_ref)
        else:
            raise NotImplementedError(
                "Other than 'sigmoid' or single-value mean is not implemented")
    return mean


def _default_lgp_hyperpriors(
    X: torch.Tensor,
    residuals: torch.Tensor,
) -> Priors:
    """Build scale-aware priors for log GP hyperparameters."""
    input_scales = X.std(dim=0, unbiased=False)
    input_scales = torch.where(
        torch.isfinite(input_scales) & (input_scales > 0),
        input_scales,
        torch.ones_like(input_scales),
    )

    target_scale = residuals.std(unbiased=False)
    if not torch.isfinite(target_scale) or target_scale <= 0:
        target_scale = torch.sqrt(torch.mean(residuals.square()))
    if not torch.isfinite(target_scale) or target_scale <= 0:
        target_scale = torch.ones((), dtype=residuals.dtype)

    # LocalGaussianProcess adds ``sigma`` directly to the covariance diagonal,
    # so its natural scale is a fraction of the target variance.
    noise_scale = 0.1 * target_scale.square()
    tiny = torch.finfo(noise_scale.dtype).tiny
    noise_scale = torch.clamp(noise_scale, min=tiny)

    return Priors(
        [
            Prior("normal", float(torch.log(scale)), 2.0, name=f"length_{i}")
            for i, scale in enumerate(input_scales)
        ]
        + [
            Prior("normal", float(torch.log(target_scale)), 2.0, name="width"),
            Prior("normal", float(torch.log(noise_scale)), 3.0, name="noise"),
        ]
    )


def fit_lgp_committee(
    X: torch.Tensor,
    y: torch.Tensor,
    y_mean: MeanFunction,
    test_fraction: float,
    n_hyper: int,
    committee: int,
    reference_values: np.ndarray,
    n_curves: int,
    nuisance: float | None,
    fn_out: PathLike | None,
    device: str,
    logger: Optional[Logger] = None,
    opt_kwargs: Optional[dict[str, Union[int, float, str]]] = None,
    hyperpriors: Optional[Priors | Sequence] = None,
    dataset_fingerprint: str | None = None,
) -> LGPCommittee:
    """Fit a committee of local Gaussian-process surrogates."""
    check_device(device)
    logger = logger or Logger("fit-lgp")
    opt_kwargs = dict(opt_kwargs or {})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction)
    n_hyper = min(n_hyper, len(X_train))

    X_hyper = check_tensor(X_train[:n_hyper], device="cpu")
    y_hyper = check_tensor(y_train[:n_hyper], device="cpu")
    y_hyper_mean = evaluate_mean(y_mean, X_hyper, device="cpu")

    n_params = X.shape[1]
    if hyperpriors is None:
        priors = _default_lgp_hyperpriors(X_hyper, y_hyper - y_hyper_mean)
    else:
        priors = Priors.from_any(hyperpriors)
    if len(priors) != n_params + 2:
        raise ValueError(
            "LGP hyperpriors must define one length scale per input parameter, "
            "followed by width and noise."
        )
    p0 = torch.tensor(priors.means, dtype=torch.float32)

    log_likelihood = partial(
        loo_log_likelihood,
        X=X_hyper,
        y=y_hyper - y_hyper_mean,
    )
    log_probability = partial(
        log_posterior,
        priors=priors,
        log_likelihood_fn=log_likelihood,
        device="cpu",
    )

    map_theta = find_map(log_probability, p0, logger=logger, **opt_kwargs)

    if committee > 1:
        cov = laplace_approximation(log_probability, map_theta, device="cpu")
        hyper_dist = torch.distributions.MultivariateNormal(map_theta, cov)
        hyper_samples = hyper_dist.sample((committee,))
    else:
        hyper_samples = map_theta.unsqueeze(0)

    hyper_samples = hyper_samples.exp()
    lengths = hyper_samples[:, :-2]
    widths = hyper_samples[:, -2]
    sigmas = hyper_samples[:, -1]

    logger.status("Committee", f"0/{committee}", level=2, overwrite=True)
    lgps = []
    for i, (length, width, sigma) in enumerate(
        zip(lengths, widths, sigmas),
        start=1,
    ):
        lgps.append(
            LocalGaussianProcess(
                X_train,
                y_train,
                y_mean,
                length,
                width,
                sigma,
                device,
            )
        )
        if i < committee:
            logger.status("Committee", f"{i}/{committee}", level=2, overwrite=True)

    lgp_committee = LGPCommittee(
        lgps=lgps,
        reference_values=reference_values,
        n_curves=n_curves,
        nuisance=nuisance,
        dataset_fingerprint=dataset_fingerprint,
    )
    lgp_committee.validate(X_test, y_test)
    logger.done(
        "Committee",
        detail=f"{committee}/{committee} (100%) | MAPE = {lgp_committee.error:.2f}%",
        level=2,
    )

    if fn_out is not None:
        lgp_committee.write(fn_out)

    return lgp_committee


def fit_surrogates(
    datasets: Sequence[QoIDataset],
    *,
    y_means: Optional[Mapping[str, MeanFunction | str]] = None,
    hyperpriors: Optional[Mapping[str, Priors | Sequence]] = None,
    model_paths: Optional[Mapping[str, PathLike | None]] = None,
    reuse_models: bool = True,
    n_hyper_max: int = 200,
    committee_size: int = 1,
    test_fraction: float = 0.2,
    device: str = "cuda",
    logger: Optional[Logger] = None,
    **opt_kwargs,
) -> dict[str, LGPCommittee]:
    """Fit or load QoI surrogate models."""
    owns_logger = logger is None
    logger = logger or Logger("fit-lgp")
    y_means = dict(y_means or {})
    hyperpriors = dict(hyperpriors or {})
    model_paths = dict(model_paths or {})

    if owns_logger:
        logger.section("Surrogate Fitting")
        logger.blank()

    models: dict[str, LGPCommittee] = {}
    for dataset in datasets:
        if not isinstance(dataset, QoIDataset):
            raise TypeError(
                f"Invalid dataset type: {type(dataset)}. Expected QoIDataset."
            )

        qoi = dataset.name
        logger.info(f"QoI {qoi}", level=1)

        fn_model_raw = model_paths.get(qoi)
        fn_model = None if fn_model_raw is None else Path(fn_model_raw).resolve()
        if fn_model is not None:
            fn_model.parent.mkdir(parents=True, exist_ok=True)

        if reuse_models and fn_model is not None and fn_model.exists():
            models[qoi] = LGPCommittee.load(fn_model)
            fingerprint = dataset.fingerprint()
            if models[qoi].dataset_fingerprint != fingerprint:
                raise ValueError(
                    f"Cached surrogate for {qoi!r} at {fn_model} was fitted "
                    "from different QoI data. Set fit.reuse_models: false "
                    "or remove the stale model."
                )
            models[qoi].reference_values = np.asarray(
                dataset.outputs_ref,
                dtype=float,
            ).reshape(-1)
            models[qoi].n_eff = float(models[qoi].reference_values.size)
            models[qoi].n_curves = dataset.n_curves
            if models[qoi].reference_values.size != models[qoi].y_size:
                raise ValueError(
                    f"Cached surrogate for {qoi!r} is incompatible with the "
                    "current reference observation size."
                )
            models[qoi].write(fn_model)
            logger.info(
                f"Using cached surrogate model. | MAPE = {models[qoi].error:.2f}",
                level=2,
            )
            logger.blank()
            continue

        models[qoi] = fit_lgp_committee(
            X=dataset.inputs,
            y=dataset.outputs,
            y_mean=_resolve_mean(dataset, y_means.get(qoi, 0)),
            test_fraction=test_fraction,
            n_hyper=n_hyper_max,
            committee=committee_size,
            reference_values=dataset.outputs_ref,
            n_curves=dataset.n_curves,
            nuisance=dataset.nuisance,
            fn_out=fn_model,
            device=device,
            logger=logger,
            opt_kwargs=opt_kwargs,
            hyperpriors=hyperpriors.get(qoi),
            dataset_fingerprint=dataset.fingerprint(),
        )
        logger.blank()

    return models
