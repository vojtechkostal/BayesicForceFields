"""Fit local Gaussian-process surrogate committees to QoI datasets.

Hyperparameters are found by MAP optimization of a leave-one-out likelihood;
committees draw further hyperparameter sets from a Laplace approximation.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.optimize import minimize
from torch.autograd.functional import hessian

from ..io.logs import Logger
from ..qoi.dataset import QoIDataset
from .gaussian_process import LGPCommittee, LocalGaussianProcess
from .likelihoods import loo_log_likelihood
from .means import (
    MeanSpec,
    build_mean,
    describe_spec,
    evaluate_mean,
    resolve_spec,
)
from .posterior import log_posterior
from .priors import Prior, Priors

PathLike = Union[str, Path]

_BAD_VALUE = 1e10  # objective at points where the log posterior is not finite
# Log hyperparameters are searched within this many prior standard deviations.
_SEARCH_WIDTH = 6.0
_N_RESTARTS = 2


def train_test_split(
    X: torch.Tensor, y: torch.Tensor, test_fraction: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly split the dataset into training and testing sets."""
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)
    n = len(X)
    if n != len(y):
        raise ValueError("X and y must have the same length.")
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must be between 0 and 1.")
    if n < 2:
        raise ValueError("X and y must have at least 2 samples.")

    indices = torch.randperm(n)
    test_size = min(max(int(n * test_fraction), 1), n - 1)
    idx_train = indices[test_size:]
    idx_test = indices[:test_size]
    return X[idx_train], X[idx_test], y[idx_train], y[idx_test]


@dataclass(frozen=True)
class MapResult:
    """Outcome of :func:`find_map`: the maximizer and what it took."""

    theta: torch.Tensor
    value: float  # log posterior at ``theta``
    iterations: int
    evaluations: int  # objective and gradient evaluations
    converged: bool


def find_map(
    fn: Callable,
    x0: torch.Tensor,
    bounds: Tuple[np.ndarray, np.ndarray],
    max_iter: int = 500,
    tol_grad: float = 1e-4,
    starts: Optional[Sequence[torch.Tensor]] = None,
    logger: Optional[Logger] = None,
) -> MapResult:
    """Maximize a log posterior with L-BFGS-B.

    The gradient comes from autograd; ``fn`` maps a parameter vector to a
    scalar. The search starts at ``x0`` and, if that does not converge to an
    interior point, at each of ``starts``; the best result is kept. Points
    where ``fn`` is not finite (for example a covariance that is not positive
    definite) are treated as very bad, so the line search backs off.

    Parameters
    ----------
    fn : Callable
        Log posterior of one parameter vector, as a scalar tensor.
    x0 : torch.Tensor
        First starting point.
    bounds : tuple of arrays
        Lower and upper limits of every parameter.
    max_iter : int
        Maximum iterations of each search.
    tol_grad : float
        Convergence: largest component of the (projected) gradient.
    starts : sequence of torch.Tensor, optional
        Further starting points, used only if needed.
    """
    lower, upper = (np.asarray(b, dtype=float) for b in bounds)
    scipy_bounds = list(zip(lower, upper))
    evaluations = 0

    def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        theta = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        value = fn(theta)
        if not torch.isfinite(value):
            return _BAD_VALUE, np.zeros_like(x)
        (-value).backward()
        return -value.item(), theta.grad.numpy().copy()

    def report(x: np.ndarray) -> None:
        if logger is not None and n_iter[0] % 10 == 0:
            logger.status(
                "MAP search",
                f"it. {n_iter[0]}/{max_iter} | evaluations: {evaluations}",
                level=2,
                overwrite=True,
            )
        n_iter[0] += 1

    best: Optional[MapResult] = None
    total_iterations = 0
    for start in [x0, *(starts or [])]:
        n_iter = [0]
        result = minimize(
            objective,
            start.detach().cpu().numpy().astype(float),
            jac=True,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            callback=report,
            options={
                "maxiter": max_iter,
                "maxfun": 4 * max_iter,
                "gtol": tol_grad,
                "ftol": 1e-13,
            },
        )
        total_iterations += int(result.nit)
        interior = np.all(
            (result.x > lower + 1e-6) & (result.x < upper - 1e-6)
        )
        candidate = MapResult(
            theta=torch.tensor(result.x, dtype=torch.float64),
            value=-float(result.fun),
            iterations=total_iterations,
            evaluations=evaluations,
            converged=bool(result.success and interior),
        )
        if best is None or (
            (candidate.converged, candidate.value) > (best.converged, best.value)
        ):
            best = candidate
        if best.converged and start is x0:
            break
    best = MapResult(
        best.theta, best.value, total_iterations, evaluations, best.converged
    )
    if logger is not None:
        detail = (
            f"{best.iterations} iterations | {best.evaluations} evaluations | "
            f"log posterior {best.value:.4f}"
        )
        if best.converged:
            logger.done("MAP search", detail=detail, level=2)
        else:
            logger.warn(f"MAP search did not converge | {detail}", level=2)
    return best


def laplace_approximation(
    fn: Callable,
    map_theta: torch.Tensor,
    min_curvature: float = 1e-3,
) -> torch.Tensor:
    """Covariance of the Laplace approximation around the MAP estimate.

    It is the inverse of the negative Hessian of ``fn``. Curvatures below
    ``min_curvature`` (flat or non-maximal directions) are raised to it, so
    the covariance stays finite and positive definite.

    Parameters
    ----------
    fn : Callable
        Log posterior of one parameter vector, as a scalar tensor.
    map_theta : torch.Tensor
        MAP estimate around which to approximate.
    min_curvature : float
        Smallest curvature (inverse variance) allowed.
    """
    H = -hessian(lambda theta: fn(theta).sum(), map_theta)
    H = 0.5 * (H + H.T)
    curvature, directions = torch.linalg.eigh(H)
    curvature = curvature.clamp(min=min_curvature)
    return (directions / curvature) @ directions.T


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

    # ``noise_variance`` is a variance, so its natural scale is a fraction of
    # the residual variance.
    noise_scale = 0.1 * target_scale.square()
    tiny = torch.finfo(noise_scale.dtype).tiny
    noise_scale = torch.clamp(noise_scale, min=tiny)

    return Priors(
        [
            Prior("normal", float(torch.log(scale)), 2.0, name=f"lengthscale_{i}")
            for i, scale in enumerate(input_scales)
        ]
        + [
            Prior("normal", float(torch.log(target_scale)), 2.0, name="amplitude"),
            Prior(
                "normal", float(torch.log(noise_scale)), 3.0, name="noise_variance"
            ),
        ]
    )


def fit_lgp_committee(
    X: torch.Tensor,
    y: torch.Tensor,
    mean: MeanSpec,
    test_fraction: float,
    n_hyper: int,
    committee: int,
    y_ref: np.ndarray,
    n_curves: int,
    nuisance: float | None,
    fn_out: PathLike | None,
    logger: Optional[Logger] = None,
    opt_kwargs: Optional[dict[str, Union[int, float]]] = None,
    hyperpriors: Optional[Priors | Sequence[Prior]] = None,
    dataset_fingerprint: str | None = None,
    parameter_names: Sequence[str] | None = None,
) -> LGPCommittee:
    """Fit a committee of local Gaussian-process surrogates.

    The fit is small (a few hundred points) and runs on the CPU in float64.
    The returned committee lives there; ``bff learn`` moves it to its device.
    """
    logger = logger or Logger("fit-lgp")
    opt_kwargs = dict(opt_kwargs or {})

    X = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction)
    n_hyper = min(n_hyper, len(X_train))
    # The mean comes from the training split only, never the test samples.
    mean = build_mean(mean, X_train, y_train)

    X_hyper = X_train[:n_hyper]
    y_hyper = y_train[:n_hyper]
    y_hyper_mean = evaluate_mean(mean, X_hyper)

    n_params = X.shape[1]
    if hyperpriors is None:
        priors = _default_lgp_hyperpriors(X_hyper, y_hyper - y_hyper_mean)
    else:
        priors = Priors(list(hyperpriors))
    if len(priors) != n_params + 2:
        raise ValueError(
            "LGP hyperpriors must define one length scale per input, followed "
            "by the amplitude and the noise variance."
        )
    centers = torch.tensor(priors.means, dtype=torch.float64)
    widths = _SEARCH_WIDTH * torch.tensor(priors.scales, dtype=torch.float64)

    log_probability = partial(
        log_posterior,
        priors=priors,
        log_likelihood_fn=partial(
            loo_log_likelihood, X=X_hyper, y=y_hyper - y_hyper_mean
        ),
    )
    starts = [
        torch.stack([prior.distribution.sample() for prior in priors]).double()
        for _ in range(_N_RESTARTS)
    ]
    search = find_map(
        log_probability,
        centers,
        bounds=(centers - widths, centers + widths),
        starts=starts,
        logger=logger,
        **opt_kwargs,
    )
    map_theta = search.theta

    if committee > 1:
        cov = laplace_approximation(log_probability, map_theta)
        hyper_dist = torch.distributions.MultivariateNormal(map_theta, cov)
        hyper_samples = hyper_dist.sample((committee,))
    else:
        hyper_samples = map_theta.unsqueeze(0)

    hyper_samples = hyper_samples.exp()
    logger.status("Committee", f"0/{committee}", level=2, overwrite=True)
    members = []
    for i, sample in enumerate(hyper_samples, start=1):
        members.append(
            LocalGaussianProcess(
                X_train,
                y_train,
                mean,
                lengthscales=sample[:-2],
                amplitude=sample[-2],
                noise_variance=sample[-1],
            )
        )
        if i < committee:
            logger.status("Committee", f"{i}/{committee}", level=2, overwrite=True)

    lgp_committee = LGPCommittee(
        members=members,
        y_ref=y_ref,
        n_curves=n_curves,
        nuisance=nuisance,
        dataset_fingerprint=dataset_fingerprint,
        parameter_names=parameter_names,
    )
    lgp_committee.validate(X_test, y_test)
    logger.done(
        "Committee",
        detail=f"{committee} member(s) | test sMAPE "
        f"{lgp_committee.test_error:.2f}%",
        level=2,
    )

    if fn_out is not None:
        lgp_committee.write(fn_out)

    return lgp_committee


def fit_surrogates(
    datasets: Sequence[QoIDataset],
    *,
    means: Optional[Mapping[str, MeanSpec]] = None,
    hyperpriors: Optional[Mapping[str, Priors | Sequence[Prior]]] = None,
    model_paths: Optional[Mapping[str, PathLike | None]] = None,
    reuse_models: bool = True,
    n_hyper_max: int = 200,
    committee_size: int = 1,
    test_fraction: float = 0.2,
    logger: Optional[Logger] = None,
    **opt_kwargs,
) -> dict[str, LGPCommittee]:
    """Fit one committee per QoI dataset, or reuse a saved one.

    ``means`` maps QoI names to mean specifications (default ``"data"``;
    see :mod:`bff.bayes.means`). A saved model is reused only if it was
    fitted to the same dataset with the same mean.
    """
    owns_logger = logger is None
    logger = logger or Logger("fit-lgp")
    means = dict(means or {})
    hyperpriors = dict(hyperpriors or {})
    model_paths = dict(model_paths or {})

    if owns_logger:
        logger.section("Fit LGP")
        logger.blank()

    models: dict[str, LGPCommittee] = {}
    for dataset in datasets:
        if not isinstance(dataset, QoIDataset):
            raise TypeError(
                f"Invalid dataset type: {type(dataset)}. Expected QoIDataset."
            )

        qoi = dataset.name
        spec = means.get(qoi, "data")
        logger.info(f"{qoi}: mean {describe_spec(spec) or 'custom'}", level=1)

        fn_model = None if model_paths.get(qoi) is None else Path(model_paths[qoi])
        if fn_model is not None:
            fn_model = fn_model.resolve()
            fn_model.parent.mkdir(parents=True, exist_ok=True)

        if reuse_models and fn_model is not None and fn_model.exists():
            cached = LGPCommittee.load(fn_model)
            if (
                cached.dataset_fingerprint == dataset.fingerprint()
                and describe_spec(spec) is not None
                and cached.mean_spec == describe_spec(spec)
            ):
                models[qoi] = cached
                logger.done(
                    "Reused",
                    detail=f"{fn_model} | test sMAPE {cached.test_error:.2f}%",
                    level=2,
                )
                logger.blank()
                continue
            logger.info(
                f"{fn_model} was fitted to other data or another mean; refitting.",
                level=2,
            )

        models[qoi] = fit_lgp_committee(
            X=dataset.X,
            y=dataset.y,
            mean=resolve_spec(spec, dataset),
            test_fraction=test_fraction,
            n_hyper=n_hyper_max,
            committee=committee_size,
            y_ref=dataset.y_ref,
            n_curves=dataset.n_curves,
            nuisance=dataset.nuisance,
            fn_out=None,
            logger=logger,
            opt_kwargs=opt_kwargs,
            hyperpriors=hyperpriors.get(qoi),
            dataset_fingerprint=dataset.fingerprint(),
            parameter_names=dataset.parameter_names,
        )
        models[qoi].mean_spec = describe_spec(spec)
        if fn_model is not None:
            models[qoi].write(fn_model)
        logger.blank()

    return models
