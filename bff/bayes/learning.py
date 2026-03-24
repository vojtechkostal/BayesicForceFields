from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from ..io.logs import Logger, print_progress_mcmc
from ..mcmc.proposal import AdaptiveGaussianProposal
from ..mcmc.sampler import Sampler
from ..qoi.data import QoIDataset
from ..tools import rdf_sigmoid_mean
from .gaussian_process import LGPCommittee, LocalGaussianProcess
from .likelihoods import gaussian_log_likelihood, loo_log_likelihood
from .posterior import log_posterior
from .priors import Prior, Priors
from .results import InferenceResults
from .utils import (
    check_device,
    check_tensor,
    find_map,
    initialize_walkers,
    laplace_approximation,
    train_test_split,
)


PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True, slots=True)
class InferenceProblem:
    """Complete Bayesian learning problem for force-field parameter inference."""

    models: dict[str, LGPCommittee]
    observations: dict[str, np.ndarray]
    constraint: Optional[Callable] = None

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("InferenceProblem requires at least one surrogate model.")

        model_names = set(self.models)
        observation_names = set(self.observations)
        missing = model_names - observation_names
        extra = observation_names - model_names
        if missing:
            raise ValueError(
                "Missing reference observations for QoI(s): "
                + ", ".join(sorted(missing))
            )
        if extra:
            raise ValueError(
                "Unused reference observations provided for QoI(s): "
                + ", ".join(sorted(extra))
            )

        n_params = {model.n_params for model in self.models.values()}
        if len(n_params) != 1:
            raise ValueError("All surrogate models must have the same input dimension.")

    @property
    def qoi_names(self) -> list[str]:
        return list(self.models)

    @property
    def n_params(self) -> int:
        return next(iter(self.models.values())).n_params

    @property
    def parameter_bounds(self) -> np.ndarray:
        if self.constraint is None:
            return np.tile([-1e5, 1e5], (self.n_params, 1))
        return np.asarray(self.constraint.explicit_bounds, dtype=float)

    @property
    def parameter_names(self) -> list[str] | None:
        if self.constraint is None:
            return None
        if not hasattr(self.constraint, "bounds") or not hasattr(
            self.constraint, "implicit_param"
        ):
            return None
        bounds = self.constraint.bounds.without(self.constraint.implicit_param)
        return bounds.names.tolist()

    @property
    def nuisance_names(self) -> list[str]:
        return [
            f"log_sigma_{qoi}"
            for qoi, model in self.models.items()
            if model.nuisance is None
        ]

    @property
    def n_free_nuisance(self) -> int:
        return len(self.nuisance_names)

    @classmethod
    def from_datasets(
        cls,
        models: Mapping[str, LGPCommittee],
        datasets: Sequence[QoIDataset],
        *,
        qoi: Optional[Sequence[str]] = None,
        constraint: Optional[Callable] = None,
    ) -> "InferenceProblem":
        qoi_selected = list(qoi) if qoi else list(models)
        missing = set(qoi_selected) - set(models)
        if missing:
            raise ValueError(
                "Requested QoI(s) not found in trained models: "
                + ", ".join(repr(name) for name in sorted(missing))
            )

        observations = {
            dataset.name: dataset.outputs_ref
            for dataset in datasets
            if dataset.name in qoi_selected
        }
        selected_models = {name: models[name] for name in qoi_selected}
        return cls(
            models=selected_models,
            observations=observations,
            constraint=constraint,
        )

    def build_priors(self, dist_type: str = "normal") -> Priors:
        return Priors.from_bounds(
            bounds=self.parameter_bounds,
            dist_type=dist_type,
            n_nuisance=self.n_free_nuisance,
            names=self.parameter_names,
            nuisance_names=self.nuisance_names,
        )

    def to_torch(
        self,
        device: str,
        dtype: torch.dtype = torch.float32,
    ) -> "InferenceProblem":
        return InferenceProblem(
            models=self.models,
            observations={
                qoi: torch.as_tensor(values, device=device, dtype=dtype)
                for qoi, values in self.observations.items()
            },
            constraint=self.constraint,
        )

    def infer(
        self,
        *,
        priors_disttype: str = "normal",
        total_steps: int = 1500,
        warmup: int = 500,
        thin: int = 1,
        progress_stride: int = 100,
        n_walkers: Optional[int] = None,
        fn_posterior: PathLike = "./posterior.pt",
        fn_checkpoint: Optional[PathLike] = "./mcmc-checkpoint.pt",
        fn_priors: Optional[PathLike] = "./priors.pt",
        restart: bool = True,
        device: str = "cuda",
        logger: Optional[Logger] = None,
        rhat_tol: float = 1.01,
        ess_min: int = 100,
    ) -> InferenceResults:
        """Run posterior sampling for this inference problem."""
        logger = logger or Logger("BFFLearn")
        logger.info("=== Posterior Inference ===", level=0)
        logger.info("", level=0)

        priors = self.build_priors(dist_type=priors_disttype)
        n_walkers = 5 * len(priors) if n_walkers is None else n_walkers
        initial_positions = initialize_walkers(
            priors.distributions,
            n_walkers,
            self.constraint,
        )
        proposal_cov = check_tensor(
            torch.diag(torch.tensor(priors.scales, dtype=torch.float32) ** 2),
            device=device,
        )
        proposal = AdaptiveGaussianProposal(proposal_cov, device=device)
        log_likelihood = partial(
            gaussian_log_likelihood,
            problem=self.to_torch(device),
        )
        log_probability = partial(
            log_posterior,
            priors=priors,
            log_likelihood_fn=log_likelihood,
            device=device,
        )
        sampler = Sampler(
            log_prob=log_probability,
            proposal=proposal,
            device=device,
            dtype=torch.float32,
        )

        fn_posterior = Path(fn_posterior).resolve()
        fn_checkpoint = (
            None if fn_checkpoint is None else Path(fn_checkpoint).resolve()
        )
        fn_priors = None if fn_priors is None else Path(fn_priors).resolve()

        if fn_checkpoint is None:
            fn_checkpoint = _default_checkpoint_path(fn_posterior)

        if fn_priors is not None:
            priors.write(fn_priors)

        print_progress_mcmc(
            sampler,
            initial_positions,
            total_steps=total_steps,
            warmup=warmup,
            thin=thin,
            progress_stride=progress_stride,
            logger=logger,
            restart=restart,
            fn_checkpoint=fn_checkpoint,
            rhat_tol=rhat_tol,
            ess_min=ess_min,
        )

        sampler.write_posterior(fn_posterior)
        return InferenceResults.load(posterior=fn_posterior, priors=priors)


def _resolve_mean(
    dataset: QoIDataset,
    mean: ArrayLike | float | str,
) -> ArrayLike | float:
    """Resolve a configured surrogate mean specification."""
    if dataset.name == "rdf" and mean == "sigmoid":
        n_bins = dataset.settings_kwargs.get("n_bins")
        r_range = dataset.settings_kwargs.get("r_range")
        return rdf_sigmoid_mean(n_bins, r_range, dataset.outputs_ref)
    return mean


def _default_checkpoint_path(fn_posterior: Path) -> Path:
    suffix = "".join(fn_posterior.suffixes) or ".pt"
    stem = fn_posterior.name[: -len(suffix)] if suffix else fn_posterior.name
    return fn_posterior.with_name(f"{stem}.ckpt{suffix}")


def fit_lgp_committee(
    X: torch.Tensor,
    y: torch.Tensor,
    y_mean: torch.Tensor,
    test_fraction: float,
    n_hyper: int,
    committee: int,
    observations: int,
    nuisance: float | None,
    fn_out: PathLike | None,
    device: str,
    logger: Optional[Logger] = None,
    opt_kwargs: Optional[dict[str, Union[int, float, str]]] = None,
) -> LGPCommittee:
    """Fit a committee of local Gaussian-process surrogates."""
    check_device(device)
    logger = logger or Logger("LGP")
    opt_kwargs = dict(opt_kwargs or {})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_fraction)
    n_hyper = min(n_hyper, len(X_train))

    X_hyper = check_tensor(X_train[:n_hyper], device="cpu")
    y_hyper = check_tensor(y_train[:n_hyper], device="cpu")
    y_mean = check_tensor(y_mean, device="cpu")

    n_params = X.shape[1]
    priors = Priors(
        [Prior("normal", -2.0, 2.0, name=f"length_{i}") for i in range(n_params)]
        + [
            Prior("normal", -2.0, 2.0, name="width"),
            Prior("normal", -2.0, 3.0, name="noise"),
        ]
    )
    p0 = torch.tensor(priors.means, dtype=torch.float32)

    log_likelihood = partial(loo_log_likelihood, X=X_hyper, y=y_hyper - y_mean)
    log_probability = partial(
        log_posterior,
        priors=priors,
        log_likelihood_fn=log_likelihood,
        device="cpu",
        numpy_output=False,
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

    logger.info(f"Committee: 0/{committee}", level=2, overwrite=True)
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
        logger.info(f"Committee: {i}/{committee}", level=2, overwrite=True)

    lgp_committee = LGPCommittee(lgps, observations, nuisance)
    lgp_committee.validate(X_test, y_test)
    logger.info(
        f"Committee: {committee} (100%) | MAPE = {lgp_committee.error:.2f}%",
        level=2,
    )

    if fn_out is not None:
        lgp_committee.write(fn_out)

    return lgp_committee


def _effective_observations(
    dataset: QoIDataset,
    observation_scale: float = 1.0,
) -> int:
    """Return the effective observation count used in the likelihood term."""
    return max(1, int(round(dataset.n_observations * float(observation_scale))))


def train_surrogates(
    datasets: Sequence[QoIDataset],
    *,
    y_means: Optional[Mapping[str, ArrayLike | float | str]] = None,
    observation_scales: Optional[Mapping[str, float]] = None,
    model_paths: Optional[Mapping[str, PathLike | None]] = None,
    reuse_models: bool = True,
    n_hyper_max: int = 200,
    committee_size: int = 1,
    test_fraction: float = 0.2,
    device: str = "cuda",
    logger: Optional[Logger] = None,
    **opt_kwargs,
) -> dict[str, LGPCommittee]:
    """Train or load QoI surrogate models."""
    logger = logger or Logger("BFFLearn")
    y_means = dict(y_means or {})
    observation_scales = dict(observation_scales or {})
    model_paths = dict(model_paths or {})

    logger.info("=== Optimizing LGP surrogates ===", level=0)
    logger.info("", level=0)

    models: dict[str, LGPCommittee] = {}
    for dataset in datasets:
        if not isinstance(dataset, QoIDataset):
            raise TypeError(
                f"Invalid dataset type: {type(dataset)}. Expected QoIDataset."
            )

        qoi = dataset.name
        observations = _effective_observations(
            dataset,
            observation_scales.get(qoi, 1.0),
        )
        logger.info(f"QoI: {qoi}", level=1)

        fn_model_raw = model_paths.get(qoi)
        fn_model = None if fn_model_raw is None else Path(fn_model_raw).resolve()
        if fn_model is not None:
            fn_model.parent.mkdir(parents=True, exist_ok=True)

        if reuse_models and fn_model is not None and fn_model.exists():
            models[qoi] = LGPCommittee.load(fn_model)
            models[qoi].observations = observations
            logger.info(
                f"Using cached model. | obs = {observations} "
                f"| MAPE = {models[qoi].error:.2f}",
                level=2,
            )
            logger.info("", level=0)
            continue

        models[qoi] = fit_lgp_committee(
            X=dataset.inputs,
            y=dataset.outputs,
            y_mean=_resolve_mean(dataset, y_means.get(qoi, 0)),
            test_fraction=test_fraction,
            n_hyper=n_hyper_max,
            committee=committee_size,
            observations=observations,
            nuisance=dataset.nuisance,
            fn_out=fn_model,
            device=device,
            logger=logger,
            opt_kwargs=opt_kwargs,
        )
        logger.info(f"Effective observations: {observations}", level=2)
        logger.info("", level=0)

    return models
