from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Mapping, Optional, Union

import numpy as np
import torch

from ..io.logs import Logger, print_progress_mcmc
from ..io.utils import mapping_fingerprint
from ..mcmc.proposal import AdaptiveGaussianProposal
from ..mcmc.sampler import Sampler
from .gaussian_process import (
    LGPCommittee,
)
from .likelihoods import gaussian_log_likelihood
from .posterior import log_posterior
from .priors import Priors
from .results import PosteriorResults
from .utils import (
    check_tensor,
    initialize_walkers,
)

PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class LearningProblem:
    """Complete Bayesian learning problem for force-field parameters."""

    models: dict[str, LGPCommittee]
    constraint: Optional[Callable] = None
    observations: dict[str, np.ndarray | torch.Tensor] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("LearningProblem requires at least one surrogate model.")

        empty_models = [qoi for qoi, model in self.models.items() if not model.lgps]
        if empty_models:
            raise ValueError(
                "Surrogate models without committee members: "
                + ", ".join(sorted(empty_models))
            )

        object.__setattr__(
            self,
            "observations",
            {
                qoi: np.asarray(model.reference_values, dtype=float)
                for qoi, model in self.models.items()
            },
        )

        inconsistent_inputs = []
        for qoi, model in self.models.items():
            input_shapes = {
                tuple(int(dim) for dim in lgp.X_train.shape)
                for lgp in model.lgps
            }
            if len(input_shapes) != 1:
                inconsistent_inputs.append(qoi)
        if inconsistent_inputs:
            raise ValueError(
                "Surrogate committees with inconsistent training input shapes: "
                + ", ".join(sorted(inconsistent_inputs))
            )

        n_params = {model.n_params for model in self.models.values()}
        if len(n_params) != 1:
            raise ValueError("All surrogate models must have the same input dimension.")

        if self.constraint is not None and self.constraint.n_params != self.n_params:
            raise ValueError(
                "The selected surrogate models and the charge constraint disagree "
                f"on the number of explicit parameters: models expect "
                f"{self.n_params}, constraint defines {self.constraint.n_params}."
            )

        invalid = [
            qoi
            for qoi, model in self.models.items()
            if model.reference_values.size != model.y_size
        ]
        if invalid:
            raise ValueError(
                "Models with incompatible reference output size: "
                + ", ".join(sorted(invalid))
            )

        invalid_curve_schema = []
        for qoi, model in self.models.items():
            n_curves = int(getattr(model, "n_curves", 0))
            if n_curves <= 0 or model.reference_values.size % n_curves != 0:
                invalid_curve_schema.append(qoi)
        if invalid_curve_schema:
            raise ValueError(
                "Models with incompatible n_curves metadata: "
                + ", ".join(sorted(invalid_curve_schema))
            )

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
        if not hasattr(self.constraint, "explicit_parameter_names"):
            return None
        return list(self.constraint.explicit_parameter_names)

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
    def from_models(
        cls,
        models: Mapping[str, LGPCommittee],
        *,
        constraint: Optional[Callable] = None,
    ) -> "LearningProblem":
        return cls(models=dict(models), constraint=constraint)

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
    ) -> "LearningProblem":
        problem = LearningProblem(
            models=self.models,
            constraint=self.constraint,
        )
        object.__setattr__(
            problem,
            "observations",
            {
                qoi: torch.as_tensor(values, device=device, dtype=dtype)
                for qoi, values in self.observations.items()
            },
        )
        return problem

    def learn(
        self,
        *,
        priors_disttype: str = "normal",
        total_steps: int = 1500,
        warmup: int = 500,
        thin: int = 1,
        progress_stride: int = 100,
        n_walkers: Optional[int] = None,
        fn_posterior: PathLike = "./posterior.pt",
        fn_checkpoint: Optional[PathLike] = "./mcmc.ckpt",
        fn_priors: Optional[PathLike] = "./prior.pt",
        resume: bool = False,
        device: str = "cuda",
        logger: Optional[Logger] = None,
        rhat_tol: float = 1.01,
        ess_min: int = 100,
        include_implicit_charge: bool = False,
        compatibility: Optional[dict] = None,
    ) -> PosteriorResults:
        """Run posterior sampling for this learning problem."""
        owns_logger = logger is None
        logger = logger or Logger("learn")
        if owns_logger:
            logger.section("Posterior Learning")
            logger.blank()

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

        specs = getattr(self.constraint, "specs", None)
        specifications_fingerprint = (
            None
            if specs is None
            else mapping_fingerprint(specs.to_dict())
        )
        checkpoint_compatibility = dict(compatibility or {})
        checkpoint_compatibility.update(
            {
                "specifications_fingerprint": specifications_fingerprint,
                "n_params": self.n_params,
                "n_dimensions": len(priors),
                "n_walkers": n_walkers,
                "warmup": warmup,
                "thin": thin,
                "proposal": {
                    "type": "adaptive_gaussian",
                    "adapt": proposal.do_adapt,
                    "adapt_start": proposal.adapt_start,
                    "adapt_interval": proposal.adapt_interval,
                    "target_acceptance": proposal.target_acceptance,
                },
                "priors_disttype": priors_disttype,
            }
        )

        if fn_priors is not None:
            priors.write(
                fn_priors,
                metadata={
                    "specifications_fingerprint": specifications_fingerprint,
                },
            )

        print_progress_mcmc(
            sampler,
            initial_positions,
            total_steps=total_steps,
            warmup=warmup,
            thin=thin,
            progress_stride=progress_stride,
            logger=logger,
            restart=resume,
            fn_checkpoint=fn_checkpoint,
            rhat_tol=rhat_tol,
            ess_min=ess_min,
            checkpoint_compatibility=checkpoint_compatibility,
        )

        sampler.write_posterior(
            fn_posterior,
            metadata={
                "n_params": self.n_params,
                "qoi_names": self.qoi_names,
                "parameter_labels": list(self.parameter_names or []),
                "nuisance_labels": list(self.nuisance_names),
                "sample_labels": list(priors.names),
                "specifications_fingerprint": specifications_fingerprint,
                "model_fingerprints": checkpoint_compatibility.get(
                    "model_fingerprints", {}
                ),
                "effective_observations": {
                    qoi: float(model.n_eff) for qoi, model in self.models.items()
                },
                "mcmc": {
                    "priors_disttype": priors_disttype,
                    "total_steps": total_steps,
                    "warmup": warmup,
                    "thin": thin,
                    "n_walkers": n_walkers,
                    "rhat_tol": rhat_tol,
                    "ess_min": ess_min,
                    "converged": sampler.converged,
                },
            },
            priors=priors,
            sample_labels=list(priors.names),
            specs=specs,
        )
        return PosteriorResults.load(
            posterior=fn_posterior,
            priors=priors,
            specs=specs,
            include_implicit_charge=include_implicit_charge,
        )


def _default_checkpoint_path(fn_posterior: Path) -> Path:
    suffix = "".join(fn_posterior.suffixes) or ".pt"
    stem = fn_posterior.name[: -len(suffix)] if suffix else fn_posterior.name
    return fn_posterior.with_name(f"{stem}.ckpt{suffix}")
