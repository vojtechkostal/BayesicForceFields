"""Bayesian learning of force-field parameters from surrogate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import torch

from ..domain.specs import Specs
from ..io.logs import Logger, print_progress_mcmc
from ..io.utils import mapping_fingerprint
from ..mcmc.proposal import AdaptiveGaussianProposal
from ..mcmc.sampler import Sampler
from .gaussian_process import LGPCommittee
from .likelihoods import gaussian_log_likelihood, gaussian_log_likelihood_by_qoi
from .posterior import log_posterior
from .priors import Priors
from .results import Results
from .utils import evenly_spaced_indices, initialize_walkers, resolve_device

PathLike = Union[str, Path]


@dataclass(frozen=True, slots=True)
class LearningProblem:
    """Surrogate models of QoIs, to be learned against their references.

    ``specs`` defines the sampled parameters (the model inputs), their bounds,
    and the implicit charges; without it, parameters are unbounded.
    """

    models: dict[str, LGPCommittee]
    specs: Optional[Specs] = None
    observations: dict[str, np.ndarray | torch.Tensor] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("LearningProblem requires at least one surrogate model.")

        empty_models = [qoi for qoi, model in self.models.items() if not model.members]
        if empty_models:
            raise ValueError(
                "Surrogate models without committee members: "
                + ", ".join(sorted(empty_models))
            )

        object.__setattr__(
            self,
            "observations",
            {
                qoi: np.asarray(model.y_ref, dtype=float)
                for qoi, model in self.models.items()
            },
        )

        inconsistent_inputs = []
        for qoi, model in self.models.items():
            input_shapes = {
                tuple(int(dim) for dim in member.X_train.shape)
                for member in model.members
            }
            if len(input_shapes) != 1:
                inconsistent_inputs.append(qoi)
        if inconsistent_inputs:
            raise ValueError(
                "Surrogate committees with inconsistent training input shapes: "
                + ", ".join(sorted(inconsistent_inputs))
            )

        n_params = {model.n_inputs for model in self.models.values()}
        if len(n_params) != 1:
            raise ValueError("All surrogate models must have the same input dimension.")

        if self.specs is not None and len(self.specs.explicit_names) != self.n_params:
            raise ValueError(
                "The selected surrogate models and the specification disagree "
                f"on the number of sampled parameters: models expect "
                f"{self.n_params}, specs define {len(self.specs.explicit_names)}."
            )

        invalid = [
            qoi
            for qoi, model in self.models.items()
            if model.y_ref.size != model.n_outputs
        ]
        if invalid:
            raise ValueError(
                "Models with incompatible reference output size: "
                + ", ".join(sorted(invalid))
            )

        invalid_curve_schema = []
        for qoi, model in self.models.items():
            n_curves = int(getattr(model, "n_curves", 0))
            if n_curves <= 0 or model.y_ref.size % n_curves != 0:
                invalid_curve_schema.append(qoi)
        if invalid_curve_schema:
            raise ValueError(
                "Models with incompatible n_curves metadata: "
                + ", ".join(sorted(invalid_curve_schema))
            )

    @property
    def qoi_names(self) -> list[str]:
        """Names of the QoIs, in model order."""
        return list(self.models)

    @property
    def n_params(self) -> int:
        """Number of sampled parameters."""
        return next(iter(self.models.values())).n_inputs

    @property
    def parameter_bounds(self) -> np.ndarray:
        """Bounds of the sampled parameters, shape ``(n_params, 2)``."""
        if self.specs is None:
            return np.tile([-1e5, 1e5], (self.n_params, 1))
        return self.specs.explicit_bounds

    @property
    def parameter_names(self) -> list[str] | None:
        """Names of the sampled parameters; ``None`` without ``specs``."""
        return None if self.specs is None else list(self.specs.explicit_names)

    @property
    def nuisances(self) -> list[str]:
        """QoIs whose noise (sigma) is learned: those without a fixed nuisance."""
        return [qoi for qoi, model in self.models.items() if model.nuisance is None]

    @classmethod
    def from_models(
        cls,
        models: Mapping[str, LGPCommittee],
        *,
        specs: Optional[Specs] = None,
    ) -> LearningProblem:
        """Problem of ``models`` (QoI name to committee) and optional ``specs``."""
        return cls(models=dict(models), specs=specs)

    def build_priors(self, dist_type: str = "normal") -> Priors:
        """Priors of the sampled parameters (``normal`` or ``uniform`` over
        their bounds), then of the log noise of each learned nuisance."""
        return Priors.from_bounds(
            self.parameter_bounds,
            dist_type=dist_type,
            names=self.parameter_names,
            nuisance_names=[f"log noise {qoi}" for qoi in self.nuisances],
        )

    def warn_if_bounds_exceed_training_samples(
        self, logger: Logger, fraction: float = 0.1
    ) -> None:
        """Warn about parameters whose bounds extend over more than ``fraction``
        of their width beyond the samples a surrogate was trained on: there it
        falls back to its mean and the data cannot constrain the parameter."""
        if self.specs is None:
            return
        lower, upper = self.parameter_bounds.T
        for qoi, model in self.models.items():
            X = model.members[0].X_train.cpu().numpy()
            slack = fraction * (upper - lower)
            outside = (X.min(axis=0) - lower > slack) | (upper - X.max(axis=0) > slack)
            if outside.any():
                names = [n for n, flag in zip(self.parameter_names, outside) if flag]
                logger.warn(
                    f"{qoi}: the bounds of {', '.join(names)} extend well beyond "
                    "the training samples; the surrogate falls back to its mean "
                    "there and the posterior is unconstrained. Narrow the bounds "
                    "or sample more widely.",
                    level=1,
                )

    def to_torch(
        self,
        device: str,
        dtype: torch.dtype = torch.float32,
    ) -> LearningProblem:
        """Copy with the reference observations as tensors on ``device``.

        The surrogate models move to ``device`` and ``dtype`` too, in place:
        the device is chosen once and nothing is transferred afterwards.
        """
        for model in self.models.values():
            model.to(device, dtype)
        problem = LearningProblem(models=self.models, specs=self.specs)
        object.__setattr__(
            problem,
            "observations",
            {
                qoi: torch.as_tensor(values, device=device, dtype=dtype)
                for qoi, values in self.observations.items()
            },
        )
        return problem

    def log_likelihood_by_qoi(
        self,
        theta: np.ndarray,
        device: str,
        batch_size: int = 256,
    ) -> dict[str, np.ndarray]:
        """Log likelihood of each QoI at the rows of ``theta`` (sampled
        parameters, then the log noise of each learned nuisance).

        Batches are halved when a CUDA allocation fails.
        """
        problem = self.to_torch(device)
        chunks: dict[str, list[np.ndarray]] = {}
        start = 0
        size = max(1, min(batch_size, len(theta)))
        with torch.inference_mode():
            while start < len(theta):
                stop = min(start + size, len(theta))
                try:
                    batch = torch.as_tensor(
                        theta[start:stop], dtype=torch.float32, device=device
                    )
                    for qoi, values in gaussian_log_likelihood_by_qoi(
                        batch, problem
                    ).items():
                        chunks.setdefault(qoi, []).append(values.cpu().numpy())
                    start = stop
                except torch.OutOfMemoryError:
                    if not str(device).startswith("cuda") or size == 1:
                        raise
                    size = max(1, size // 2)
                    torch.cuda.empty_cache()
        return {qoi: np.concatenate(values) for qoi, values in chunks.items()}

    def learn(
        self,
        *,
        priors_disttype: str = "normal",
        total_steps: int = 1500,
        warmup: int = 500,
        thin: int = 1,
        progress_stride: int = 100,
        n_walkers: Optional[int] = None,
        fn_results: PathLike = "./results.pt",
        fn_checkpoint: Optional[PathLike] = "./mcmc.ckpt",
        resume: bool = False,
        device: str = "auto",
        logger: Optional[Logger] = None,
        rhat_tol: float = 1.01,
        ess_min: int = 400,
        max_qoi_samples: int = 10_000,
        qoi_batch_size: int = 256,
        compatibility: Optional[dict] = None,
    ) -> Results:
        """Sample the posterior; write ``fn_results`` and return it.

        ``max_qoi_samples`` posterior samples (evenly spaced) get their
        per-QoI log likelihood stored for :func:`bff.plotting.plot_qoi_marginals`.
        """
        owns_logger = logger is None
        logger = logger or Logger("learn")
        if owns_logger:
            logger.section("Learn")
            logger.blank()

        device = resolve_device(device)
        self.warn_if_bounds_exceed_training_samples(logger)
        priors = self.build_priors(dist_type=priors_disttype)
        n_walkers = 5 * len(priors) if n_walkers is None else n_walkers
        initial_positions = initialize_walkers(priors, n_walkers, self.specs)
        proposal_cov = torch.diag(
            torch.tensor(priors.scales, dtype=torch.float32, device=device) ** 2
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
        )
        sampler = Sampler(
            log_prob=log_probability,
            proposal=proposal,
            device=device,
            dtype=torch.float32,
        )

        fn_results = Path(fn_results).resolve()
        if fn_checkpoint is None:
            fn_checkpoint = fn_results.with_suffix(".ckpt")
        fn_checkpoint = Path(fn_checkpoint).resolve()

        specs_fingerprint = (
            None if self.specs is None else mapping_fingerprint(self.specs.to_dict())
        )
        checkpoint_compatibility = dict(compatibility or {})
        checkpoint_compatibility.update(
            {
                "specifications_fingerprint": specs_fingerprint,
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

        chain = sampler.chain.detach().cpu().numpy()
        flat = chain.reshape(-1, chain.shape[2])
        index = evenly_spaced_indices(len(flat), max_qoi_samples)
        log_likelihood_by_qoi = self.log_likelihood_by_qoi(
            flat[index], device, qoi_batch_size
        )
        diag = sampler.diagnostics
        results = Results(
            chain=chain,
            log_prob=sampler.chain_logp.detach().cpu().numpy(),
            specs=self.specs if self.specs is not None else _unbounded_specs(self),
            prior=priors,
            nuisances=self.nuisances,
            qoi={
                qoi: {
                    "n_eff": float(model.n_eff),
                    "tolerance": float(getattr(model, "tolerance", 0.0)),
                    "nuisance": model.nuisance,
                }
                for qoi, model in self.models.items()
            },
            qoi_index=index,
            qoi_log_likelihood=log_likelihood_by_qoi,
            info={
                "mcmc": {
                    "priors_disttype": priors_disttype,
                    "total_steps": total_steps,
                    "warmup": warmup,
                    "thin": thin,
                    "n_walkers": n_walkers,
                    "rhat_tol": rhat_tol,
                    "ess_min": ess_min,
                    "converged": sampler.converged,
                    "max_rhat": None if diag is None else diag.max_rhat,
                    "min_ess": None if diag is None else diag.min_ess,
                },
                "specifications_fingerprint": specs_fingerprint,
                "compatibility": dict(compatibility or {}),
            },
        )
        results.save(fn_results)
        return results


def _unbounded_specs(problem: LearningProblem) -> Specs:
    """Specification of parameters without bounds or constraints."""
    names = [f"theta_{i}" for i in range(problem.n_params)]
    return Specs(
        {"bounds": {name: [-1e5, 1e5] for name in names}, "charge_constraints": []}
    )
