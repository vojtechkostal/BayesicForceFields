from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from scipy.stats import gaussian_kde

from .priors import Priors
from ..io.utils import save_yaml
from ..mcmc.convergence import integrated_autocorr_time


PathLike = Union[str, Path]


@dataclass
class InferenceResults:
    posterior: np.ndarray
    priors: Optional[Priors] = None
    parameter_names: Optional[List[str]] = None
    sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None

    _posterior_samples: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _tau: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @classmethod
    def load(
        cls,
        posterior: PathLike | np.ndarray | torch.Tensor,
        priors: Optional[Union[Priors, PathLike, list]] = None,
        parameter_names: Optional[List[str]] = None,
        sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> "InferenceResults":
        return cls(
            posterior=posterior,
            priors=priors,
            parameter_names=parameter_names,
            sample_transform=sample_transform,
        )

    def __post_init__(self) -> None:
        self.posterior = self._coerce_posterior(self.posterior)
        if self.priors is not None:
            self.priors = self._coerce_priors(self.priors)
        if self.parameter_names is None and self.priors is not None:
            self.parameter_names = self._default_labels()
        if self.sample_transform is None and self._nuisance_indices:
            self.sample_transform = self._default_transform

    @property
    def _nuisance_indices(self) -> list[int]:
        if self.priors is None:
            return []
        return [
            i for i, name in enumerate(self.priors.names)
            if name.startswith("log_sigma_")
        ]

    def _default_labels(self) -> list[str]:
        if self.priors is None:
            return [f"theta_{i}" for i in range(self.n_dim)]
        labels = []
        for i, name in enumerate(self.priors.names):
            if name.startswith("log_sigma_"):
                qoi = name.removeprefix("log_sigma_")
                labels.append(f"$\\sigma_{{\\mathrm{{{qoi}}}}}$")
            else:
                labels.append(name or f"theta_{i}")
        return labels

    def _default_transform(self, samples: np.ndarray) -> np.ndarray:
        transformed = np.asarray(samples, dtype=float).copy()
        if self._nuisance_indices:
            transformed[:, self._nuisance_indices] = np.exp(
                transformed[:, self._nuisance_indices]
            )
        return transformed

    @staticmethod
    def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    @staticmethod
    def _coerce_priors(priors: Priors | PathLike | list) -> Priors:
        if isinstance(priors, Priors):
            return priors
        return Priors.from_any(priors)

    @staticmethod
    def _load_posterior_payload(fn_in: PathLike) -> dict[str, Any]:
        fn_in = Path(fn_in)
        if fn_in.suffix != ".pt":
            raise ValueError(
                f"Unsupported posterior file {fn_in!r}. "
                "InferenceResults accepts only .pt files."
            )
        return torch.load(fn_in, weights_only=False)

    @classmethod
    def _coerce_posterior(
        cls,
        posterior: PathLike | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        if isinstance(posterior, (np.ndarray, torch.Tensor)):
            posterior_array = cls._to_numpy(posterior)
        elif isinstance(posterior, (str, Path)):
            payload = cls._load_posterior_payload(posterior)
            if "posterior" not in payload:
                raise KeyError(
                    f"Posterior file {posterior!r} does not contain posterior samples."
                )
            posterior_array = cls._to_numpy(payload["posterior"])
        else:
            raise TypeError(f"Unsupported posterior source: {type(posterior)}")

        if posterior_array.ndim != 3:
            raise ValueError(
                "Expected raw posterior with shape "
                f"(n_saved, n_walkers, n_dim), got {posterior_array.shape}."
            )
        return posterior_array

    @property
    def n_dim(self) -> int:
        return self.posterior.shape[-1]

    @property
    def autocorr_time(self) -> np.ndarray:
        if self._tau is None:
            tau = integrated_autocorr_time(
                torch.as_tensor(self.posterior, dtype=torch.float32)
            ).mean(dim=0)
            self._tau = np.atleast_1d(tau.detach().cpu().numpy().astype(float))
        return self._tau

    @property
    def samples(self) -> np.ndarray:
        if self._posterior_samples is None:
            raise ValueError(
                "No prepared samples available. Call prepare_samples() first."
            )
        return self._posterior_samples

    @property
    def is_prepared(self) -> bool:
        return self._posterior_samples is not None

    @property
    def labels(self) -> List[str]:
        if self.parameter_names is not None:
            if len(self.parameter_names) != self.n_dim:
                raise ValueError(
                    "parameter_names length does not match the posterior dimension."
                )
            return self.parameter_names
        return [f"theta_{i}" for i in range(self.n_dim)]

    def prepare_samples(
        self,
        discard: Optional[int] = None,
        thin: Optional[int] = None,
        sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        strip_outliers: bool = True
    ) -> None:
        tau = self.autocorr_time
        discard = discard if discard is not None else int(2 * np.max(tau))
        thin = thin if thin is not None else max(1, int(0.5 * np.min(tau)))

        samples = self.posterior[discard::thin]
        if samples.size == 0:
            raise ValueError(
                "No posterior samples remain after applying discard/thin. "
                "Try smaller values."
            )

        prepared = samples.reshape(-1, samples.shape[-1]).copy()
        transform_fn = sample_transform or self.sample_transform
        if transform_fn is not None:
            prepared = np.asarray(transform_fn(prepared), dtype=float)
        self._posterior_samples = prepared

        if strip_outliers:
            # Remove samples outside the 99.9% quantiles to avoid skewing the plots.
            q_low = 0.01 / 2
            q_high = 1 - q_low
            confint = np.quantile(prepared, [q_low, q_high], axis=0)
            mask = np.all((prepared >= confint[0]) & (prepared <= confint[1]), axis=1)
            self._posterior_samples = prepared[mask]

    @property
    def map_estimates(self) -> dict[str, float]:
        modes: list[float] = []
        for col in self.samples.T:
            grid = np.linspace(col.min(), col.max(), 1000)
            density = gaussian_kde(col)
            modes.append(float(np.round(grid[np.argmax(density(grid))], 3)))
        return dict(zip(self.labels, modes))

    def quantiles(self, confidence: float = 0.95) -> np.ndarray:
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        return np.quantile(self.samples, [q_low, 0.5, q_high], axis=0)

    def sample_posterior(
        self,
        n_samples: int = 10,
        distribution: str = "normal",
        confidence: float = 0.9,
        fn_out: Optional[str] = None,
        overwrite: bool = False,
    ) -> np.ndarray:
        param_samples = self.samples
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        confint = np.quantile(param_samples, [q_low, q_high], axis=0)

        if distribution == "normal":
            mean = np.mean(param_samples, axis=0)
            cov = np.cov(param_samples, rowvar=False)
            if np.ndim(cov) == 0:
                draws = np.random.normal(mean, np.sqrt(cov), size=n_samples)[:, None]
            else:
                draws = np.random.multivariate_normal(mean, cov, size=n_samples)
        elif distribution == "uniform":
            widths = np.diff(confint, axis=0).ravel()
            draws = np.random.rand(n_samples, widths.size) * widths + confint[0]
        else:
            raise ValueError('distribution must be "uniform" or "normal".')

        if fn_out:
            fn_out = Path(fn_out).resolve()
            if fn_out.exists() and not overwrite:
                raise FileExistsError(f"File '{fn_out}' already exists.")
            if fn_out.name.endswith(".npy"):
                np.save(fn_out, draws)
            elif fn_out.name.endswith(".yaml"):
                save_yaml(
                    {
                        "schema_version": 1,
                        "kind": "bff.parameter_samples",
                        "parameter_names": list(self.labels),
                        "n_samples": int(draws.shape[0]),
                        "samples": [
                            {
                                "sample_id": f"{i:03d}",
                                "params": {
                                    label: float(value)
                                    for label, value in zip(self.labels, row)
                                },
                            }
                            for i, row in enumerate(draws)
                        ],
                    },
                    fn_out,
                )
            else:
                raise ValueError("fn_out must end with .npy or .yaml")

        return draws
