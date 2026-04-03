from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from scipy.stats import gaussian_kde

from ..domain.specs import Specs
from ..io.utils import save_yaml
from ..mcmc.convergence import integrated_autocorr_time
from .priors import Priors

PathLike = Union[str, Path]


@dataclass
class InferenceResults:
    posterior: np.ndarray
    priors: Optional[Priors] = None
    sample_labels: Optional[List[str]] = None
    sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    specs: Optional[Specs] = None
    include_implicit_charge: bool = False

    _prepared_samples: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _tau: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prepared_labels: Optional[list[str]] = field(default=None, init=False, repr=False)

    @classmethod
    def load(
        cls,
        posterior: PathLike | np.ndarray | torch.Tensor,
        priors: Optional[Union[Priors, PathLike, list]] = None,
        sample_labels: Optional[List[str]] = None,
        sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        specs: Optional[Specs | PathLike] = None,
        include_implicit_charge: bool = False,
    ) -> "InferenceResults":
        return cls(
            posterior=posterior,
            priors=priors,
            sample_labels=sample_labels,
            sample_transform=sample_transform,
            specs=specs,
            include_implicit_charge=include_implicit_charge,
        )

    def __post_init__(self) -> None:
        self.posterior = self._coerce_posterior(self.posterior)
        if self.priors is not None:
            self.priors = self._coerce_priors(self.priors)
        if self.specs is not None and not isinstance(self.specs, Specs):
            self.specs = Specs(self.specs)
        if self.sample_labels is None and self.priors is not None:
            self.sample_labels = self._default_sample_labels()
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

    def _default_sample_labels(self) -> list[str]:
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
    def prepared_samples(self) -> np.ndarray:
        if self._prepared_samples is None:
            raise ValueError(
                "No prepared samples available. Call prepare_samples() first."
            )
        return self._prepared_samples

    @property
    def has_prepared_samples(self) -> bool:
        return self._prepared_samples is not None

    @property
    def labels(self) -> List[str]:
        if self._prepared_samples is not None and self._prepared_labels is not None:
            return self._prepared_labels
        if self.sample_labels is not None:
            if len(self.sample_labels) != self.n_dim:
                raise ValueError(
                    "sample_labels length does not match the posterior dimension."
                )
            return self.sample_labels
        return [f"theta_{i}" for i in range(self.n_dim)]

    def _labels_with_implicit_charge(self) -> list[str]:
        if self.specs is None:
            raise ValueError(
                "Implicit-charge expansion requires Specs to be attached to "
                "InferenceResults."
            )
        raw_labels = (
            list(self.sample_labels)
            if self.sample_labels is not None
            else [f"theta_{i}" for i in range(self.n_dim)]
        )
        n_explicit = self.specs.explicit_bounds.n_params
        if len(raw_labels) < n_explicit:
            raise ValueError(
                "Posterior labels do not cover all explicit parameters required "
                "by the specs."
            )
        insert_at = self.specs.implicit_param_index
        implicit_param = self.specs.implicit_param
        return (
            raw_labels[:insert_at]
            + [implicit_param]
            + raw_labels[insert_at:]
        )

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
        if self.include_implicit_charge:
            if self.specs is None:
                raise ValueError(
                    "include_implicit_charge=True requires Specs to be provided."
                )
            prepared = self.specs.with_implicit_charge(prepared)
            self._prepared_labels = self._labels_with_implicit_charge()
        else:
            self._prepared_labels = (
                list(self.sample_labels)
                if self.sample_labels is not None
                else [f"theta_{i}" for i in range(prepared.shape[1])]
            )
        self._prepared_samples = prepared

        if strip_outliers:
            # Remove samples outside the 99% central interval to avoid skewing plots.
            q_low = 0.01 / 2
            q_high = 1 - q_low
            confint = np.quantile(prepared, [q_low, q_high], axis=0)
            mask = np.all((prepared >= confint[0]) & (prepared <= confint[1]), axis=1)
            self._prepared_samples = prepared[mask]

    @property
    def map_estimates(self) -> dict[str, float]:
        modes: list[float] = []
        for col in self.prepared_samples.T:
            grid = np.linspace(col.min(), col.max(), 1000)
            density = gaussian_kde(col)
            modes.append(float(np.round(grid[np.argmax(density(grid))], 3)))
        return dict(zip(self.labels, modes))

    def quantiles(self, confidence: float = 0.95) -> np.ndarray:
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        return np.quantile(self.prepared_samples, [q_low, 0.5, q_high], axis=0)

    def _export_parameter_indices_and_labels(self) -> tuple[list[int], list[str]]:
        if self.specs is not None:
            explicit_names = list(self.specs.parameter_names(explicit_only=True))
            label_to_index = {label: i for i, label in enumerate(self.labels)}
            missing = [name for name in explicit_names if name not in label_to_index]
            if missing:
                raise ValueError(
                    "Prepared samples do not contain all explicit parameter "
                    "labels required by the specs: "
                    + ", ".join(repr(name) for name in missing)
                )
            return [label_to_index[name] for name in explicit_names], explicit_names

        if self.priors is not None:
            indices = [
                i for i, name in enumerate(self.priors.names)
                if not name.startswith("log_sigma_")
            ]
            labels = [self.labels[i] for i in indices]
            return indices, labels

        return list(range(self.prepared_samples.shape[1])), list(self.labels)

    def _parameter_draw_source(self) -> tuple[np.ndarray, list[str]]:
        indices, labels = self._export_parameter_indices_and_labels()
        return np.asarray(self.prepared_samples[:, indices], dtype=float), labels

    def sample_parameters(
        self,
        n_samples: int = 10,
        distribution: str = "normal",
        confidence: float = 0.9,
        fn_out: Optional[str] = None,
        overwrite: bool = False,
    ) -> np.ndarray:
        param_samples, export_labels = self._parameter_draw_source()
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
            if fn_out.suffix != ".yaml":
                raise ValueError("fn_out must end with .yaml")
            save_yaml(
                {
                    label: draws[:, i].tolist()
                    for i, label in enumerate(export_labels)
                },
                fn_out,
            )

        return np.asarray(draws, dtype=float)
