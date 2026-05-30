from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from scipy.stats import gaussian_kde

from ..domain.specs import Specs
from ..io.utils import save_yaml
from ..mcmc.convergence import integrated_autocorr_time
from .priors import Priors

PathLike = Union[str, Path]


@dataclass
class PosteriorResults:
    posterior: np.ndarray
    priors: Optional[Priors] = None
    sample_labels: Optional[list[str]] = None
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
        sample_labels: Optional[list[str]] = None,
        sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        specs: Optional[Specs | PathLike] = None,
        include_implicit_charge: bool = False,
    ) -> 'PosteriorResults':
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
            if name.startswith('log_sigma_')
        ]

    def _default_sample_labels(self) -> list[str]:
        if self.priors is None:
            return [f'theta_{i}' for i in range(self.n_dim)]
        labels = []
        for i, name in enumerate(self.priors.names):
            if name.startswith('log_sigma_'):
                qoi = name.removeprefix('log_sigma_')
                labels.append(fr'$\sigma_{{\mathrm{{{qoi}}}}}$')
            else:
                labels.append(name or f'theta_{i}')
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
        if fn_in.suffix != '.pt':
            raise ValueError(
                f'Unsupported posterior file {fn_in!r}. '
                'PosteriorResults accepts only .pt files.'
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
            if 'posterior' not in payload:
                raise KeyError(
                    f'Posterior file {posterior!r} does not contain posterior samples.'
                )
            posterior_array = cls._to_numpy(payload['posterior'])
        else:
            raise TypeError(f'Unsupported posterior source: {type(posterior)}')

        if posterior_array.ndim != 3:
            raise ValueError(
                'Expected raw posterior with shape '
                f'(n_saved, n_walkers, n_dim), got {posterior_array.shape}.'
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
                'No prepared samples available. Call prepare_samples() first.'
            )
        return self._prepared_samples

    @property
    def has_prepared_samples(self) -> bool:
        return self._prepared_samples is not None

    @property
    def labels(self) -> list[str]:
        if self._prepared_samples is not None and self._prepared_labels is not None:
            return self._prepared_labels
        if self.sample_labels is not None:
            if len(self.sample_labels) != self.n_dim:
                raise ValueError(
                    'sample_labels length does not match the posterior dimension.'
                )
            return self.sample_labels
        return [f'theta_{i}' for i in range(self.n_dim)]

    def _labels_with_implicit_charges(self) -> list[str]:
        if self.specs is None:
            raise ValueError(
                'Implicit-charge expansion requires Specs to be attached to '
                'PosteriorResults.'
            )
        raw_labels = (
            list(self.sample_labels)
            if self.sample_labels is not None
            else [f'theta_{i}' for i in range(self.n_dim)]
        )
        n_explicit = self.specs.explicit_bounds.n_params
        if len(raw_labels) < n_explicit:
            raise ValueError(
                'Posterior labels do not cover all explicit parameters required '
                'by the specs.'
            )
        explicit_labels = dict(zip(
            self.specs.explicit_bounds.names,
            raw_labels[:n_explicit],
        ))
        return [
            explicit_labels.get(name, name) for name in self.specs.bounds.names
        ] + raw_labels[n_explicit:]

    def prepare_samples(
        self,
        discard: Optional[int] = None,
        thin: Optional[int] = None,
        sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        strip_outliers: bool = True,
    ) -> None:
        tau = self.autocorr_time
        discard = discard if discard is not None else int(2 * np.max(tau))
        thin = thin if thin is not None else max(1, int(0.5 * np.min(tau)))

        samples = self.posterior[discard::thin]
        if samples.size == 0:
            raise ValueError(
                'No posterior samples remain after applying discard/thin. '
                'Try smaller values.'
            )

        prepared = samples.reshape(-1, samples.shape[-1]).copy()
        transform_fn = sample_transform or self.sample_transform
        if transform_fn is not None:
            prepared = np.asarray(transform_fn(prepared), dtype=float)
        if self.include_implicit_charge:
            if self.specs is None:
                raise ValueError(
                    'include_implicit_charge=True requires Specs to be provided.'
                )
            prepared = self.specs.with_implicit_charges(prepared)
            self._prepared_labels = self._labels_with_implicit_charges()
        else:
            self._prepared_labels = (
                list(self.sample_labels)
                if self.sample_labels is not None
                else [f'theta_{i}' for i in range(prepared.shape[1])]
            )
        self._prepared_samples = prepared

        if strip_outliers:
            q_low = 0.01 / 2
            q_high = 1 - q_low
            lower = np.quantile(prepared, q_low, axis=0)
            upper = np.quantile(prepared, q_high, axis=0)
            mask = np.logical_and(prepared >= lower, prepared <= upper).all(axis=1)
            self._prepared_samples = prepared[mask]

    def posterior_mode(
        self,
        sample_id: int,
        x: Optional[np.ndarray] = None,
    ) -> tuple[float, float]:
        samples = self.prepared_samples[:, sample_id]
        if x is None:
            x = np.linspace(np.min(samples), np.max(samples), 1000)
        posterior = gaussian_kde(samples)
        y = posterior(x)
        idx = np.argmax(y)
        return float(x[idx]), float(y[idx])

    def summary(self) -> dict[str, Any]:
        if self._prepared_samples is None:
            self.prepare_samples()
        assert self._prepared_samples is not None

        labels = self.labels
        means = np.mean(self._prepared_samples, axis=0)
        stds = np.std(self._prepared_samples, axis=0)
        summary = {
            label: {
                'mean': float(mean),
                'std': float(std),
            }
            for label, mean, std in zip(labels, means, stds)
        }
        summary['autocorr_time'] = [float(x) for x in np.atleast_1d(self.autocorr_time)]
        return summary

    def write_summary(self, fn_out: PathLike) -> None:
        save_yaml(self.summary(), fn_out)
