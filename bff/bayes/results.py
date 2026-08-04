from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

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
    metadata: dict[str, Any] = field(default_factory=dict)

    _prepared_samples: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _tau: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _prepared_labels: Optional[list[str]] = field(default=None, init=False, repr=False)
    _prepare_info: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def load(
        cls,
        posterior: PathLike | np.ndarray | torch.Tensor,
        priors: Optional[Union[Priors, PathLike, list]] = None,
        sample_labels: Optional[list[str]] = None,
        sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        specs: Optional[Specs | PathLike] = None,
        include_implicit_charge: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> 'PosteriorResults':
        return cls(
            posterior=posterior,
            priors=priors,
            sample_labels=sample_labels,
            sample_transform=sample_transform,
            specs=specs,
            include_implicit_charge=include_implicit_charge,
            metadata=dict(metadata or {}),
        )

    def __post_init__(self) -> None:
        if self.specs is None and self._looks_like_specs_source(self.sample_labels):
            self.specs = self.sample_labels
            self.sample_labels = None
        if isinstance(self.posterior, (str, Path)):
            payload = self._load_posterior_payload(self.posterior)
            self.posterior = payload
            if not self.metadata:
                self.metadata = dict(payload.get('metadata', {}))
            if self.priors is None and payload.get('priors') is not None:
                raw_priors = payload['priors']
                self.priors = (
                    raw_priors['priors']
                    if isinstance(raw_priors, Mapping) and 'priors' in raw_priors
                    else raw_priors
                )
            if self.specs is None and payload.get('specs') is not None:
                self.specs = payload['specs']
            if (
                self.sample_labels is None
                and payload.get('sample_labels') is not None
            ):
                self.sample_labels = list(payload['sample_labels'])
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
    def _looks_like_specs_source(value: object) -> bool:
        if isinstance(value, Specs):
            return True
        if isinstance(value, dict):
            return 'bounds' in value and 'charge_constraints' in value
        if isinstance(value, (str, Path)):
            path = Path(value)
            return path.suffix in {'.yaml', '.yml'} and path.exists()
        return False

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
        posterior: PathLike | np.ndarray | torch.Tensor | Mapping[str, Any],
    ) -> np.ndarray:
        if isinstance(posterior, (np.ndarray, torch.Tensor)):
            posterior_array = cls._to_numpy(posterior)
        elif isinstance(posterior, Mapping):
            if 'posterior' not in posterior:
                raise KeyError('Posterior payload does not contain posterior samples.')
            posterior_array = cls._to_numpy(posterior['posterior'])
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
    def preparation_info(self) -> dict[str, Any]:
        return dict(self._prepare_info)

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
        n_before_outlier_filter = len(prepared)
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

        self._prepare_info = {
            'discard': int(discard),
            'thin': int(thin),
            'strip_outliers': bool(strip_outliers),
            'n_saved_steps': int(self.posterior.shape[0]),
            'n_walkers': int(self.posterior.shape[1]),
            'n_before_outlier_filter': int(n_before_outlier_filter),
            'n_prepared': int(len(self._prepared_samples)),
            'n_removed_outliers': int(
                n_before_outlier_filter - len(self._prepared_samples)
            ),
        }

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

    def _export_parameter_indices_and_labels(self) -> tuple[list[int], list[str]]:
        if self.specs is not None:
            explicit_names = list(self.specs.parameter_names(explicit_only=True))
            label_to_index = {label: i for i, label in enumerate(self.labels)}
            missing = [name for name in explicit_names if name not in label_to_index]
            if missing:
                raise ValueError(
                    'Prepared samples do not contain all explicit parameter '
                    'labels required by the specs: '
                    + ', '.join(repr(name) for name in missing)
                )
            return [label_to_index[name] for name in explicit_names], explicit_names

        if self.priors is not None:
            indices = [
                i for i, name in enumerate(self.priors.names)
                if not name.startswith('log_sigma_')
            ]
            labels = [self.labels[i] for i in indices]
            return indices, labels

        return list(range(self.prepared_samples.shape[1])), list(self.labels)

    def _parameter_draw_source(self) -> tuple[np.ndarray, list[str]]:
        indices, labels = self._export_parameter_indices_and_labels()
        samples = np.asarray(self.prepared_samples[:, indices], dtype=float)
        if samples.ndim != 2 or samples.shape[1] == 0:
            raise ValueError('No parameter columns are available for posterior draws.')
        if not np.all(np.isfinite(samples)):
            raise ValueError('Prepared posterior samples contain non-finite values.')
        return samples, labels

    @staticmethod
    def _coerce_rng(
        random_state: Optional[int | np.random.Generator],
    ) -> np.random.Generator:
        if isinstance(random_state, np.random.Generator):
            return random_state
        return np.random.default_rng(random_state)

    def _draw_posterior_approximation(
        self,
        samples: np.ndarray,
        n_samples: int,
        distribution: str,
        confidence: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if n_samples == 0:
            return np.empty((0, samples.shape[1]), dtype=float)

        if distribution == 'empirical':
            indices = rng.integers(0, len(samples), size=n_samples)
            return samples[indices].copy()

        if distribution == 'uniform':
            q_low = (1 - confidence) / 2
            lower, upper = np.quantile(samples, [q_low, 1 - q_low], axis=0)
            return rng.uniform(lower, upper, size=(n_samples, samples.shape[1]))

        if samples.shape[0] < 2:
            raise ValueError(
                f'{distribution} posterior sampling requires at least two samples.'
            )

        if distribution == 'normal':
            mean = np.mean(samples, axis=0)
            cov = np.atleast_2d(np.cov(samples, rowvar=False))
            if samples.shape[1] == 1:
                scale = float(np.sqrt(cov[0, 0]))
                return rng.normal(mean[0], scale, size=(n_samples, 1))
            return rng.multivariate_normal(mean, cov, size=n_samples)

        kde_samples = samples[:, 0] if samples.shape[1] == 1 else samples.T
        return gaussian_kde(kde_samples).resample(n_samples, seed=rng).T

    def sample_posterior(
        self,
        n_samples: int = 10,
        distribution: str = 'normal',
        confidence: float = 0.9,
        fn_out: Optional[PathLike] = None,
        overwrite: bool = False,
        random_state: Optional[int | np.random.Generator] = None,
        enforce_bounds: bool = True,
        max_attempts: Optional[int] = None,
        include_implicit_charge: bool = False,
    ) -> np.ndarray:
        """Draw parameter samples from a fitted posterior approximation."""
        if self._prepared_samples is None:
            self.prepare_samples()
        if n_samples < 0:
            raise ValueError('n_samples must be non-negative.')
        if not 0 < confidence < 1:
            raise ValueError('confidence must be between 0 and 1.')

        if include_implicit_charge and self.specs is None:
            raise ValueError(
                'include_implicit_charge=True requires Specs to be provided.'
            )

        posterior_samples, labels = self._parameter_draw_source()
        distribution = distribution.lower()
        if distribution not in {'empirical', 'kde', 'normal', 'uniform'}:
            raise ValueError(
                'distribution must be "empirical", "kde", "normal", or "uniform".'
            )
        rng = self._coerce_rng(random_state)
        must_validate = (
            self.specs is not None
            and (enforce_bounds or include_implicit_charge)
            and n_samples > 0
        )

        if not must_validate:
            explicit_draws = self._draw_posterior_approximation(
                posterior_samples,
                n_samples,
                distribution,
                confidence,
                rng,
            )
        else:
            assert self.specs is not None
            lower, upper = self.specs.bounds.array.T
            max_attempts = max_attempts or max(1000, 100 * n_samples)
            accepted: list[np.ndarray] = []
            n_accepted = 0
            attempts = 0

            while n_accepted < n_samples:
                if attempts >= max_attempts:
                    raise RuntimeError(
                        'Failed to draw enough samples satisfying the parameter '
                        f'bounds after {attempts} attempts.'
                    )
                n_remaining = n_samples - n_accepted
                batch_size = min(max(2 * n_remaining, 16), max_attempts - attempts)
                candidates = self._draw_posterior_approximation(
                    posterior_samples,
                    batch_size,
                    distribution,
                    confidence,
                    rng,
                )
                full_candidates = self.specs.with_implicit_charges(candidates)
                valid = np.logical_and(
                    full_candidates >= lower,
                    full_candidates <= upper,
                ).all(axis=1)
                accepted.append(candidates[valid])
                n_accepted += int(np.count_nonzero(valid))
                attempts += batch_size

            explicit_draws = np.concatenate(accepted, axis=0)[:n_samples]

        if include_implicit_charge:
            assert self.specs is not None
            draws = self.specs.with_implicit_charges(explicit_draws)
            labels = list(self.specs.parameter_names(explicit_only=False))
        else:
            draws = explicit_draws

        if fn_out:
            fn_out = Path(fn_out).resolve()
            if fn_out.exists() and not overwrite:
                raise FileExistsError(f"File '{fn_out}' already exists.")
            if fn_out.suffix == '.npy':
                np.save(fn_out, draws)
            elif fn_out.suffix in {'.yaml', '.yml'}:
                save_yaml(
                    {label: draws[:, i].tolist() for i, label in enumerate(labels)},
                    fn_out,
                )
            else:
                raise ValueError('fn_out must end with .npy, .yaml, or .yml')

        return draws

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
