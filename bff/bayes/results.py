"""Results of a learning run: prior, posterior samples, and what they mean.

``bff learn`` writes one file, ``results.pt``, that holds everything needed to
inspect, summarize, plot, and reuse a run::

    results = Results.load("06-learn/outputs/results.pt")
    results.summary()            # mean, std, quantiles, mode, and MAP per parameter
    results.map                  # the maximum a posteriori parameters
    results["charge C1"]         # posterior samples of one parameter
    results.draw(10)             # parameter draws for validation
    results.plot_marginals()     # figures (see bff.plotting)

``results.samples`` are the posterior samples in physical units, one column
per name in ``results.names``: every parameter of the specification (sampled
and implicit charges), then one ``noise <qoi>`` (the noise sigma of a QoI) per
learned nuisance. The prior lives next to the posterior (``results.prior``),
not inside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from scipy.stats import gaussian_kde

from ..domain.specs import Specs
from ..io.utils import atomic_torch_save, save_yaml
from ..mcmc.convergence import diagnose
from .priors import Priors
from .utils import evenly_spaced_indices

PathLike = Union[str, Path]
FORMAT = 1


def marginal_mode(values: np.ndarray, lower: float, upper: float) -> float:
    """Mode of the kernel density estimate of ``values``, within the bounds."""
    values = np.asarray(values, dtype=float)
    values = values[evenly_spaced_indices(len(values), 5000)]
    grid_lower = max(float(lower), float(values.min()))
    grid_upper = min(float(upper), float(values.max()))
    if grid_lower >= grid_upper:
        return grid_lower
    grid = np.linspace(grid_lower, grid_upper, 512)
    try:
        return float(grid[np.argmax(gaussian_kde(values)(grid))])
    except np.linalg.LinAlgError:
        return float(np.median(values))


class Results:
    """Prior, posterior samples, and run information of one learning run.

    Parameters
    ----------
    chain : array, shape ``(n_saved, n_walkers, n_dim)``
        Sampler states after warmup. Columns are the sampled parameters in
        ``specs.explicit_names`` order, then one *logarithm* of the noise
        (nuisance) of each QoI in ``nuisances``.
    log_prob : array, shape ``(n_saved, n_walkers)``
        Log posterior of every state; its maximum defines the MAP.
    specs : Specs
        Parameter specification of the run.
    prior : Priors, optional
        Priors of the ``n_dim`` chain columns.
    nuisances : sequence of str
        QoI names whose noise was learned, in chain-column order.
    qoi : mapping, optional
        Per QoI: ``n_eff``, ``tolerance``, and a fixed ``nuisance`` if any.
    qoi_index, qoi_log_likelihood : optional
        Indices (into the flattened chain) of the samples for which each QoI's
        log likelihood was evaluated, and those log likelihoods per QoI. They
        feed :func:`bff.plotting.plot_qoi_marginals`.
    info : mapping, optional
        MCMC settings, convergence, and fingerprints; free-form.
    """

    def __init__(
        self,
        chain: np.ndarray | torch.Tensor,
        log_prob: np.ndarray | torch.Tensor,
        specs: Specs,
        prior: Optional[Priors] = None,
        nuisances: Sequence[str] = (),
        qoi: Optional[Mapping[str, Mapping[str, Any]]] = None,
        qoi_index: Optional[np.ndarray] = None,
        qoi_log_likelihood: Optional[Mapping[str, np.ndarray]] = None,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.chain = np.asarray(_numpy(chain), dtype=float)
        self.log_prob = np.asarray(_numpy(log_prob), dtype=float)
        self.specs = specs if isinstance(specs, Specs) else Specs(specs)
        self.prior = prior
        self.nuisances = tuple(nuisances)
        self.qoi = {name: dict(values) for name, values in (qoi or {}).items()}
        self.qoi_index = None if qoi_index is None else np.asarray(qoi_index, dtype=int)
        self.qoi_log_likelihood = {
            name: np.asarray(values, dtype=float)
            for name, values in (qoi_log_likelihood or {}).items()
        }
        self.info = dict(info or {})

        n_dim = len(self.specs.explicit_names) + len(self.nuisances)
        if self.chain.ndim != 3 or self.chain.shape[2] != n_dim:
            raise ValueError(
                f"chain must have shape (n_saved, n_walkers, {n_dim}) for "
                f"{n_dim} sampled columns, got {self.chain.shape}."
            )
        if self.log_prob.shape != self.chain.shape[:2]:
            raise ValueError(
                f"log_prob must have shape {self.chain.shape[:2]}, "
                f"got {self.log_prob.shape}."
            )
        if prior is not None and len(prior) != n_dim:
            raise ValueError(f"prior has {len(prior)} entries for {n_dim} columns.")
        if self.qoi_log_likelihood and self.qoi_index is None:
            raise ValueError("qoi_log_likelihood requires qoi_index.")

        explicit = self.chain.reshape(-1, n_dim)[:, :len(self.specs.explicit_names)]
        log_sigma = self.chain.reshape(-1, n_dim)[:, len(self.specs.explicit_names):]
        self.samples = np.column_stack(
            [self.specs.complete(explicit), np.exp(log_sigma)]
        )
        self._map_index = int(np.argmax(self.log_prob.reshape(-1)))

    # -- names ---------------------------------------------------------------

    @property
    def explicit_names(self) -> tuple[str, ...]:
        """Sampled parameters."""
        return self.specs.explicit_names

    @property
    def implicit_names(self) -> tuple[str, ...]:
        """Charges computed from the sampled parameters."""
        return self.specs.implicit_names

    @property
    def nuisance_names(self) -> tuple[str, ...]:
        """Learned noise (sigma) of each QoI, named ``noise <qoi>``."""
        return tuple(f"noise {qoi}" for qoi in self.nuisances)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the columns of ``samples``."""
        return self.specs.names + self.nuisance_names

    @property
    def n_samples(self) -> int:
        """Number of posterior samples (saved steps times walkers)."""
        return len(self.samples)

    def __getitem__(self, name: str) -> np.ndarray:
        """Posterior samples of one parameter."""
        try:
            return self.samples[:, self.names.index(name)]
        except ValueError:
            raise KeyError(f"{name!r} is not one of {list(self.names)}.") from None

    def __repr__(self) -> str:
        return (
            f"Results({self.n_samples} samples of {len(self.names)} parameters "
            f"from {self.chain.shape[1]} walkers)"
        )

    # -- point estimates -----------------------------------------------------

    @property
    def map(self) -> dict[str, float]:
        """Maximum a posteriori parameters: the saved state of highest log
        posterior, with implicit charges and noise in physical units."""
        return dict(zip(self.names, map(float, self.samples[self._map_index])))

    def summary(self) -> dict[str, dict[str, float]]:
        """Mean, standard deviation, median, 16/84 % quantiles, mode (of the
        marginal density), and MAP of every parameter."""
        q16, median, q84 = np.quantile(self.samples, [0.16, 0.5, 0.84], axis=0)
        summary = {}
        for j, name in enumerate(self.names):
            column = self.samples[:, j]
            lower, upper = self.specs.bounds.get(name, (-np.inf, np.inf))
            summary[name] = {
                "mean": float(column.mean()),
                "std": float(column.std()),
                "median": float(median[j]),
                "q16": float(q16[j]),
                "q84": float(q84[j]),
                "mode": marginal_mode(column, lower, upper),
                "map": float(self.samples[self._map_index, j]),
            }
        return summary

    def write_summary(self, fn_out: PathLike) -> None:
        """Write :meth:`summary` to a YAML file."""
        save_yaml(self.summary(), fn_out)

    def diagnostics(self) -> dict[str, dict[str, float]]:
        """R-hat, bulk ESS, and tail ESS of every chain column (noise as
        ``log noise <qoi>``) and of the log probability, as the sampler's
        convergence check computes them."""
        diagnostics = diagnose(
            torch.as_tensor(self.chain), torch.as_tensor(self.log_prob)
        )
        names = self.explicit_names + tuple(f"log noise {q}" for q in self.nuisances)
        return {
            name: {
                "rhat": float(diagnostics.rhat[j]),
                "ess_bulk": float(diagnostics.ess_bulk[j]),
                "ess_tail": float(diagnostics.ess_tail[j]),
            }
            for j, name in enumerate((*names, "log_prob"))
        }

    # -- prior and draws -----------------------------------------------------

    def prior_density(self, name: str, grid: np.ndarray) -> Optional[np.ndarray]:
        """Prior density of a sampled parameter on ``grid``, or ``None``."""
        if self.prior is None or name not in self.explicit_names:
            return None
        distribution = self.prior[self.explicit_names.index(name)].distribution
        values = torch.as_tensor(grid, dtype=torch.float32)
        return distribution.log_prob(values).exp().numpy()

    def draw(
        self,
        n: int = 10,
        *,
        distribution: str = "normal",
        confidence: float = 0.9,
        seed: Optional[int | np.random.Generator] = None,
        include_mean: bool = False,
        include_map: bool = False,
        implicit: bool = False,
        fn_out: Optional[PathLike] = None,
        overwrite: bool = False,
    ) -> dict[str, np.ndarray]:
        """Draw parameter sets from an approximation of the posterior.

        The approximation is fitted to the posterior samples of the sampled
        parameters: ``normal`` (multivariate), ``kde``, ``uniform`` (within
        the central ``confidence`` interval), or ``empirical`` (resampling).
        Draws whose implicit charges leave their bounds are redrawn. The
        posterior mean (``include_mean``) and the MAP (``include_map``) are
        prepended as the first sets.

        Returns a mapping from parameter name to values, ready for
        ``yaml`` export (``fn_out``) and for explicit-mode validation; with
        ``implicit`` it also holds the implicit charges.
        """
        if n < 0:
            raise ValueError("n must be non-negative.")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1.")
        if distribution not in {"empirical", "kde", "normal", "uniform"}:
            raise ValueError(
                'distribution must be "empirical", "kde", "normal", or "uniform".'
            )
        if n == 0 and not (include_mean or include_map):
            raise ValueError("Nothing to draw: n is 0 and no mean or MAP requested.")
        specs = self.specs
        columns = [self.names.index(name) for name in self.explicit_names]
        samples = self.samples[:, columns]
        rng = np.random.default_rng(seed)

        fixed = []
        if include_mean:
            mean = samples.mean(axis=0, keepdims=True)
            if not specs.is_valid(mean).all():
                raise ValueError("The posterior mean violates the parameter bounds.")
            fixed.append(mean)
        if include_map:
            fixed.append(samples[self._map_index:self._map_index + 1])
        accepted: list[np.ndarray] = []
        n_accepted = attempts = 0
        max_attempts = max(1000, 100 * n)
        while n_accepted < n:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "Failed to draw enough samples satisfying the parameter "
                    f"bounds after {attempts} attempts."
                )
            size = min(max(2 * (n - n_accepted), 16), max_attempts - attempts)
            candidates = _approximate(samples, size, distribution, confidence, rng)
            valid = candidates[specs.is_valid(candidates)]
            accepted.append(valid)
            n_accepted += len(valid)
            attempts += size
        draws = np.concatenate([*fixed, *accepted])[: len(fixed) + n]

        if implicit:
            result = dict(zip(specs.names, specs.complete(draws).T))
        else:
            result = dict(zip(self.explicit_names, draws.T))
        if fn_out is not None:
            fn_out = Path(fn_out)
            if fn_out.exists() and not overwrite:
                raise FileExistsError(f"File '{fn_out}' already exists.")
            save_yaml({k: v.tolist() for k, v in result.items()}, fn_out)
        return result

    # -- files ---------------------------------------------------------------

    def save(self, fn_out: PathLike) -> None:
        """Write everything to one ``results.pt`` file."""
        atomic_torch_save(
            {
                "format": FORMAT,
                "chain": torch.as_tensor(self.chain, dtype=torch.float32),
                "log_prob": torch.as_tensor(self.log_prob, dtype=torch.float32),
                "specs": self.specs.to_dict(),
                "prior": None if self.prior is None else self.prior.to_dicts(),
                "nuisances": list(self.nuisances),
                "qoi": self.qoi,
                "qoi_index": self.qoi_index,
                "qoi_log_likelihood": self.qoi_log_likelihood,
                "info": self.info,
            },
            Path(fn_out),
        )

    @classmethod
    def load(cls, fn_in: PathLike) -> Results:
        """Read a ``results.pt`` file."""
        fn_in = Path(fn_in)
        data = torch.load(fn_in, weights_only=False)
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ValueError(
                f"{fn_in} is not a BFF results file (format {FORMAT}); results "
                "of older BFF versions cannot be read. Rerun bff learn."
            )
        return cls(
            chain=data["chain"],
            log_prob=data["log_prob"],
            specs=Specs(data["specs"]),
            prior=None if data["prior"] is None else Priors.from_dicts(data["prior"]),
            nuisances=data["nuisances"],
            qoi=data["qoi"],
            qoi_index=data["qoi_index"],
            qoi_log_likelihood=data["qoi_log_likelihood"],
            info=data["info"],
        )

    # -- plots ---------------------------------------------------------------

    def plot_marginals(self, **kwargs):
        """See :func:`bff.plotting.plot_marginals`."""
        from ..plotting import plot_marginals

        return plot_marginals(self, **kwargs)

    def plot_qoi_marginals(self, **kwargs):
        """See :func:`bff.plotting.plot_qoi_marginals`."""
        from ..plotting import plot_qoi_marginals

        return plot_qoi_marginals(self, **kwargs)

    def plot_corner(self, **kwargs):
        """See :func:`bff.plotting.plot_corner`."""
        from ..plotting import plot_corner

        return plot_corner(self, **kwargs)


def _numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _approximate(
    samples: np.ndarray,
    n: int,
    distribution: str,
    confidence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``n`` draws from an approximation of the distribution of ``samples``."""
    if distribution == "empirical":
        return samples[rng.integers(0, len(samples), size=n)].copy()
    if distribution == "uniform":
        low = (1 - confidence) / 2
        lower, upper = np.quantile(samples, [low, 1 - low], axis=0)
        return rng.uniform(lower, upper, size=(n, samples.shape[1]))
    if len(samples) < 2:
        raise ValueError(f"{distribution} draws require at least two samples.")
    if distribution == "normal":
        mean = samples.mean(axis=0)
        cov = np.atleast_2d(np.cov(samples, rowvar=False))
        return rng.multivariate_normal(mean, cov, size=n)
    kde_input = samples[:, 0] if samples.shape[1] == 1 else samples.T
    return gaussian_kde(kde_input).resample(n, seed=rng).T.reshape(n, -1)
