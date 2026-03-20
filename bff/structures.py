from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import (
    Any, Callable,
    Dict, List, Mapping,
    Optional, Sequence,
    Tuple, Union, Self
)

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube
from scipy.stats import gaussian_kde

from .bayes.priors import Priors
from .mcmc.sampler import Sampler
from .mcmc.convergence import integrated_autocorr_time
from .io.utils import load_yaml, save_yaml, extract_train_dir


PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, torch.Tensor]


def lookup(filename, directories: Sequence[PathLike]) -> Path:
    """
    Find a file in a list of directories.
    """
    for d in directories:
        candidate_path = Path(d) / filename
        if candidate_path.exists():
            return candidate_path.resolve()
    raise FileNotFoundError(
        f"File '{filename}' not found in any of the directories: {directories}"
    )


class QoI:
    """Container for quantities of interest (QoI)."""

    def __init__(self, **kwargs):
        self.add(**kwargs)

    def add(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def names(self) -> list[str]:
        return list(self.__dict__)

    @property
    def observations(self) -> dict[str, int]:
        return {
            name: len(getattr(self, name))
            for name in self.names
        }

    def __repr__(self):
        return f"QoI: {', '.join(self.names)}"


@dataclass
class TrainSetInfo:
    """
    Information about the training set.
    """
    train_dir: PathLike
    specs: Optional[Dict[str, Any]]
    samples: Optional[Mapping[str, Any]]
    fn_topol: List[Path]
    fn_coord: List[Path]
    restraints: List[Dict[str, Any]]
    settings: Optional[Dict[str, Any]]

    @classmethod
    def from_dir(cls, train_dir: PathLike) -> "TrainSetInfo":
        train_dir = Path(train_dir).resolve()
        specs, samples, fn_topol, fn_coord, restraints = extract_train_dir(train_dir)
        return cls(train_dir, specs, samples, fn_topol, fn_coord, restraints, None)

    @property
    def hashes(self) -> List[str]:
        return list((self.samples or {}).keys())

    @property
    def inputs(self) -> np.ndarray:
        return np.array([sample["params"] for sample in self.samples.values()])

    @property
    def n_samples(self) -> int:
        return len(self.samples or {})

    @property
    def fn_trj(self) -> List[List[Path]]:
        return [
            [self.train_dir / trj for trj in s["fn_trj"]]
            for s in (self.samples or {}).values()
        ]

    def setup_settings(self, settings: Optional[Dict[str, Any]] = None) -> None:
        self.settings = settings or {}


@dataclass(slots=True)
class QoIDataset:
    name: str
    inputs: np.ndarray
    outputs: np.ndarray
    outputs_ref: np.ndarray
    n_observations: int = 0
    nuisance: float = 0.0
    settings: PathLike | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.name = self.name
        self.inputs = np.asarray(self.inputs, dtype=float)
        self.outputs = np.asarray(self.outputs, dtype=float)
        self.outputs_ref = np.asarray(self.outputs_ref, dtype=float)

        if self.inputs.shape[0] != self.outputs.shape[0]:
            raise ValueError(
                f"Number of input samples ({self.inputs.shape[0]}) does not match "
                f"number of output samples ({self.outputs.shape[0]})."
            )

        if self.outputs.shape[1] != self.outputs_ref.shape[0]:
            raise ValueError(
                f"Number of output features ({self.outputs.shape[1]}) does not match "
                f"number of reference output features ({self.outputs_ref.shape[0]})."
            )

    @property
    def n_samples(self) -> int:
        return self.inputs.shape[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "outputs_ref": self.outputs_ref,
            "n_observations": self.n_observations,
            "nuisance": self.nuisance,
            "settings": self.settings,
        }

    @classmethod
    def load(cls, fn_dataset: PathLike) -> Self:
        data = torch.load(Path(fn_dataset).resolve(), weights_only=False)
        return cls(**data)

    def write(self, fn_out: PathLike) -> None:
        torch.save(self.to_dict(), Path(fn_out).resolve())

    def __repr__(self):
        return (
            f"QoIData(n_samples={self.n_samples}, "
            f"n_features={self.outputs.shape[1]}, "
            f"n_observations={self.n_observations}, "
            f"nuisance={self.nuisance})"
        )


@dataclass(frozen=True)
class Bounds:
    """Container for parameter bounds.

    Attributes
    ----------
    by_name : Mapping[str, Tuple[float, float]]
        Dictionary mapping parameter names to their (lower, upper) bounds.

    Properties
    ----------
    names : np.ndarray
        Array of parameter names sorted alphabetically.
    array : np.ndarray
        2D array of shape (n_params, 2) containing the lower and upper bounds.
    lower : np.ndarray
        1D array of lower bounds for each parameter.
    upper : np.ndarray
        1D array of upper bounds for each parameter.
    n_params : int
        Number of parameters (i.e., number of entries in by_name).

    Methods
    -------
    without(name: str) -> "Bounds"
        Return a new Bounds instance excluding the specified parameter.
    """
    by_name: Mapping[str, Tuple[float, float]]

    def __post_init__(self):
        sorted_items = sorted(self.by_name.items())
        for name, (lower, upper) in sorted_items:
            if lower > upper:
                raise ValueError(
                    f"Lower bound {lower} is greater than upper bound {upper} "
                    f"for parameter '{name}'."
                )
            object.__setattr__(self, "_items", sorted_items)

    @property
    def names(self) -> np.ndarray:
        return np.array([name for name, _ in self._items], dtype=str)

    @property
    def array(self) -> np.ndarray:
        return np.array([bounds for _, bounds in self._items], dtype=float)

    @property
    def lower(self) -> np.ndarray:
        return self.array[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self.array[:, 1]

    @property
    def n_params(self) -> int:
        return len(self._items)

    def without(self, name: str) -> "Bounds":
        return Bounds({k: v for k, v in self.by_name.items() if k != name})


@dataclass(frozen=True)
class Specs:
    """Specifications for the parameter bounds and constraints.

    Parameters
    ----------
    source : dict or PathLike
        A dictionary containing the specifications
        or a path to a YAML file with the specifications.

    Attributes
    ----------
    data : dict
        The raw specifications data loaded from the source.
    mol_resname : str
        The residue name of the molecule (optional).
    bounds : Bounds
        The parameter bounds as a Bounds instance.
    total_charge : float
        The total charge constraint for the system.
    constraint_charge : float
        The charge that must be satisfied by the explicit parameters.
    implicit_atoms : list of str
        List of implicit atoms that are not explicitly parameterized but
        contribute to the charge constraint.

    Methods
    -------
    write(fn_out: PathLike) -> None
        Write the specifications data to a YAML file.

    Properties
    ----------
    implicit_param -> str
        Get the name of the implicit charge parameter.
    implicit_param_index -> int
        Get the index of the implicit charge parameter in the bounds.
    constraint_matrix -> np.ndarray
        Get the matrix of coefficients for the charge constraint, excluding
        implicit atoms.
    """
    source: InitVar[Union[Dict[str, Any], PathLike]]

    data: Dict[str, Any] = field(init=False)
    mol_resname: str = field(init=False)
    bounds: Bounds = field(init=False)
    total_charge: float = field(init=False)
    constraint_charge: float = field(init=False)
    implicit_atoms: List[str] = field(init=False)

    def __post_init__(self, source):
        if isinstance(source, dict):
            data = dict(source)
        elif isinstance(source, (str, Path)):
            data = load_yaml(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        data["bounds"] = {k: data["bounds"][k] for k in sorted(data["bounds"].keys())}

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "mol_resname", data.get("mol_resname", ""))
        object.__setattr__(self, "bounds", Bounds(data["bounds"]))
        object.__setattr__(self, "total_charge", data["total_charge"])
        object.__setattr__(self, "constraint_charge", data["constraint_charge"])
        object.__setattr__(self, "implicit_atoms", data["implicit_atoms"])

    def write(self, fn_out: PathLike):
        save_yaml(self.data, fn_out)

    @property
    def implicit_param(self):
        implicit_atoms = " ".join(self.implicit_atoms)
        return f"charge {implicit_atoms}"

    @property
    def implicit_param_index(self) -> int:
        mask = self.bounds.names == self.implicit_param
        if not np.any(mask):
            raise ValueError(
                f"Implicit parameter '{self.implicit_param}' "
                "not found in bounds."
            )
        elif np.sum(mask) > 1:
            raise ValueError(
                f"Multiple entries found for implicit parameter "
                f"'{self.implicit_param}' in bounds."
            )
        return np.argwhere(mask).flatten()[0]

    @property
    def constraint_matrix(self) -> np.ndarray:
        """Return charge constraint coefficients excluding implicit atoms."""
        coeffs: list[int] = []

        for param in self.bounds.names:
            atoms = param.split()[1:]
            if param.startswith("charge"):
                if atoms == self.implicit_atoms:
                    continue
                coeffs.append(len(atoms))
            else:
                coeffs.append(0)

        return np.asarray(coeffs, dtype=int)


class ChargeConstraint(Specs):
    """Callable class to check if parameter samples satisfy the charge constraint.

    Parameters
    ----------
    specs : dict or PathLike
        A dictionary containing the specifications or a path to a YAML file
        with the specifications.

    Methods
    -------
    __call__(x: np.ndarray) -> bool
        Check if the input parameter samples satisfy the explicit bounds and
        the implicit charge constraint.

    Properties
    ----------
    n_params -> int
        Get the number of explicit parameters (excluding the implicit charge).
    """
    def __init__(self, specs: Union[Dict[str, Any], PathLike]) -> None:
        super().__init__(specs)

        self.explicit_bounds = self.bounds.without(self.implicit_param).array
        self.implicit_bounds = self.bounds.by_name[self.implicit_param]

        if self.explicit_bounds.ndim != 2 or self.explicit_bounds.shape[1] != 2:
            raise ValueError("Bounds must be a 2D array with shape (n_params, 2).")

        if self.constraint_matrix.shape != (self.n_params, ):
            raise ValueError(
                f"Constraint matrix must have shape ({self.n_params},), "
                f"but got {self.constraint_matrix.shape}."
            )

    @property
    def n_params(self) -> int:
        return self.explicit_bounds.shape[0]

    @staticmethod
    def _to_2d_array(values: ArrayLike) -> np.ndarray:
        """Convert input values to a 2D numpy array
        with shape (n_samples, n_params)."""
        if isinstance(values, torch.Tensor):
            arr = values.detach().cpu().numpy()
        else:
            arr = np.asarray(values, dtype=float)
        return np.atleast_2d(arr)

    def __call__(self, x: np.ndarray) -> bool:
        x = self._to_2d_array(x)
        if x.shape[1] != self.n_params:
            raise ValueError(
                f"Input array must have shape (n_samples, {self.n_params}), "
                f"but got {x.shape}."
            )

        explicit_lower, explicit_upper = self.explicit_bounds.T
        implicit_lower, implicit_upper = self.implicit_bounds

        in_explicit_bounds = (
            (x >= explicit_lower) & (x <= explicit_upper)
        ).all(axis=1)

        explicit_total_charge = np.sum(x * self.constraint_matrix, axis=1)
        implicit_charge = self.constraint_charge - explicit_total_charge

        in_implicit_bounds = (
            (implicit_charge >= implicit_lower) & (implicit_charge <= implicit_upper)
        )

        return in_explicit_bounds & in_implicit_bounds


@dataclass
class InferenceResults:
    posterior: np.ndarray
    priors: Optional[Priors] = None
    parameter_names: Optional[List[str]] = None
    sample_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None

    _posterior_samples: Optional[np.ndarray] = field(
        default=None, init=False, repr=False)
    _tau: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @classmethod
    def load(
        cls,
        posterior: PathLike | Sampler | np.ndarray | torch.Tensor,
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
    def _load_checkpoint(fn_in: PathLike) -> dict[str, Any]:
        fn_in = Path(fn_in)
        if fn_in.suffix != ".pt":
            raise ValueError(
                f"Unsupported posterior file '{fn_in}'. "
                "InferenceResults accepts only .pt files."
            )
        return torch.load(fn_in, weights_only=False)

    @classmethod
    def _coerce_posterior(
        cls,
        posterior: PathLike | Sampler | np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        if isinstance(posterior, (np.ndarray, torch.Tensor)):
            posterior_array = cls._to_numpy(posterior)
        elif isinstance(posterior, Sampler):
            posterior_array = cls._to_numpy(posterior.chain)
        elif isinstance(posterior, (str, Path)):
            payload = cls._load_checkpoint(posterior)
            if "posterior" in payload:
                posterior_array = cls._to_numpy(payload["posterior"])
            else:
                raise KeyError(
                    f"Posterior file '{posterior}' does not contain posterior samples."
                )
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
        if self._tau is not None:
            return self._tau

        raw_posterior = torch.as_tensor(self.posterior, dtype=torch.float32)
        tau = integrated_autocorr_time(
            raw_posterior
        ).mean(dim=0).detach().cpu().numpy()

        self._tau = np.atleast_1d(tau.astype(float))
        return self._tau

    @property
    def samples(self) -> np.ndarray:
        if self._posterior_samples is None:
            raise ValueError(
                "No prepared samples available. "
                "Call prepare_samples() first."
            )
        return self._posterior_samples

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
    ) -> None:
        """Prepare flattened posterior samples for downstream analysis."""
        tau = self.autocorr_time
        discard = discard if discard is not None else int(2 * np.max(tau))
        thin = thin if thin is not None else max(1, int(0.5 * np.min(tau)))
        samples = self.posterior[discard::thin]
        if samples.size == 0:
            raise ValueError(
                "No posterior samples remain after applying discard/thin. "
                "Try smaller values."
            )
        samples = samples.reshape(-1, samples.shape[-1]).copy()
        transform_fn = sample_transform or self.sample_transform
        if transform_fn is not None:
            samples = np.asarray(transform_fn(samples), dtype=float)
        self._posterior_samples = samples

    @property
    def map_estimates(self) -> dict[str, float]:
        modes: List[float] = []
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
                save_yaml(dict(zip(self.labels, draws.T)), fn_out)
            else:
                raise ValueError("fn_out must end with .npy or .yaml")

        return draws


class RandomParamsGenerator:
    """Class to generate random parameter samples within specified bounds.

    Attributes
    ----------
    n_generated : int
        Total number of samples generated so far.

    Methods
    -------
    __call__(n)
        Advance the sampler to skip the next n samples.
    """

    def __init__(
        self,
        bounds: np.ndarray,
        constraint: Optional[Callable] = None
    ) -> None:

        assert bounds.ndim == 2 and bounds.shape[1] == 2, (
            "Bounds must be a 2D array with shape (n_params, 2)."
        )

        self.bounds = bounds
        n_dim = self.bounds.shape[0]
        self.n_generated = 0
        self.sampler = LatinHypercube(n_dim)
        self.constraint = constraint

    def __call__(self, n: int) -> None:
        """Advance the sampler to skip the next n samples."""
        assert n >= 0, "Number of samples to skip must be non-negative."
        self.sampler.fast_forward(n)
        self.n_generated += n

        # Generate samples and scale them
        rnd = self.sampler.random(n)
        samples = self._scale_to_bounds(rnd)

        # Validate samples against implicit charge constraints
        if self.constraint is not None:
            mask = self.constraint(samples)
            return samples[mask]
        return samples

    def _scale_to_bounds(self, unit_samples: np.ndarray) -> np.ndarray:
        """Scale unit samples to the specified bounds."""
        lower, upper = self.bounds.T
        return unit_samples * (upper - lower) + lower
