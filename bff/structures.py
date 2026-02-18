from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import (
    Any, Callable,
    Dict, List, Mapping,
    Optional, Sequence,
    Tuple, Union, Set
)

import numpy as np
import torch
from emcee.ensemble import EnsembleSampler
from torch.distributions import Normal, Uniform
from scipy.stats.qmc import LatinHypercube
from scipy.stats import gaussian_kde

from .bayes.utils import initialize_backend
from .io.utils import load_yaml, save_yaml, extract_train_dir
from .tools import sigmoid


PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, torch.Tensor]
PriorSpec = tuple[str, tuple[float, float]]


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


@dataclass
class TrainData:
    """Data structure for training surrogate models.

    Parameters
    ----------
    inputs : np.ndarray or PathLike
        Array of input parameter samples or path to a .npy file containing it.
    outputs : dict of str to np.ndarray or PathLike
        Dictionary mapping QoI names to arrays of predicted values
        or paths to .npy files.
    outputs_ref : dict of str to np.ndarray or PathLike
        Dictionary mapping QoI names to arrays of reference values
        or paths to .npy files.
    observations : dict of str to int or PathLike, optional (default None)
        Dictionary mapping QoI names to the number of observations
        or path to a YAML file containing it.
    nuisances : dict of str to float or PathLike, optional (default None)
        Dictionary mapping QoI names to nuisance parameters
        (e.g., noise levels) or path to a YAML file containing it.
    settings : dict or PathLike, optional (default None)
    """

    inputs: Union[np.ndarray, PathLike]
    outputs: Union[Dict[str, np.ndarray], PathLike]
    outputs_ref: Union[Dict[str, np.ndarray], PathLike]
    observations: Optional[Union[Dict[str, int], PathLike]] = None
    nuisances: Optional[Union[Dict[str, float], PathLike]] = None
    settings: Optional[Union[Dict[str, Any], PathLike]] = None

    def __post_init__(self) -> None:
        self.inputs = self._load_array(self.inputs)
        self.outputs = self._load_dict(self.outputs)
        self.outputs_ref = self._load_dict(self.outputs_ref)
        self.observations = self._load_yaml(self.observations) or {}
        self.nuisances = self._load_yaml(self.nuisances) or {}
        self.settings = self._load_yaml(self.settings) or {}

        if len(self.inputs) != len(next(iter(self.outputs.values()))):
            raise ValueError(
                "Number of input samples does not match number of output samples.")
        if self.outputs.keys() != self.outputs_ref.keys():
            raise ValueError("Output keys do not match reference output keys.")

    @staticmethod
    def _load_array(x: Union[np.ndarray, Sequence[Any], PathLike]) -> np.ndarray:
        """Load an array from a .npy file or convert a sequence to a numpy array."""
        return np.load(x) if isinstance(x, (str, Path)) else np.asarray(x)

    @staticmethod
    def _load_dict(d: dict) -> dict:
        """Load a dictionary of arrays from .npy files or
        convert values to numpy arrays."""
        loaded: dict[str, np.ndarray] = {}
        for qoi_name, data in d.items():
            arr = np.load(data) if isinstance(data, (str, Path)) else np.asarray(data)
            loaded[qoi_name] = np.asarray(arr.item() if arr.ndim == 0 else arr)

        return loaded

    @staticmethod
    def _load_yaml(
        data: Mapping[str, Union[np.ndarray, PathLike]]
    ) -> Dict[str, np.ndarray]:
        return load_yaml(data) if isinstance(data, (str, Path)) else data

    @property
    def qoi_names(self) -> Set[str]:
        """Get the set of QoI names from the outputs reference data."""
        return set(self.outputs_ref.keys())

    @property
    def rdf_sigmoid_mean(self) -> np.ndarray:
        """Create a sigmoid function for concatenated RDFs."""
        n_bins = self.settings['rdf_kwargs']['n_bins']
        r0, r1 = self.settings['rdf_kwargs']['r_range']
        dr_half = (r1 - r0) / (2 * n_bins)
        r = np.linspace(r0, r1, n_bins, endpoint=False) + dr_half
        n_rdf = self.outputs_ref['rdf'].size // n_bins
        return np.tile(sigmoid(r), n_rdf)

    def write(self, fn_base: PathLike) -> None:
        """Write the training data to disk with a common base filename.

        Parameters
        ----------
        fn_base : PathLike
            Base filename (without extension) for saving the training data.
        """
        fn_base = Path(fn_base).resolve()

        # Helper to save .npy files
        def save_npy(suffix: str, array: np.ndarray) -> None:
            np.save(fn_base.with_name(fn_base.name + suffix), array,
                    allow_pickle=False)

        # Save inputs
        save_npy("-train-inputs.npy", self.inputs)

        # Save outputs
        for qoi, data in self.outputs.items():
            save_npy(f"-train-{qoi}.npy", data)

        # Save reference data
        for qoi, data in self.outputs_ref.items():
            save_npy(f"-ref-{qoi}.npy", data)

        # Save YAML files
        save_yaml(self.settings or {},
                  fn_base.with_name(fn_base.name + "-settings.yaml"))
        save_yaml(self.observations or {},
                  fn_base.with_name(fn_base.name + "-observations.yaml"))
        save_yaml(self.nuisances or {},
                  fn_base.with_name(fn_base.name + "-nuisances.yaml"))

    def __repr__(self):
        return (
            "TrainData\n"
            f"n_samples: {len(self.inputs)}\n"
            f"QoIs: {', '.join(self.qoi_names)}"
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
            if not param.startswith("charge"):
                continue

            atoms = param.split()[1:]
            if atoms == self.implicit_atoms:
                continue

            coeffs.append(len(atoms))

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


class MCMCResults:
    """Container for MCMC results, including the chain,
    priors, and autocorrelation times.

    Parameters
    ----------
    chain_src : object or PathLike, optional
        An object representing the MCMC chain (e.g., an emcee backend)
        or a path to a file containing the chain data. Defaults to None.
    priors_src : list or PathLike, optional
        A list of prior specifications
        (e.g., tuples of distribution type and parameters)
        or a path to a YAML file containing the priors. Defaults to None.
    tau_src : array-like or PathLike, optional
        An array of autocorrelation times or a path to a file containing the tau data.
        Defaults to None.

    Attributes
    ----------
    chain : EnsembleSampler or None
        The loaded MCMC chain object.
    priors_data : list
        The normalized priors data loaded from the source.
    priors : list
        The list of torch.distributions objects representing the priors.
    tau : np.ndarray
        The array of autocorrelation times for each parameter.

    Methods
    -------
    get_chain(discard: int, thin: int) -> np.ndarray
        Retrieve the MCMC samples from the chain, applying burn-in and thinning.
    write_priors(fn_out: PathLike) -> None
        Write the priors data to a YAML file.
    write_tau(fn_out: PathLike) -> None
        Write the autocorrelation times to a .npy file.
    """

    def __init__(
        self,
        chain_src: Optional[Union[PathLike, EnsembleSampler]] = None,
        priors_src: Optional[Union[PathLike, list]] = None,
        tau_src: Optional[Union[PathLike, list, float, int, np.ndarray]] = None,
    ) -> None:
        self.chain = self._load_chain(chain_src)
        if priors_src is not None:
            self.priors_data = self._normalize_priors(priors_src)
        else:
            self.priors_data = []
        self.priors = self._build_priors(self.priors_data) if self.priors_data else None
        self.tau = self._load_tau(tau_src)
        self._chain_samples = None

    @staticmethod
    def _load_chain(
        chain_src: Optional[Union[PathLike, EnsembleSampler]]
    ) -> EnsembleSampler | None:
        """Load the MCMC chain from a backend file
        or return the provided chain object."""
        if chain_src is None:
            return None
        if isinstance(chain_src, (str, Path)):
            return initialize_backend(chain_src)
        return chain_src

    def _normalize_priors(self, priors_src: Union[PathLike, list]) -> list[PriorSpec]:
        """Normalize the priors data from the source into a consistent format."""
        if isinstance(priors_src, (str, Path)):
            raw = load_yaml(priors_src)
        else:
            raw = priors_src
        normalized: list[PriorSpec] = []

        for item in raw:
            if isinstance(item, (tuple, list)):
                name, (a, b) = item
                normalized.append((str(name).lower(), (float(a), float(b))))
            elif isinstance(item, torch.distributions.Distribution):
                name = type(item).__name__.lower()
                if name == "normal":
                    normalized.append((name, (float(item.mean), float(item.scale))))
                elif name == "uniform":
                    normalized.append((name, (float(item.low), float(item.high))))
                else:
                    raise ValueError(f"Unsupported distribution type: {name}")
            else:
                raise TypeError(f"Unsupported prior entry type: {type(item)}")
        return normalized

    def _build_priors(self, priors_data: list[PriorSpec]) -> list:
        """Build a list of torch.distributions objects
        based on the normalized priors data."""
        priors = []
        for name, (a, b) in priors_data:
            if name == "normal":
                priors.append(Normal(loc=a, scale=b, validate_args=False))
            elif name == "uniform":
                priors.append(Uniform(low=a, high=b, validate_args=False))
            else:
                raise ValueError(f"Unsupported distribution type: {name}")
        return priors

    def _load_tau(
        self, tau_src: Optional[Union[PathLike, list, float, int, np.ndarray]]
    ) -> np.ndarray:
        """Load the autocorrelation times from a file or convert the provided data"""
        if isinstance(tau_src, (str, Path)):
            tau = np.load(str(tau_src))
        elif isinstance(tau_src, (list, float, int, np.ndarray)):
            tau = np.asarray(tau_src, dtype=float)
        elif self.chain is not None:
            try:
                tau = self.chain.get_autocorr_time(tol=0)
            except AttributeError:
                tau = np.ones(self.chain.get_chain().shape[1], dtype=float)
        else:
            tau = np.array([1.0], dtype=float)
        return np.atleast_1d(tau)

    def get_chain(
        self,
        discard: Optional[int] = None,
        thin: Optional[int] = None
    ) -> np.ndarray:
        if self.chain is None:
            raise ValueError("No chain backend available.")
        discard = discard if discard is not None else int(2 * np.max(self.tau))
        thin = thin if thin is not None else max(1, int(0.5 * np.min(self.tau)))
        samples = self.chain.get_chain(discard=discard, flat=True, thin=thin)
        self._chain_samples = samples
        return samples

    def write_priors(self, fn_out: PathLike) -> None:
        save_yaml(self.priors_data, fn_out)

    def write_tau(self, fn_out: PathLike) -> None:
        np.save(fn_out, self.tau)


@dataclass
class InferenceResults:
    """Container for inference results,
    including MCMC results, specifications, and QoI names.

    Parameters
    ----------
    mcmc : MCMCResults
        An instance of MCMCResults containing the MCMC sampling results.
    specs : Specs
        An instance of Specs containing the parameter specifications and constraints.
    qoi_names : list of str, optional
        A list of QoI names corresponding to the
        nuisance parameters in the MCMC samples.
        Defaults to an empty list.

    Attributes
    ----------
    mcmc : MCMCResults
        The MCMC results containing the chain, priors, and autocorrelation times.
    specs : Specs
        The specifications for parameter bounds and constraints.
    qoi_names : list of str
        The names of the QoI corresponding to the nuisance parameters.
    _posterior_samples_explicit : np.ndarray or None
        A private attribute to store the explicit posterior samples after preparation.
        Initialized to None and populated by the prepare_samples() method.

    Methods
    -------
    from_sources(chain, priors, tau, specs, qoi) -> InferenceResults
        A class method to create an InferenceResults instance from various sources.
    prepare_samples(discard, thin, remove_out_of_bounds, remove_sigma_outliers) -> None
        Prepare the posterior samples by applying burn-in, thinning, and optional
        filtering of out-of-bounds and high-sigma samples.
    sample_posterior(n_samples, distribution, confidence, fn_out) -> np.ndarray
        Sample from the posterior distribution using
        either a normal or uniform distribution
        based on the confidence interval of the parameter samples.

    Properties
    ----------
    posterior_samples_explicit -> np.ndarray
        Accesses the explicit posterior samples after preparation.
    posterior_samples -> np.ndarray
        Accesses the full posterior samples, including implicit charge.
    labels -> list of str
        Generates the labels for the parameters and nuisance parameters.
    map_estimates -> dict of str to float
        Computes the MAP estimates for each parameter and nuisance parameter.
    quantiles(confidence) -> np.ndarray
    """
    mcmc: MCMCResults
    specs: Specs
    qoi_names: List[str] = field(default_factory=list)

    _posterior_samples_explicit: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_sources(
        cls,
        chain: Optional[Union[PathLike, EnsembleSampler]] = None,
        priors: Optional[Union[PathLike, list]] = None,
        tau: Optional[Union[PathLike, list, float, int, np.ndarray]] = None,
        specs: Optional[Union[PathLike, dict]] = None,
        qoi: Optional[List[str]] = None,
    ) -> "InferenceResults":
        if specs is None:
            raise ValueError("specs source is required.")
        return cls(
            mcmc=MCMCResults(chain_src=chain, priors_src=priors, tau_src=tau),
            specs=Specs(specs),
            qoi_names=qoi or [],
        )

    @property
    def posterior_samples_explicit(self) -> np.ndarray:
        if self._posterior_samples_explicit is None:
            raise ValueError("Call prepare_samples() first.")
        return self._posterior_samples_explicit

    @property
    def posterior_samples(self) -> np.ndarray:
        return self._insert_implicit_charge(self.posterior_samples_explicit)

    @property
    def labels(self) -> List[str]:
        nuisance = [f"$\\sigma_{{\\mathrm{{{qoi}}}}}$" for qoi in self.qoi_names]
        return self.specs.bounds.names.tolist() + nuisance

    def prepare_samples(
        self,
        discard: Optional[int] = None,
        thin: Optional[int] = None,
        remove_out_of_bounds: bool = True,
        remove_sigma_outliers: bool = True,
    ) -> None:
        samples = self.mcmc.get_chain(discard=discard, thin=thin).copy()
        n_explicit = self.specs.bounds.without(self.specs.implicit_param).n_params

        samples[:, n_explicit:] = np.exp(samples[:, n_explicit:])

        if remove_out_of_bounds:
            explicit = samples[:, :n_explicit]
            b = self.specs.bounds.without(self.specs.implicit_param)
            mask = np.all((explicit >= b.lower) & (explicit <= b.upper), axis=1)
            samples = samples[mask]

        if remove_sigma_outliers and samples.shape[1] > n_explicit:
            nuisance = samples[:, n_explicit:]
            threshold = np.quantile(nuisance, 0.99, axis=0)
            mask = np.all(nuisance <= threshold, axis=1)
            samples = samples[mask]

        self._posterior_samples_explicit = samples

    @property
    def map_estimates(self) -> dict[str, float]:
        modes: List[float] = []
        for col in self.posterior_samples.T:
            grid = np.linspace(col.min(), col.max(), 1000)
            density = gaussian_kde(col)
            modes.append(float(np.round(grid[np.argmax(density(grid))], 3)))
        return dict(zip(self.labels, modes))

    @property
    def quantiles(self, confidence: float = 0.95) -> np.ndarray:
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        return np.quantile(self.posterior_samples, [q_low, 0.5, q_high], axis=0)

    def sample_posterior(
        self,
        n_samples: int = 10,
        distribution: str = "normal",
        confidence: float = 0.9,
        fn_out: Optional[str] = None,
        overwrite: bool = False,
    ) -> np.ndarray:
        param_samples = self.posterior_samples[:, :self.specs.bounds.n_params]
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        confint = np.quantile(param_samples, [q_low, q_high], axis=0)

        if distribution == "normal":
            mean = np.mean(param_samples, axis=0)
            cov = np.cov(param_samples, rowvar=False)
            draws = (
                np.random.normal(mean, cov, size=n_samples)
                if np.ndim(cov) == 0
                else np.random.multivariate_normal(mean, cov, size=n_samples)
            )
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
                param_names = self.specs.bounds.names.tolist()
                save_yaml(dict(zip(param_names, draws.T)), fn_out)
            else:
                raise ValueError("fn_out must end with .npy or .yaml")

        return draws

    def _insert_implicit_charge(self, samples: np.ndarray) -> np.ndarray:
        explicit_bounds = self.specs.bounds.without(self.specs.implicit_param)
        n_implicit_atoms = len(self.specs.implicit_atoms)
        constraint_charge = self.specs.constraint_charge
        insert_at = self.specs.implicit_param_index

        n_explicit_params = explicit_bounds.n_params
        explicit_charge = samples[:, :n_explicit_params] @ self.specs.constraint_matrix
        implicit_charge = (constraint_charge - explicit_charge) / n_implicit_atoms

        return np.insert(samples, insert_at, implicit_charge, axis=1)


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
