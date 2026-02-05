from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict, List, Mapping,
    Optional, Sequence,
    Tuple, Union, Set
)

import numpy as np
import torch
from torch.distributions import Normal, Uniform
from scipy.stats.qmc import LatinHypercube
from scipy.stats import gaussian_kde

from .bayes.utils import initialize_backend
from .io.utils import load_yaml, save_yaml, extract_train_dir
from .tools import sigmoid


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
    def names(self) -> List[str]:
        return list(self.__dict__)

    @property
    def observations(self) -> Dict[str, int]:
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
    train_dir: Path
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


class TrainData:
    def __init__(
        self,
        inputs: Union[np.ndarray, str, Path],
        outputs: Mapping[str, Union[np.ndarray, PathLike]],
        outputs_ref: Mapping[str, np.ndarray],
        observations: Union[Mapping[str, int], PathLike] = None,
        nuisances: Union[Mapping[str, float], PathLike] = None,
        settings: Union[Mapping[str, Any], PathLike] = None
    ) -> None:

        self.X = self._load_array(inputs)
        self.y = self._load_dict(outputs)
        self.y_ref = self._load_dict(outputs_ref)

        self.observations = self._load_yaml(observations)
        self.nuisances = self._load_yaml(nuisances) or {}
        self.settings = self._load_yaml(settings)

        assert len(self.X) == len(next(iter(self.y.values()))), (
            "Number of input samples does not match number of output samples."
        )

        assert self.y.keys() == self.y_ref.keys(), (
            "Output keys do not match reference output keys."
        )

    @staticmethod
    def _load_array(x: Union[np.ndarray, Sequence[Any], PathLike]) -> np.ndarray:
        if isinstance(x, (str, Path)):
            return np.load(x)
        else:
            return np.asarray(x)

    @staticmethod
    def _load_dict(d: dict) -> dict:
        loaded_dict = {}
        for qoi, data in d.items():
            if isinstance(data, (str, Path)):
                data = np.load(data)
                if data.ndim == 0:
                    data = data.item()
            loaded_dict[qoi] = np.asarray(data)

        return loaded_dict

    @staticmethod
    def _load_yaml(
        data: Mapping[str, Union[np.ndarray, PathLike]]
    ) -> Dict[str, np.ndarray]:
        return load_yaml(data) if isinstance(data, (str, Path)) else data

    @property
    def qoi_names(self) -> Set[str]:
        return set(self.y_ref.keys())

    @property
    def rdf_sigmoid_mean(self) -> np.ndarray:
        """Create a sigmoid function for concatenated RDFs."""
        n_bins = self.settings['rdf_kwargs']['n_bins']
        r0, r1 = self.settings['rdf_kwargs']['r_range']
        dr_half = (r1 - r0) / (2 * n_bins)
        r = np.linspace(r0, r1, n_bins, endpoint=False) + dr_half
        n_rdf = self.y_ref['rdf'].size // n_bins
        return np.tile(sigmoid(r), n_rdf)

    def write(self, fn_base: PathLike) -> None:
        fn_base = Path(fn_base).resolve()

        # Helper to save .npy files
        def save_npy(suffix: str, array: np.ndarray) -> None:
            np.save(fn_base.with_name(fn_base.name + suffix), array,
                    allow_pickle=False)

        # Save inputs
        save_npy("-train-inputs.npy", self.X)

        # Save outputs
        for qoi, data in self.y.items():
            save_npy(f"-train-{qoi}.npy", data)

        # Save reference data
        for qoi, data in self.y_ref.items():
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
            f"n_samples: {len(self.X)}\n"
            f"QoIs: {', '.join(self.qoi_names)}"
        )


class Bounds:
    def __init__(self, bounds: Mapping[str, Tuple[float, float]]) -> None:
        """
        bounds : dict
            Mapping parameter name -> (lower, upper)
        """
        self._bounds = bounds
        # store keys and values as arrays
        self._params = np.array(list(bounds.keys()))
        self._values = np.array(list(bounds.values()), dtype=float)

    @property
    def params(self) -> np.ndarray:
        return self._params

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def lower(self) -> np.ndarray:
        return self._values[:, 0]

    @property
    def upper(self) -> np.ndarray:
        return self._values[:, 1]

    @property
    def n(self):
        return len(self._bounds)

    def __repr__(self) -> str:
        return f"Bounds({self._bounds})"

    def __str__(self) -> str:
        lines = [f"{p}: [{l:.3f}, {u:.3f}]"
                 for p, (l, u) in self._bounds.items()]
        return "Bounds:\n" + "\n".join(lines)


class Specs:
    """Handles molecular specifications,
    ensuring consistency across multiple inputs."""

    def __init__(self, specs: Union[dict, PathLike]) -> None:
        self.data = self._load(specs)

        # Extract attributes
        self.mol_resname = self.data.get('mol_resname', "")
        self.implicit_atoms = self.data.get('implicit_atoms', [])
        self.total_charge = self.data.get('total_charge', 0.0)
        self.constraint_charge = self.data.get('constraint_charge', 0.0)
        self._bounds = Bounds(self.data['bounds'])

    @staticmethod
    def _load(spec: Union[dict, PathLike]) -> dict:
        """Load a specification from a file, dictionary, or path."""
        if isinstance(spec, dict):
            data = spec
        elif isinstance(spec, (str, Path)):
            data = load_yaml(str(spec))
        else:
            raise TypeError(f"Unsupported spec type: {type(spec)}")

        # Ensure bounds are sorted for consistency
        data['bounds'] = {k: data['bounds'][k] for k in sorted(data['bounds'])}
        return data

    def write(self, fn_out: PathLike) -> None:
        save_yaml(self.data, fn_out)

    @property
    def implicit_param(self):
        implicit_atoms = " ".join(self.implicit_atoms)
        return f"charge {implicit_atoms}"

    @property
    def implicit_param_pos(self):
        return np.argmax(self.bounds_explicit.params == self.implicit_param)

    @property
    def implicit_charge_bound(self):
        return self.bounds_explicit.values[self.implicit_param_pos]

    @property
    def total_charge_bounds(self):
        bounds = self.total_charge
        if isinstance(bounds, float):
            bounds = [bounds] * 2
        return bounds

    @property
    def n_params_implicit(self):
        """Number of parameters excluding the implicit charge."""
        return self.bounds_implicit.params.size

    @property
    def bounds_implicit(self):
        bounds_copy = self._bounds._bounds.copy()
        bounds_copy.pop(self.implicit_param, None)
        return Bounds(bounds_copy)

    @property
    def bounds_explicit(self):
        return self._bounds

    @property
    def constraint_matrix(self):
        """Determine constraint matrix."""

        return np.array(
            [len(p.split()[1:]) if p.startswith("charge") else 0
             for p in self.bounds_implicit.params],
            dtype=int,
        )


class ChargeConstraint:
    def __init__(
        self,
        bounds: np.ndarray,
        implicit_bound: tuple[float, float],
        constraint_matrix: np.ndarray,
        constraint_charge: float,
    ) -> None:
        self.bounds = np.asarray(bounds)
        self.implicit_bound = np.asarray(implicit_bound)
        self.constraint_matrix = np.asarray(constraint_matrix)
        self.constraint_charge = constraint_charge

    @property
    def n_params(self) -> int:
        return self.bounds.shape[0]

    def __call__(self, x: np.ndarray) -> bool:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x, dtype=float)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        lbe, ube = self.bounds.T
        lbi, ubi = self.implicit_bound

        valid_explicit = ((arr >= lbe) & (arr <= ube)).all(axis=1)

        q_explicit = np.sum(arr * self.constraint_matrix, axis=1)
        q_implicit = self.constraint_charge - q_explicit
        valid_implicit = (q_implicit >= lbi) & (q_implicit <= ubi)

        return valid_explicit & valid_implicit


class MCMCResults:

    DIST_REGISTRY = {
        'Normal': Normal,
        'Uniform': Uniform
    }

    def __init__(
        self,
        chain: str | object = None,
        priors: Union[str, Path, list] = None,
        tau: Union[str, Path, list, float] = None
    ) -> None:

        """Store and process Bayesian inference results."""
        self.chain = self._load_chain(chain) if chain else None
        self.priors = self._load_priors(priors) if priors else None
        self.tau = self._load_tau(tau)
        self._chain_samples = None

    def _load_chain(self, chain: Union[str, Path, object]) -> object:
        """Initialize the chain, handling HDF5 backend if necessary."""
        if isinstance(chain, (str, Path)):
            return initialize_backend(chain)
        return chain

    def _load_priors(self, priors: Union[str, PathLike]) -> List:
        if isinstance(priors, (str, Path)):
            priors_data = load_yaml(priors)

        elif isinstance(priors, list):
            priors_data = []
            for prior in priors:
                if isinstance(prior, tuple):
                    priors_data.append(prior)
                elif isinstance(prior, torch.distributions.Distribution):
                    dist_type = type(prior).__name__.lower()
                    if dist_type == 'normal':
                        arg1, arg2 = float(prior.mean), float(prior.scale)
                    elif dist_type == 'uniform':
                        arg1, arg2 = float(prior.low), float(prior.high)
                    else:
                        raise ValueError(
                            f"Unsupported distribution type: {dist_type}"
                        )
                    priors_data.append((dist_type, (arg1, arg2)))

        priors_list = []
        for dist_type, (arg1, arg2) in priors_data:
            if dist_type == 'normal':
                distribution = Normal(loc=arg1, scale=arg2, validate_args=False)
            elif dist_type == 'uniform':
                distribution = Uniform(low=arg1, high=arg2, validate_args=False)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")

            priors_list.append(distribution)

        self.priors_data = priors_data

        return priors_list

    def _load_tau(self, tau: Union[str, Path, list, float, int]) -> np.ndarray:
        """Load tau array from file, list, or numeric array."""
        if isinstance(tau, (str, Path)):
            tau = np.load(str(tau))
        elif isinstance(tau, (list, float, int)):
            tau = np.asarray(tau)
        else:
            try:
                tau = self.chain.get_autocorr_time(tol=0)
            except AttributeError:
                tau = np.ones(self.chain.get_chain().shape[1])
        return np.atleast_1d(tau)

    def chain_samples(self, discard: int = None, stride: int = None) -> np.ndarray:
        """
        Retrieve MCMC samples with optional filtering.
        """
        discard = discard or int(2 * np.max(self.tau))
        stride = stride or int(0.5 * np.min(self.tau))
        samples = self.chain.get_chain(discard=discard, flat=True, thin=stride)

        self._chain_samples = samples
        return samples

    def write_priors(self, fn_out: PathLike) -> None:
        """Save priors to a YAML file."""
        save_yaml(self.priors_data, fn_out)

    def write_tau(self, fn_out: PathLike) -> None:
        """Save tau array to a .npy file."""
        np.save(fn_out, self.tau)


class InferenceResults(MCMCResults, Specs):
    def __init__(
        self,
        chain: Union[str, Path, object] = None,
        priors: Union[str, Path, list] = None,
        tau: Union[str, Path, list, float, int] = None,
        specs: Union[str, Path, dict] = None,
        qoi: List[str] = None
    ) -> None:
        """Store and process learning results."""
        MCMCResults.__init__(self, chain, priors, tau)
        Specs.__init__(self, specs)

        self._chain = None
        self.QoIs = qoi or []

    @property
    def chain_implicit_(self) -> np.ndarray:
        return self._chain

    @property
    def chain_(self) -> np.ndarray:
        """Get the explicit parameters, computing them if necessary."""
        return self._compute_explicit_params()

    @property
    def labels_implicit_(self) -> list:
        """Generate labels for the parameters."""
        return self._get_labels('implicit')

    @property
    def labels_(self) -> list:
        """Generate labels for the explicit parameters."""
        return self._get_labels('explicit')

    @property
    def map(self):
        """Return maximum a posteriori (MAP) estimates for all parameters."""
        param_modes, nuisance_modes = [], []

        for param, label in zip(self.chain_.T, self.labels_):
            x = np.linspace(param.min(), param.max(), 1000)
            density = gaussian_kde(param)
            mode = np.round(x[np.argmax(density(x))], 3)
            if "sigma_" in label:
                nuisance_modes.append(mode)
            else:
                param_modes.append(mode)

        map_all = np.concatenate((param_modes, nuisance_modes))

        return dict(zip(self.labels_, map_all))

    def _get_labels(self, kind: str) -> list:
        if kind == 'implicit':
            param_labels = self.bounds_implicit.params.tolist()
        elif kind == 'explicit':
            param_labels = self.bounds_explicit.params.tolist()
        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'implicit' or 'explicit'.")

        nuisance_labels = [
            f'$\\sigma_{{\\mathrm{{{qoi}}}}}$'
            for qoi in self.QoIs
        ]

        return param_labels + nuisance_labels

    def quantiles(self, confidence: float = 0.95) -> np.ndarray:
        """
        Compute quantiles of the samples for a given confidence level.
        """
        q_lo = (1 - confidence) / 2
        q_hi = 1 - q_lo
        return np.quantile(self.chain_, [q_lo, 0.5, q_hi], axis=0)

    def get_chain(
        self,
        discard: int = None,
        stride: int = None,
        remove_defects: bool = True,
        remove_sigma_outliers: bool = True
    ) -> None:
        """
        Retrieve MCMC samples with optional filtering.
        """

        n = self.n_params_implicit
        samples = self.chain_samples(discard, stride).copy()
        samples[:, n:] = np.exp(samples[:, n:])

        if remove_defects:
            bounds = self.bounds_implicit.values
            data = samples[:, :self.n_params_implicit]
            mask = np.all(
                np.logical_and(data >= bounds[:, 0], data <= bounds[:, 1]),
                axis=1
            )
            samples = samples[mask]

        if remove_sigma_outliers:
            # Remove outliers based on nuisance parameters
            nuisance_params = samples[:, self.n_params_implicit:]
            treshold = np.quantile(nuisance_params, 0.99, axis=0)
            mask = np.all(nuisance_params <= treshold, axis=1)
            samples = samples[mask]

        self._chain = samples

    def sample_posterior(
        self,
        n_samples: int = 10,
        distribution: str = 'normal',
        confidence: float = 0.9,
        fn_out: str = None
    ) -> np.ndarray:
        """
        Draw samples from the posterior using
        either uniform or Laplace distribution.
        """
        lower, upper = (1 - confidence) / 2, 1 - (1 - confidence) / 2
        samples = self.chain_[:, :self.bounds_explicit.n]
        confint = np.quantile(samples, [lower, upper], axis=0)

        if distribution == 'normal':
            mean = np.mean(samples, axis=0)
            cov = np.cov(samples, rowvar=False)
        elif distribution != 'uniform':
            raise ValueError(
                (
                    f'Unknown distribution "{distribution}". '
                    'Options are "uniform" or "normal".'
                )
            )

        if distribution == "normal":
            if cov.size == 1:
                samples = np.random.normal(mean, cov, size=n_samples)
            else:
                samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        elif distribution == "uniform":
            uniform_range = np.diff(confint, axis=0).ravel()
            n = len(uniform_range)
            samples = np.random.rand(n_samples, n) * uniform_range + confint[0]

        if fn_out:
            if fn_out.endswith('.npy'):
                np.save(fn_out, samples)
            elif fn_out.endswith('.yaml'):
                param_names = self.bounds_explicit.params.tolist()
                samples_dict = dict(zip(param_names, samples.T))
                save_yaml(samples_dict, fn_out)
            else:
                raise ValueError('fn_out must end with .npy or .yaml')

        return samples

    def _compute_explicit_params(self) -> np.ndarray:
        """
        Compute the explicit parameters from the implicit parameters.
        """

        params = self.chain_implicit_[:, :self.n_params_implicit]
        n_implicit = len(self.implicit_atoms)
        q_explicit = np.sum(params * self.constraint_matrix, axis=1)
        q_implicit = (self.total_charge - q_explicit) / n_implicit
        return np.insert(
            self.chain_implicit_, self.implicit_param_pos, q_implicit, axis=1)


class RandomParamsGenerator(Specs):
    """Class to generate random parameter samples within specified bounds.

    Attributes
    ----------
    settings : object
        Settings object containing the bounds and constraints.
    sampler : object
        LatinHypercube sampler.

    Methods
    -------
    __call__(n)
        Advance the sampler to skip the next n samples.
    generate(n, assign_hash=False)
        Generate random parameter samples within specified bounds.
    """

    def __init__(self, specs: Union[str, Path, dict] = None) -> None:
        super().__init__(specs)
        if (
            not hasattr(self, 'bounds_implicit') or
            not hasattr(self.bounds_implicit, '_bounds') or
            not hasattr(self.bounds_implicit, 'values')
        ):
            raise AttributeError(
                "The 'bounds_implicit' attribute is not properly initialized "
                "or missing required attributes ('bounds', 'values')."
            )

        n_dim = len(self.bounds_implicit._bounds)
        self.n_samples = 0
        self.sampler = LatinHypercube(n_dim)
        self.lbe, self.ube = self.bounds_implicit.values.T
        self.lbi, self.ubi = getattr(
            self, 'implicit_param_bounds', (None, None)
        )
        if self.lbi is None or self.ubi is None:
            raise AttributeError(
                "The 'implicit_param_bounds' attribute is not properly "
                "initialized."
            )

        if len(self.constraint_matrix) != len(self.bounds_implicit._bounds):
            raise ValueError(
                "The constraint matrix must match the\
                    number of implicit parameters."
            )

    def __call__(self, n: int) -> None:
        """Advance the sampler to skip the next n samples."""
        self.sampler.fast_forward(n)
        self.n_samples += n

    def generate(self, n: int) -> np.ndarray:
        """
        Generate random parameter samples within specified bounds.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        assign_hash : bool, optional
            Whether to assign a hash to the generated samples.

        Returns
        -------
        tuple or np.ndarray
            If assign_hash is True, returns a tuple (hash, samples).
            Otherwise, returns the generated samples.
        """

        # Generate samples and scale them
        samples = self.sampler.random(n)
        scaled_samples = samples * (self.ube - self.lbe) + self.lbe

        # Validate samples against implicit charge constraints
        q_explicit = np.sum(scaled_samples * self.constraint_matrix, axis=1)
        q_implicit = (self.total_charge - q_explicit) / self.implicit_param_count
        mask = (self.lbi <= q_implicit) & (q_implicit <= self.ubi)

        # Filtered samples based on the mask
        filtered_samples = scaled_samples[mask]

        return filtered_samples
