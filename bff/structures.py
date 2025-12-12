from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal, Uniform
from scipy.stats.qmc import LatinHypercube
from scipy.stats import gaussian_kde

from .bayes.utils import initialize_backend
from .io.utils import load_yaml, save_yaml, extract_train_dir
from .tools import sigmoid


def lookup(filename, directories):
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

    # def flatten

    def __repr__(self):
        return f"QoI: {', '.join(self.names)}"


@dataclass
class TrainSetInfo:
    """
    Information about the training set.
    """
    train_dir: Path
    specs: dict | None
    samples: dict | None
    fn_topol: list[Path]
    fn_coord: list[Path]
    restraints: list[dict]
    settings: dict | None

    @classmethod
    def from_dir(cls, train_dir: str | Path):
        train_dir = Path(train_dir).resolve()
        specs, samples, fn_topol, fn_coord, restraints = extract_train_dir(train_dir)
        return cls(train_dir, specs, samples, fn_topol, fn_coord, restraints, None)

    @property
    def hashes(self) -> list[str]:
        return list(self.samples.keys() or {})

    @property
    def inputs(self) -> dict[str, dict]:
        return np.array([sample["params"] for sample in self.samples.values()])

    @property
    def n_samples(self) -> int:
        return len(self.samples or {})

    @property
    def fn_trj(self) -> list[list[Path]]:
        return [
            [self.train_dir / trj for trj in s["fn_trj"]]
            for s in (self.samples or {}).values()
        ]

    def setup_settings(self, settings: dict | None):
        self.settings = settings or {}


class TrainData:
    def __init__(
        self,
        inputs: np.ndarray | list | str | Path,
        outputs: dict[str, np.ndarray],
        outputs_ref: dict[str, np.ndarray],
        observations: dict[str, int] | str | Path,
        nuisances: dict[str, float] | str | Path = None,
        settings: dict | str | Path = None
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
    def _load_array(x: np.ndarray | str | Path) -> np.ndarray:
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
    def _load_yaml(data: str | Path | dict) -> dict:
        return load_yaml(data) if isinstance(data, (str, Path)) else data

    @property
    def qoi_names(self) -> set[str]:
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

    def write(self, fn_base: str | Path) -> None:
        fn_base = Path(fn_base).resolve()

        # Helper to save .npy files
        def save_npy(suffix: str, array):
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
    def __init__(self, bounds: dict) -> None:
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

    def __repr__(self) -> str:
        return f"Bounds({self._bounds})"

    def __str__(self) -> str:
        lines = [f"{p}: [{l:.3f}, {u:.3f}]"
                 for p, (l, u) in self._bounds.items()]
        return "Bounds:\n" + "\n".join(lines)


class Specs:
    """Handles molecular specifications,
    ensuring consistency across multiple inputs."""

    def __init__(self, specs: str | dict | Path) -> None:
        self.data = self._load(specs)

        # Extract attributes
        self.atomtype_counts = self.data.get('atomtype_counts', {})
        self.mol_resname = self.data.get('mol_resname', "")
        self.implicit_atomtype = self.data.get('implicit_atomtype', "")
        self.total_charge = self.data.get('total_charge', 0.0)
        self._bounds = Bounds(self.data['bounds'])

    @staticmethod
    def _load(spec: str | dict | Path) -> dict:
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

    def save(self, fn_out: str):
        save_yaml(self.data, fn_out)

    @property
    def atomtypes(self):
        return list(self.atomtype_counts.keys())

    @property
    def implicit_param(self):
        return f"charge {self.implicit_atomtype}"

    @property
    def implicit_param_pos(self):
        return np.argmax(self.bounds_explicit.params == self.implicit_param)

    @property
    def implicit_param_count(self):
        return self.atomtype_counts[self.implicit_atomtype]

    @property
    def implicit_param_bounds(self):
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
    def n_params_explicit(self):
        """Number of parameters excluding the implicit charge."""
        return self.bounds_explicit.params.size

    @property
    def bounds_implicit(self):
        bounds_copy = self._bounds._bounds.copy()
        bounds_copy.pop(self.implicit_param, None)
        return Bounds(bounds_copy)

    @property
    def bounds_explicit(self):
        return self._bounds

    @property
    def varied_param_names(self):
        mask = self.bounds_explicit.params != self.implicit_param
        return self.bounds_explicit.params[mask]

    @property
    def constraint_matrix(self):
        """Determine constraint matrix and create a linear constraint."""
        return np.array([
            self.atomtype_counts[p.split()[1]] if 'charge' in p else 0
            for p in self.bounds_implicit.params
        ])

    def is_valid(self, params):
        if isinstance(params, list):
            params = np.array(params)
        elif isinstance(params, torch.Tensor):
            params = params.cpu().numpy()

        if params.ndim == 1:
            params = params[np.newaxis, :]

        lbe, ube = self.bounds_implicit.values.T
        lbi, ubi = self.implicit_param_bounds

        valid_explicit = ((params > lbe) & (params < ube)).all(axis=1)

        q_explicit = np.sum(params * self.constraint_matrix, axis=1)
        q_implicit = self.total_charge - q_explicit
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
        priors: str | Path | list = None,
        tau: str | Path | list | float = None
    ) -> None:

        """Store and process Bayesian inference results."""
        self.chain = self.load_chain(chain) if chain else None
        self.priors_specs = self.load_priors(priors) if priors else None
        self.tau = self.load_tau(tau)
        self._chain_samples = None

    def load_chain(self, chain):
        """Initialize the chain, handling HDF5 backend if necessary."""
        if isinstance(chain, (str, Path)):
            return initialize_backend(chain)
        return chain

    def load_priors(self, priors: str | Path | dict) -> dict:
        """Load priors from YAML or dictionary of distributions."""
        if isinstance(priors, (str, Path)):
            return load_yaml(str(priors))
        if isinstance(priors, dict):
            priors_dict = {}
            for name, p in priors.items():
                dist_type = type(p).__name__
                if dist_type == 'Normal':
                    a, b = float(p.mean), float(p.scale)
                elif dist_type == 'Uniform':
                    a, b = float(p.low), float(p.high)
                else:
                    raise ValueError(
                        f"Unsupported distribution type: {dist_type}. "
                        "Only 'Normal' and 'Uniform' are supported."
                    )
                priors_dict[f"{name} {dist_type}"] = [a, b]

            return priors_dict

        raise TypeError("Priors must be str, Path or dictionary.")

    def load_tau(self, tau):
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

    @property
    def priors(self):
        """Reconstruct distributions from specs."""
        if not self.priors_specs:
            return {}
        priors = {}
        for name, (arg1, arg2) in self.priors_specs.items():
            dist_name = name.split()[-1]
            dist_cls = self.DIST_REGISTRY.get(dist_name)
            if not dist_cls:
                raise ValueError(f"Unknown distribution '{dist_name}'")
            priors[name] = dist_cls(arg1, arg2, validate_args=False)
        return priors

    def chain_samples(self, discard=None, stride=None):
        """
        Retrieve MCMC samples with optional filtering.
        """
        discard = discard or int(2 * np.max(self.tau))
        stride = stride or int(0.5 * np.min(self.tau))
        samples = self.chain.get_chain(discard=discard, flat=True, thin=stride)

        self._chain_samples = samples
        return samples

    def save_priors(self, fn_out):
        """Save priors to a YAML file."""
        save_yaml(self.priors_specs, str(fn_out))

    def save_tau(self, fn_out):
        np.save(str(fn_out), self.tau)


class InferenceResults(MCMCResults, Specs):
    def __init__(self, chain, priors, tau, specs) -> None:
        """Store and process learning results."""
        MCMCResults.__init__(self, chain, priors, tau)
        Specs.__init__(self, specs)

        self.samples = None

    @property
    def chain_implicit_(self):
        return self.samples

    @property
    def chain_explicit_(self) -> np.ndarray:
        """Get the explicit parameters, computing them if necessary."""
        return self._compute_explicit_params()

    @property
    def labels_implicit_(self):
        """Generate labels for the parameters."""
        return self._get_labels('implicit')

    @property
    def labels_explicit_(self):
        """Generate labels for the explicit parameters."""
        return self._get_labels('explicit')

    @property
    def map(self):
        """Return maximum a posteriori (MAP) estimates for all parameters."""
        param_modes, nuisance_modes = [], []

        for param, label in zip(self.chain_implicit_.T, self.labels_implicit_):
            x = np.linspace(param.min(), param.max(), 1000)
            density = gaussian_kde(param)
            mode = x[np.argmax(density(x))]
            if "sigma_" in label:
                nuisance_modes.append(mode)
            else:
                param_modes.append(mode)

        q_implicit = self.total_charge - np.sum(param_modes * self.constraint_matrix)
        param_modes_all = np.insert(param_modes, self.implicit_param_pos, q_implicit)
        map_all = np.concat((param_modes_all, nuisance_modes))

        return dict(zip(self.labels_explicit_, map_all))

    def _get_labels(self, kind):
        if kind == 'implicit':
            param_labels = self.bounds_implicit.params.tolist()
        elif kind == 'explicit':
            param_labels = self.bounds_explicit.params.tolist()
        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'implicit' or 'explicit'.")

        nuisance_labels = [
            f'$\\sigma_{{\\mathrm{{{p.split()[1]}}}}}$'
            for p in list(self.priors.keys())
            if 'nuisance' in p
        ]

        return param_labels + nuisance_labels

    def quantiles(self, confidence=0.95):
        """
        Compute quantiles of the samples for a given confidence level.
        """
        q_lo = (1 - confidence) / 2
        q_hi = 1 - q_lo
        return np.quantile(self.chain_explicit_, [q_lo, 0.5, q_hi], axis=0)

    def get_chain(
        self, discard=None, stride=None, remove_defects=True, remove_sigma_outliers=True
    ) -> None:
        """
        Retrieve MCMC samples with optional filtering.
        """

        n = self.n_params_implicit
        chain_samples = self.chain_samples(discard, stride).copy()
        chain_samples[:, n:] = np.exp(chain_samples[:, n:])

        if remove_defects:
            bounds = self.bounds_implicit.values
            data = chain_samples[:, :self.n_params_implicit]
            mask = np.all(
                np.logical_and(data >= bounds[:, 0], data <= bounds[:, 1]),
                axis=1
            )
            chain_samples = chain_samples[mask]

        if remove_sigma_outliers:
            # Remove outliers based on nuisance parameters
            nuisance_params = chain_samples[:, self.n_params_implicit:]
            treshold = np.quantile(nuisance_params, 0.99, axis=0)
            mask = np.all(nuisance_params <= treshold, axis=1)
            chain_samples = chain_samples[mask]

        self.samples = chain_samples


    def sample_posterior(
        self,
        n_samples: int = 10,
        distribution: str = 'normal',
        confidence: float = 0.9,
        complete: bool = False,
        fn_out: str = None
    ) -> np.ndarray:
        """
        Draw samples from the posterior using
        either uniform or Laplace distribution.
        """
        lower, upper = (1 - confidence) / 2, 1 - (1 - confidence) / 2
        samples = self.chain_implicit_[:, :self.n_params_implicit]
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

        samples_out = np.empty((n_samples, len(self.bounds_implicit.params)))
        uniform_range = np.diff(confint, axis=0).ravel()

        i, attempts, max_attempts = 0, 0, n_samples * 1000  # Safety limit

        while i < n_samples and attempts < max_attempts:
            if distribution == 'uniform':
                random_values = np.random.rand(len(uniform_range)) * uniform_range
                sample = random_values + confint[0]
                sample.reshape(1, -1)
            else:
                if cov.size == 1:
                    sample = np.random.normal(mean, cov, size=1)[:, np.newaxis]
                else:
                    sample = np.random.multivariate_normal(mean, cov, size=1)

            is_within_confint = np.all(
                np.logical_and(sample >= confint[0], sample <= confint[1])
            )
            if is_within_confint and self.is_valid(sample):
                samples_out[i] = sample
                i += 1
            attempts += 1

        if i < n_samples:
            raise RuntimeError(
                "Failed to generate enough valid samples within max attempts."
            )

        if complete:
            q_explicit = np.sum(samples_out * self.constraint_matrix, axis=1)
            q_implicit = self.total_charge - q_explicit
            samples_out = np.insert(
                samples_out, self.implicit_param_pos, q_implicit, axis=1)

            param_names = self.bounds_explicit.params.tolist()
        else:
            param_names = self.bounds_implicit.params.tolist()

        if fn_out:
            if fn_out.endswith('.npy'):
                np.save(fn_out, samples_out)
            elif fn_out.endswith('.yaml'):
                samples_dict = dict(zip(param_names, samples_out.T))
                save_yaml(samples_dict, fn_out)
            else:
                raise ValueError('fn_out must end with .npy or .yaml')

        return samples_out


    def _compute_explicit_params(self) -> np.ndarray:
        """
        Compute the explicit parameters from the implicit parameters.
        """

        params = self.chain_implicit_[:, :self.n_params_implicit]
        q_explicit = np.sum(params * self.constraint_matrix, axis=1)
        q_implicit = (self.total_charge - q_explicit) / self.implicit_param_count
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

    def __init__(self, specs) -> None:
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

    def __call__(self, n) -> None:
        """Advance the sampler to skip the next n samples."""
        self.sampler.fast_forward(n)
        self.n_samples += n

    def generate(self, n, assign_hash=False):
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
        q_implicit = (
            self.total_charge - q_explicit
        ) / self.implicit_param_count
        mask = (self.lbi <= q_implicit) & (q_implicit <= self.ubi)

        # Filtered samples based on the mask
        filtered_samples = scaled_samples[mask]

        if assign_hash:
            # hash = secrets.token_hex(8)
            hash = str(self.n_samples)
            return hash, filtered_samples
        return filtered_samples
