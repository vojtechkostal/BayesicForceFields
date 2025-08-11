import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.distributions import Normal, Uniform
from scipy.stats.qmc import LatinHypercube

from .bayes.utils import initialize_backend
from .io.utils import (load_yaml, save_yaml, extract_tarball, load_md_files)
from .io.mdp import get_restraints
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


@dataclass
class TrajectoryData:
    rdf: dict
    hb: dict
    restr: dict


class TrainData:

    def __init__(self, train_dir: str | Path) -> None:
        """Initialize TrainData with training directories."""

        self.train_dir = Path(train_dir).resolve()
        self.fn_specs = None
        self.restraints = None
        self.fn_topol = None
        self.fn_coords = None
        self.samples = None
        self.scores = None
        self.settings = None
        self.qoi = None
        self.reference = None
        self.valid_hb = set()
        self.extract()

    def extract(self) -> None:
        """Extract and load data from training directories."""
        self.train_dir = self._process_directory(self.train_dir)
        fn_specs, mdps, topols, coords, sample_data = load_md_files(
            self.train_dir
        )

        self.fn_specs = fn_specs
        self.restraints = [get_restraints(mdp) for mdp in mdps]
        self.fn_topol = topols
        self.fn_coords = coords
        self.samples = sample_data

    def _process_directory(self, directory: Path) -> Path:
        """Process a directory or extract it if it's a tarball."""
        if directory.is_dir():
            return directory
        elif directory.suffix == ".gz":
            extracted_dir = directory.parent / directory.stem.replace(
                ".tar", ""
            )
            if not extracted_dir.exists():
                extract_tarball(directory)
            return extracted_dir
        else:
            raise ValueError(
                f"Invalid input: {directory} - must be a directory or "
                f".tar.gz file"
            )

    def _flatten_feature(self, sample: list[TrajectoryData]) -> list:
        rdf = np.array([g for s in sample for r, g in s.rdf.values()])
        hb = np.array([s.hb.get(name, 0) for s in sample for name in self.valid_hb])
        restr = np.array([prob for s in sample for _, prob in s.restr.values()])

        return rdf.flatten(), hb.flatten(), restr.flatten()

    @property
    def X(self) -> np.ndarray:
        """Return an array of parameters from all samples."""
        return np.array([sample["params"] for sample in self.samples.values()])

    @property
    def y(self) -> np.ndarray:
        if not self.qoi:
            raise ValueError("Features not loaded. Call load_features() first.")
        y_raw = [np.r_[self._flatten_feature(sample)] for sample in self.qoi]
        return np.stack(y_raw)

    @property
    def y_true(self) -> np.ndarray:
        """Return the true values of the features and corresponding slices."""
        if not self.reference:
            raise ValueError("Reference not loaded. Call load_reference() first.")

        features = self._flatten_feature(self.reference)

        return np.concatenate(features)

    @property
    def y_slices(self) -> dict[str, slice]:
        if not self.reference:
            raise ValueError("Reference not loaded. Call load_reference() first.")

        features = self._flatten_feature(self.reference)

        keys = ['rdf', 'hb', 'restr']
        lengths = list(map(len, features))
        offsets = np.cumsum([0] + lengths[:-1])

        return {
            key: slice(offset, offset + length)
            for key, offset, length, in zip(keys, offsets, lengths)
        }

    @property
    def observations(self):

        # Determine observation numbers
        n_rdf = sum([len(trj.rdf) for trj in self.reference])
        n_hb = len(self.valid_hb)
        n_restr = sum(len(trj.restr) for trj in self.reference)

        return {'rdf': n_rdf, 'hb': n_hb, 'restr': n_restr}

    @property
    def rdf_sigmoid_mean(self) -> np.ndarray:
        rs = [sigmoid(r) for trj in self.reference for r, g in trj.rdf.values()]
        return np.array(rs).flatten()

    @property
    def trajectories(self) -> list[list[str]]:
        """Return a list of trajectory file paths for all samples."""
        return [
            [self.train_dir / t for t in sample["fn_trj"]]
            for sample in self.samples.values()
        ]

    @property
    def hashes(self):
        return self.samples.keys()

    @property
    def n_samples(self):
        return len(self.samples)

    def load_features(self, features: str | Path | list, settings: dict = None) -> None:
        """Load features from a file or dictionary."""
        if isinstance(features, (str, Path)):
            # features = load_json(str(features))
            features = np.load(features, allow_pickle=True)
            settings = features['settings'].item()
            samples = features['samples'].item()
            if self.hashes != samples.keys():
                raise ValueError(
                    "Supplied features do not match the training samples."
                )
            qoi = [
                [TrajectoryData(**t) for t in sample]
                for sample in samples.values()
            ]
            # settings = features.get('settings', {})
        elif isinstance(features, list):
            qoi = features
            if not settings:
                raise ValueError(
                    "Settings must be provided when loading features from a list."
                )
        else:
            raise TypeError("Features must be a file path or list.")

        self.qoi = qoi
        self.settings = settings

    def load_reference(self, reference: list[TrajectoryData]) -> None:
        """Load reference data."""
        if not isinstance(reference, list):
            raise TypeError("Reference must be a list of TrajectoryData.")
        if not all(isinstance(r, TrajectoryData) for r in reference):
            raise TypeError("All items in reference must be TrajectoryData.")
        if not isinstance(reference, list):
            reference = [reference]

        self.reference = reference
        self.valid_hb = np.unique([k for trj in self.reference for k in trj.hb])

    def write_features(self, fn_out: str | Path) -> None:
        """Write features into a JSON file."""
        if not self.qoi:
            raise ValueError("No QoI to write.")

        qoi_hashed = dict(zip(self.hashes, self.qoi))
        data = {'settings': self.settings} | {'samples': qoi_hashed}
        fn_out = Path(fn_out).resolve()
        fn_out.parent.mkdir(parents=True, exist_ok=True)
        # save_json(data, fn_out)
        np.savez_compressed(fn_out, **data)

    def __repr__(self) -> str:
        """Return a string representation of the TrainData instance."""
        return (f"{self.n_samples} samples")


class Bounds:
    def __init__(self, bounds: dict) -> None:
        self.bounds = bounds

    @property
    def params(self):
        return np.array(list(self.bounds.keys()))

    @property
    def values(self):
        return np.array(list(self.bounds.values()))

    @property
    def lower(self):
        return self.values[:, 0]

    @property
    def upper(self):
        return self.values[:, 1]

    def __repr__(self) -> str:
        """Return a string representation of the Bounds object."""
        return f"{self.bounds}"


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
        return len(self.bounds_implicit.bounds)

    @property
    def n_params_explicit(self):
        """Number of parameters excluding the implicit charge."""
        return len(self.bounds_explicit.bounds)

    @property
    def bounds_implicit(self):
        bounds_copy = self._bounds.bounds.copy()
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

    def test_sample(self, params):
        params = np.asarray(params)
        for param, (lower, upper) in zip(params, self.bounds_implicit.values):
            if not (lower <= param <= upper):
                return False

        # Check the implicit charge
        lbi, ubi = self.implicit_param_bounds
        q_explicit = np.sum(params * self.constraint_matrix)
        q_implicit = self.total_charge - q_explicit

        return lbi <= q_implicit <= ubi

    def __repr__(self) -> str:
        """Return a string representation of the Specs instance."""
        return (
            f"Specs(atomtype_counts={self.atomtype_counts}, "
            f"mol_resname={self.mol_resname}, "
            f"implicit_atomtype={self.implicit_atomtype}, "
            f"total_charge={self.total_charge}, "
            f"bounds={self.bounds_explicit.bounds})"
        )


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


class OptimizationResults(MCMCResults, Specs):
    def __init__(self, chain, priors, tau, specs) -> None:
        """Store and process optimization results."""
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
            not hasattr(self.bounds_implicit, 'bounds') or
            not hasattr(self.bounds_implicit, 'values')
        ):
            raise AttributeError(
                "The 'bounds_implicit' attribute is not properly initialized "
                "or missing required attributes ('bounds', 'values')."
            )

        n_dim = len(self.bounds_implicit.bounds)
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

        if len(self.constraint_matrix) != len(self.bounds_implicit.bounds):
            raise ValueError(
                "The constraint matrix must match the\
                    number of implicit parameters."
            )

    def __call__(self, n) -> None:
        """Advance the sampler to skip the next n samples."""
        self.sampler.fast_forward(n)

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
            hash = secrets.token_hex(8)
            return hash, filtered_samples
        return filtered_samples
