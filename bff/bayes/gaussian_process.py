"""Local Gaussian processes and committees of them.

Notation follows common Gaussian-process usage (e.g. scikit-learn): ``X`` are
inputs (force-field parameter vectors, one per row), ``y`` outputs (QoI values,
one row per input), and ``y_ref`` the reference outputs the surrogate is
compared with during learning.
"""

from pathlib import Path
from typing import Sequence, TypeVar, Union

import numpy as np
import torch

from .effective_observations import effective_observations
from .kernels import gaussian_kernel
from .means import Mean, evaluate_mean
from .utils import smape

PathLike = Union[str, Path]
LGPCommitteeT = TypeVar("LGPCommitteeT", bound="LGPCommittee")


class LocalGaussianProcess:
    """Gaussian-process regression of all outputs with shared hyperparameters.

    The covariance is ``amplitude**2 * exp(-|x - x'|**2 / (2 lengthscales**2))``
    plus ``noise_variance`` on the diagonal; predictions are
    ``mean(X) + K(X, X_train) @ alpha`` with
    ``alpha = (K_train + noise_variance I)^-1 (y_train - mean(X_train))``.

    ``alpha`` is solved once in float64 on the CPU. :meth:`to` then moves the
    model to its working device and precision; do that once, before
    predicting many times.

    Parameters
    ----------
    X_train : torch.Tensor
        Training inputs, shape ``(n_train, n_inputs)``.
    y_train : torch.Tensor
        Training outputs, shape ``(n_train, n_outputs)``.
    mean : torch.Tensor or callable
        Prior mean: values of shape ``(n_outputs,)`` or a function of ``X``;
        see :mod:`bff.bayes.means`.
    lengthscales : torch.Tensor
        One kernel length scale per input, in input units.
    amplitude : float
        Kernel amplitude (signal standard deviation), in output units.
    noise_variance : float
        Variance added to the diagonal of the training covariance.
    """

    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        mean: Mean,
        lengthscales: torch.Tensor,
        amplitude: float,
        noise_variance: float,
    ) -> None:
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float64, device="cpu")

        self.X_train = tensor(X_train)
        self.y_train = tensor(y_train)
        self.mean = mean if callable(mean) else tensor(mean)
        self.lengthscales = tensor(lengthscales)
        self.amplitude = tensor(amplitude)
        self.noise_variance = tensor(noise_variance)

        n_train = len(self.X_train)
        K = gaussian_kernel(
            self.X_train, self.X_train, self.lengthscales, self.amplitude
        )
        K = K + self.noise_variance * torch.eye(n_train, dtype=K.dtype)
        L, info = torch.linalg.cholesky_ex(0.5 * (K + K.T))
        if info:
            raise ValueError(
                "The GP covariance matrix is not positive definite; the "
                "hyperparameters or training inputs are degenerate."
            )
        residuals = self.y_train - evaluate_mean(self.mean, self.X_train)
        self.alpha = torch.cholesky_solve(residuals, L)

    @property
    def device(self) -> torch.device:
        """Device of the model tensors."""
        return self.X_train.device

    @property
    def n_inputs(self) -> int:
        """Number of input parameters."""
        return self.X_train.shape[1]

    @property
    def n_outputs(self) -> int:
        """Number of output values."""
        return self.y_train.shape[1]

    @property
    def hyperparameters(self) -> dict[str, Union[np.ndarray, float]]:
        """``lengthscales`` (array), ``amplitude``, and ``noise_variance``."""
        return {
            "lengthscales": self.lengthscales.cpu().numpy(),
            "amplitude": self.amplitude.cpu().numpy().item(),
            "noise_variance": self.noise_variance.cpu().numpy().item(),
        }

    def to(
        self, device: str | torch.device, dtype: torch.dtype | None = None
    ) -> "LocalGaussianProcess":
        """Move the model, in place, to ``device`` and optionally ``dtype``."""
        names = ["X_train", "y_train", "lengthscales", "amplitude"]
        names += ["noise_variance", "alpha"]
        if not callable(self.mean):
            names.append("mean")
        for name in names:
            setattr(self, name, getattr(self, name).to(device, dtype))
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predicted outputs at ``X``, shape ``(n_samples, n_outputs)``, in the
        precision of ``X`` and on the device of the model."""
        X_model = X.to(self.X_train)
        K = gaussian_kernel(X_model, self.X_train, self.lengthscales, self.amplitude)
        return (evaluate_mean(self.mean, X_model) + K @ self.alpha).to(X.dtype)

    def state_dict(self) -> dict:
        """Constructor arguments (on the CPU), as saved in a committee file."""
        return {
            "X_train": self.X_train.cpu(),
            "y_train": self.y_train.cpu(),
            "mean": self.mean if callable(self.mean) else self.mean.cpu(),
            "lengthscales": self.lengthscales.cpu(),
            "amplitude": self.amplitude.cpu(),
            "noise_variance": self.noise_variance.cpu(),
        }

    def __repr__(self) -> str:
        return (
            f"LocalGaussianProcess(n_train={len(self.X_train)}, "
            f"n_inputs={self.n_inputs}, n_outputs={self.n_outputs})"
        )


class LGPCommittee:
    """Committee of local Gaussian processes that predicts one QoI.

    Parameters
    ----------
    members : list of LocalGaussianProcess
        The committee members, fitted with different hyperparameters.
    y_ref : np.ndarray
        Reference outputs the predictions are compared with in learning.
    n_curves : int
        Number of curves in ``y_ref``, which concatenates equally long curves.
    nuisance : float, optional
        Fixed standard deviation of the learning likelihood; ``None`` lets
        learning sample it.
    stochastic : bool
        Predict with one randomly chosen member instead of the average.
    dataset_fingerprint : str, optional
        Fingerprint of the QoI dataset the committee was fitted to.
    parameter_names : sequence of str, optional
        Names of the input columns.
    mean_spec : str, optional
        The mean specification it was fitted with (``None`` for a Python
        callable); see :mod:`bff.bayes.means`.
    """

    def __init__(
        self,
        members: list[LocalGaussianProcess],
        y_ref: np.ndarray,
        n_curves: int,
        nuisance: float | None = None,
        stochastic: bool = False,
        dataset_fingerprint: str | None = None,
        parameter_names: Sequence[str] | None = None,
        mean_spec: str | None = None,
    ) -> None:
        self.members = members
        self.mean_spec = mean_spec
        self.y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
        self.n_curves = int(n_curves)
        # Used in learning: the accepted deviation from y_ref, in output units.
        self.tolerance = 0.0
        self.nuisance = nuisance
        self.stochastic = stochastic
        self.dataset_fingerprint = dataset_fingerprint
        self.parameter_names = (
            None if parameter_names is None else tuple(parameter_names)
        )
        # Symmetric mean absolute percentage error on held-out samples.
        self.test_error: float | None = None

        if self.y_ref.size != self.members[0].n_outputs:
            raise ValueError("y_ref size does not match the surrogate output size.")
        if self.n_curves <= 0 or self.y_ref.size % self.n_curves != 0:
            raise ValueError(
                "'n_curves' must be a positive divisor of the y_ref size."
            )
        # Effective number of observations in y_ref, from its correlation length.
        self.n_eff = effective_observations(self.y_ref, self.n_curves)

    @property
    def n_members(self) -> int:
        """Number of committee members."""
        return len(self.members)

    @property
    def n_inputs(self) -> int:
        """Number of input parameters."""
        return self.members[0].n_inputs

    @property
    def n_outputs(self) -> int:
        """Number of output values (``n_curves`` curves of ``curve_length``)."""
        return self.members[0].n_outputs

    @property
    def curve_length(self) -> int:
        """Number of values per curve."""
        return int(self.y_ref.size // self.n_curves)

    def to(
        self, device: str | torch.device, dtype: torch.dtype | None = None
    ) -> "LGPCommittee":
        """Move all members, in place, to ``device`` and optionally ``dtype``."""
        for member in self.members:
            member.to(device, dtype)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Average prediction of the members (one member if ``stochastic``)."""
        if self.n_members > 1 and self.stochastic:
            index = torch.randint(self.n_members, (1,)).item()
            return self.members[index].predict(X)
        return torch.stack([member.predict(X) for member in self.members]).mean(0)

    def validate(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Store and return the test error (sMAPE, in percent)."""
        self.test_error = 100.0 * smape(y_test, self.predict(X_test))
        return self.test_error

    @classmethod
    def load(cls: type[LGPCommitteeT], fn: PathLike) -> LGPCommitteeT:
        """Read a ``.lgp`` file written by :meth:`write`."""
        state = torch.load(fn, weights_only=False, map_location="cpu")
        if "members" not in state:
            raise ValueError(
                f"{fn} was written by an older BFF version; refit it with "
                "bff fit-lgp."
            )
        committee = cls(
            members=[
                # Files of earlier versions also record a ``device``.
                LocalGaussianProcess(
                    **{k: v for k, v in member.items() if k != "device"}
                )
                for member in state["members"]
            ],
            y_ref=np.asarray(state["y_ref"], dtype=float),
            n_curves=int(state["n_curves"]),
            nuisance=state["nuisance"],
            stochastic=state["stochastic"],
            dataset_fingerprint=state["dataset_fingerprint"],
            parameter_names=state["parameter_names"],
            mean_spec=state["mean_spec"],
        )
        committee.test_error = state["test_error"]
        return committee

    def write(self, fn_out: PathLike) -> None:
        """Write the committee, with its members and reference, to one file."""
        torch.save(
            {
                "y_ref": self.y_ref.tolist(),
                "n_curves": self.n_curves,
                "nuisance": self.nuisance,
                "stochastic": self.stochastic,
                "test_error": self.test_error,
                "dataset_fingerprint": self.dataset_fingerprint,
                "parameter_names": (
                    None
                    if self.parameter_names is None
                    else list(self.parameter_names)
                ),
                "mean_spec": self.mean_spec,
                "members": [member.state_dict() for member in self.members],
            },
            fn_out,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_members={self.n_members}, "
            f"n_inputs={self.n_inputs}, n_outputs={self.n_outputs}, "
            f"test_error={self.test_error})"
        )
